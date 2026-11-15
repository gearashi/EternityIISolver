"""Guided repair integration: tiny finite boards and injected worker backends."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dashboard_server
import exact_worker
import launcher
import repair_quality
import smart_repair
from exact_optimize import evaluate_board
from repair_checkpoint import CheckpointCodec
from test_exact_sat import tiny_problem
from test_region_failure_cache import legal_boards
from test_repair_checkpoint import initial_state


class GuidedSearchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.problem = tiny_problem(True)
        self.reference = max(legal_boards(self.problem))[1]

    def test_default_checkpoint_fingerprint_preserves_version_040(self):
        # Literal settings from the released v0.4.0 continuation contract.
        legacy = dict(seed=0, gpu_backend=None, gpu_batch_size=16, bulk_arcs=True,
                      diversify=True, plateau_rounds=8, region_repair_enabled=True,
                      region_every=8, region_seconds=.25, reuse_structure=True,
                      neutral_ties=False, pair_selection=False, adaptive_regions=False)
        self.assertEqual(smart_repair.checkpoint_settings(), legacy)
        self.assertEqual(smart_repair.checkpoint_settings(repair_quality='baseline'), legacy)
        self.assertEqual(CheckpointCodec(self.problem, legacy).fingerprint,
                         'b077574b76cbd7e32b826e1f4619794954c822f08a0566a76ccdeff91cd6e090')
        self.assertNotEqual(CheckpointCodec(self.problem, legacy).fingerprint,
                            CheckpointCodec(self.problem, smart_repair.checkpoint_settings(repair_quality='guided')).fingerprint)

    @staticmethod
    def normalized_state(payload):
        state = deepcopy(payload['state'])
        state.pop('elapsed_total')
        return state

    def test_guided_split_run_preserves_exact_continuation(self):
        options = dict(seconds=3, seed=83, plateau_rounds=1, region_repair_enabled=False,
                       repair_quality='guided', on_checkpoint=lambda payload: None)
        whole = smart_repair.solve(self.problem, self.reference, max_batches=6, **options)
        first = smart_repair.solve(self.problem, self.reference, max_batches=2, **options)
        resumed = smart_repair.solve(self.problem, self.reference, max_batches=4,
                                     checkpoint=first['checkpoint'], **options)
        self.assertEqual(whole['termination'], 'batch_limit')
        self.assertEqual(resumed['termination'], 'batch_limit')
        self.assertEqual(self.normalized_state(whole['checkpoint']), self.normalized_state(resumed['checkpoint']))
        self.assertEqual(whole['board'], resumed['board'])
        self.assertGreater(whole['quality_ejection_calls'], 0)
        self.assertEqual(whole['quality_candidate_evaluations'], resumed['quality_candidate_evaluations'])
        self.assertTrue(evaluate_board(self.problem, resumed['board'])['valid'])

    def test_plateau_calls_actual_controller_and_reports_actual_work(self):
        observed = []; base = repair_quality.QualityRepairController
        class TrackingController(base):
            def eject(controller, board, **kwargs):
                answer = super().eject(board, **kwargs)
                observed.append((list(board), deepcopy(answer)))
                return answer
        with patch.object(repair_quality, 'QualityRepairController', TrackingController):
            result = smart_repair.solve(self.problem, self.reference, seconds=3, seed=9,
                        plateau_rounds=1, max_batches=4, region_repair_enabled=False, repair_quality='guided')
        self.assertTrue(observed)
        self.assertEqual(result['quality_ejection_calls'], len(observed))
        self.assertEqual(result['quality_candidate_evaluations'], sum(r['candidate_evaluations'] for _, r in observed))
        self.assertEqual(result['quality_path_expansions'], sum(r['path_expansions'] for _, r in observed))
        for reference, answer in observed:
            tested = evaluate_board(self.problem, answer['best_true_board'])
            self.assertTrue(tested['valid']); self.assertEqual(tested['score'], answer['best_true_score'])
            self.assertEqual(answer['reference_score'], evaluate_board(self.problem, reference)['score'])
        self.assertGreaterEqual(result['score'], evaluate_board(self.problem, self.reference)['score'])

    def test_interrupted_guided_improvement_retains_independent_incumbent(self):
        initial, reference = min(legal_boards(self.problem)); stopped = [False]; improvements = []
        def improvement(value):
            improvements.append(deepcopy(value)); stopped[0] = True
        # Hold the assignment operator still to exercise the real guided plateau
        # route; the guided candidate and its full-board score remain genuine.
        def unchanged(problem, board, cells, **kwargs):
            return {'board': list(board), 'optimal': True}
        with patch.object(smart_repair, 'assignment_repair', side_effect=unchanged):
            result = smart_repair.solve(self.problem, reference, seconds=3, seed=41,
                        plateau_rounds=1, max_batches=2, region_repair_enabled=False, repair_quality='guided',
                        should_stop=lambda: stopped[0], on_improvement=improvement,
                        on_checkpoint=lambda payload: None)
        self.assertTrue(improvements); self.assertEqual(result['termination'], 'stopped')
        tested = evaluate_board(self.problem, result['board'])
        self.assertTrue(tested['valid']); self.assertGreater(tested['score'], initial)
        self.assertEqual(tested['score'], result['score'])
        self.assertEqual(result['checkpoint']['state']['best'], result['board'])
        self.assertEqual(result['checkpoint']['state']['best_score'], result['score'])

    def test_tabu_rejected_guided_board_does_not_skip_fallback_or_count_adoption(self):
        _, lower = min(legal_boards(self.problem)); best = self.reference
        settings = smart_repair.checkpoint_settings(seed=17, plateau_rounds=1,
                         region_repair_enabled=False, repair_quality='guided')
        state = initial_state(self.problem, best, 17)
        state.update(bank=[lower], bank_scores=[evaluate_board(self.problem, lower)['score']],
                     peaks=[evaluate_board(self.problem, best)['score']], recent=[[best, lower]])
        state['policy_state']['quality'] = repair_quality.QualityRepairController(self.problem).export_state()
        checkpoint = CheckpointCodec(self.problem, settings).encode(state)
        base = repair_quality.QualityRepairController
        class RepeatedProposal(base):
            def eject(controller, board, **kwargs):
                # A legal higher-score proposal whose previous population state
                # is tabu. This isolates scheduler acceptance from proposal work.
                return {'best_true_board': list(best), 'changed': True}
            def record_adoption(controller, before, after):
                raise AssertionError('Rejected tabu proposal was counted as adopted')
        def unchanged(problem, board, cells, **kwargs):
            return {'board': list(board), 'optimal': True}
        def fallback(diversifier, board, **kwargs):
            return {'board': list(board), 'changed': False}
        with patch.object(repair_quality, 'QualityRepairController', RepeatedProposal), \
             patch.object(smart_repair, 'assignment_repair', side_effect=unchanged), \
             patch('repair_diversify.Diversifier.perturb', autospec=True, side_effect=fallback) as perturb:
            result = smart_repair.solve(self.problem, best, seconds=2, seed=17, plateau_rounds=1,
                        region_repair_enabled=False, repair_quality='guided', max_batches=1, checkpoint=checkpoint)
        self.assertEqual(perturb.call_count, 1)
        self.assertEqual(result['tabu_rejections'], 1)
        self.assertEqual(result['quality_accepted_piece_changes'], 0)
        self.assertEqual(result['board'], best)

    def test_baseline_and_guided_checkpoints_reject_each_other(self):
        common = dict(seconds=2, seed=11, max_batches=1, plateau_rounds=1,
                      region_repair_enabled=False, on_checkpoint=lambda payload: None)
        for source, destination in (('baseline', 'guided'), ('guided', 'baseline')):
            first = smart_repair.solve(self.problem, self.reference, repair_quality=source, **common)
            with self.assertRaisesRegex(ValueError, 'settings mismatch'):
                smart_repair.solve(self.problem, self.reference, repair_quality=destination,
                                   checkpoint=first['checkpoint'], **common)


class GuidedInterfaceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='eternity-quality-integration-')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name); self.runtime = self.home / 'runtime'; self.runtime.mkdir()

    def test_launcher_and_dashboard_forward_only_matching_guidance(self):
        for method, engine in (('exact', 'matching'), ('exact', 'gpu-matching'), ('exact', 'dfs'),
                               ('exact', 'cp-sat'), ('exact', 'core-repair'), ('gpu', 'matching')):
            for policy in ('baseline', 'guided'):
                args = launcher.parser().parse_args(['run', '--method', method, '--exact-engine', engine,
                        '--repair-quality', policy, '--state-dir', str(self.home)])
                direct = launcher.run_arguments(args)
                settings = dashboard_server.validate_settings(dict(method=method, exact_engine=engine, repair_quality=policy))
                dashboard = dashboard_server.worker_command(self.home, 8765, settings)
                expected = method == 'exact' and engine in ('matching', 'gpu-matching') and policy == 'guided'
                for command in (direct, dashboard):
                    self.assertEqual('--repair-quality' in command, expected)
                    if expected: self.assertEqual(command[command.index('--repair-quality') + 1], 'guided')
        with self.assertRaises(ValueError): dashboard_server.validate_settings({'repair_quality': 'unknown'})

    def test_worker_forwards_guided_policy_and_real_counter_names(self):
        counters = {'quality_ejection_calls': 2, 'quality_recombination_calls': 1,
                    'quality_candidate_evaluations': 17, 'quality_heuristic_state_evaluations': 29,
                    'quality_path_expansions': 31, 'quality_native_branches': 37,
                    'quality_native_conflicts': 3, 'quality_table_rows': 42,
                    'quality_accepted_piece_changes': 4, 'quality_accepted_board_improvements': 1,
                    'quality_true_score_gain': 1, 'quality_stopped_calls': 0, 'quality_deterministic_work': .01,
                    'quality_returned_piece_changes': 7, 'quality_returned_board_improvements': 2,
                    'quality_adoptions': 1, 'region_cache_hits': 2}
        expected_keys = set(repair_quality.QualityRepairController(tiny_problem(True)).stats())
        self.assertEqual(set(counters) - {'region_cache_hits'}, expected_keys)
        seen = {}
        def fake(**kwargs):
            seen.update(kwargs)
            kwargs['on_progress'](dict(phase='matching_and_cycle_repair', **counters))
            return {'outcome': 'timeout', 'board': None, **counters}
        for engine in ('matching', 'gpu-matching'):
            with patch.object(exact_worker, 'peak_memory_mb', return_value=32.), \
                 patch.object(smart_repair, 'cached_parents', return_value=[]):
                code = exact_worker.run_task(self.home, engine=engine, seconds=2, repair_quality='guided',
                                            solver_factory=lambda engine: fake)
            self.assertEqual(code, 0)
            self.assertEqual(seen['repair_quality'], 'guided')
            status = json.loads((self.runtime / 'status.json').read_text())
            session = json.loads((self.runtime / 'exact-session.json').read_text())
            self.assertEqual(status['repair_quality'], 'guided')
            self.assertEqual(session['settings']['repair_quality'], 'guided')
            for key, value in counters.items(): self.assertEqual(status.get(key), value, key)
            self.assertFalse(status['external_network_enabled'])
            self.assertNotEqual(status['state'], 'solved')


if __name__ == '__main__': unittest.main()
