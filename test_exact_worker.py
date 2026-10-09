"""Exact-worker lifecycle tests: temporary state and injected CPU-only backends."""
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
import uuid
from unittest.mock import patch

import exact_worker
from exact_model import constraint_edges, from_bundle, make_problem, problem_hash
from process_control import RunLock
from validator import load_bundle, validate_board


class ExactWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()
        cls.problem_hash = problem_hash(from_bundle(cls.bundle))

    def setUp(self):
        self.temp_root = Path(tempfile.gettempdir()).resolve()
        self.home = self.temp_root / ('eternity-exact-worker-' + uuid.uuid4().hex)
        self.home.mkdir()
        self.runtime = self.home / 'runtime'
        self.runtime.mkdir()
        self.memory_patch = patch.object(exact_worker, 'peak_memory_mb', return_value=64.0)
        self.memory_patch.start()
        self.addCleanup(self.memory_patch.stop)
        self.addCleanup(self.clean_state)

    def clean_state(self):
        if self.home.resolve().parent != self.temp_root or not self.home.name.startswith('eternity-exact-worker-'):
            raise RuntimeError('Unexpected test cleanup path')
        shutil.rmtree(self.home)

    def read(self, name='status.json'):
        return json.loads((self.runtime / name).read_text(encoding='utf-8'))

    def write(self, name, value):
        (self.runtime / name).write_text(json.dumps(value), encoding='utf-8')

    def run_fake(self, fake, **kwargs):
        return exact_worker.run_task(self.home, solver_factory=lambda _: fake, **kwargs)

    @staticmethod
    def timeout(**kwargs):
        return {'outcome': 'timeout', 'board': None, 'branches': 17, 'conflicts': 9, 'max_depth': 7}

    def test_finite_run_uses_entire_five_clue_problem_and_saves_session(self):
        seen = {}
        def solver(**kwargs):
            seen.update(kwargs)
            kwargs['on_progress']({'phase': 'searching', 'branches': 10, 'conflicts': 3, 'max_depth': 5})
            return self.timeout()
        self.assertEqual(self.run_fake(solver, engine='sat', seconds=2, seed=123), 0)
        self.assertEqual(seen['problem'].size, 16)
        self.assertEqual(seen['problem'].fixed, self.bundle.fixed_clues)
        self.assertEqual(len(constraint_edges(seen['problem'])), 480)
        self.assertGreater(seen['seconds'], 0)
        self.assertLessEqual(seen['seconds'], 2)
        status = self.read()
        self.assertEqual(status['state'], 'finished_bounded_run')
        self.assertEqual(status['method'], 'exact')
        self.assertEqual(status['search_method'], 'exact-sat')
        self.assertEqual(status['compute_device'], 'CPU')
        self.assertEqual(status['best_score'], 466)
        self.assertEqual(status['branches'], 17)
        self.assertFalse(status['external_network_enabled'])
        self.assertFalse(status['library']['network_enabled'])
        self.assertFalse(self.read('exact-session.json')['native_search_state_resumed'])
        self.assertTrue(validate_board(self.read('best.json')['board'], self.bundle)['valid'])

    def test_preserved_stop_does_not_import_or_call_backend(self):
        (self.runtime / 'STOP').write_text('Stop', encoding='utf-8')
        def forbidden(_):
            self.fail('Stopped worker imported a backend')
        self.assertEqual(exact_worker.run_task(self.home, preserve_stop=True, solver_factory=forbidden), 0)
        self.assertEqual(self.read()['state'], 'stopped')
        self.assertTrue((self.runtime / 'STOP').exists())

    def test_new_run_clears_old_stop_and_default_budget_is_unbounded(self):
        (self.runtime / 'STOP').write_text('Old stop', encoding='utf-8')
        def solver(**kwargs):
            self.assertEqual(kwargs['seconds'], exact_worker.UNBOUNDED_SECONDS)
            self.assertFalse(kwargs['should_stop']())
            return self.timeout()
        self.assertEqual(self.run_fake(solver), 0)
        self.assertFalse((self.runtime / 'STOP').exists())
        self.assertIsNone(self.read('exact-session.json')['settings']['seconds'])

    def test_stop_marker_during_search_cannot_become_global_unsat(self):
        def solver(**kwargs):
            (self.runtime / 'STOP').write_text('Requested', encoding='utf-8')
            with patch.object(exact_worker.time, 'monotonic', return_value=10**12):
                self.assertTrue(kwargs['should_stop']())
            return {'outcome': 'infeasible', 'board': None}
        self.assertEqual(self.run_fake(solver), 0)
        self.assertEqual(self.read()['state'], 'stopped')
        self.assertEqual(self.read()['exact_outcome'], 'stopped')

    def test_stop_marker_written_just_before_return_is_observed(self):
        def solver(**kwargs):
            (self.runtime / 'STOP').write_text('Late stop', encoding='utf-8')
            return {'outcome': 'infeasible', 'board': None}
        self.assertEqual(self.run_fake(solver), 0)
        self.assertEqual(self.read()['state'], 'stopped')

    def test_soft_memory_limit_prevents_search(self):
        with patch.object(exact_worker, 'peak_memory_mb', return_value=2048.0):
            self.assertEqual(exact_worker.run_task(self.home, solver_factory=lambda _: self.fail('Unexpected search')), 0)
        self.assertEqual(self.read()['state'], 'stopped')
        self.assertEqual(self.read()['stop_reason'], 'memory_soft_limit')

    def test_memory_monitor_failure_reports_error(self):
        with patch.object(exact_worker, 'peak_memory_mb', side_effect=OSError('fixture failure')):
            self.assertEqual(self.run_fake(self.timeout), 1)
        self.assertEqual(self.read()['state'], 'error')

    def test_invalid_saved_best_falls_back_without_claiming_480(self):
        invalid = list(self.bundle.record_board)
        invalid[0] = invalid[1]
        self.write('best.json', {'board': invalid, 'score': 480})
        self.assertEqual(self.run_fake(self.timeout), 0)
        self.assertEqual(self.read()['best_score'], 466)
        self.assertEqual(self.read('best.json')['board'], list(self.bundle.record_board))
        self.assertIn('saved_best_warning', self.read())

    def test_backend_cannot_promote_a_466_board_as_solved(self):
        before = {'board': list(self.bundle.record_board), 'score': 466, 'sentinel': 'keep'}
        self.write('best.json', before)
        def solver(**kwargs):
            return {'outcome': 'solved', 'board': list(self.bundle.record_board)}
        self.assertEqual(self.run_fake(solver), 1)
        self.assertEqual(self.read()['state'], 'error')
        self.assertEqual(self.read()['best_score'], 466)
        self.assertEqual(self.read('best.json'), before)
        self.assertFalse((self.home / 'results').exists())

    def test_complete_validator_result_is_required_for_promotion(self):
        # A test-only validator witness exercises result persistence. No actual
        # 480 board is claimed or added to application state by this fixture.
        candidate = list(self.bundle.record_board)
        candidate[17], candidate[18] = candidate[18], candidate[17]
        def validation(board, bundle):
            if board == candidate:
                return {'valid': True, 'complete': True, 'score': 480}
            return validate_board(board, bundle)
        def solver(**kwargs):
            return {'outcome': 'solved', 'board': candidate, 'branches': 9}
        with patch.object(exact_worker, 'validate_board', side_effect=validation):
            self.assertEqual(self.run_fake(solver), 0)
        record = self.read('best.json')
        self.assertEqual(self.read()['state'], 'solved')
        self.assertEqual(record['board'], candidate)
        self.assertEqual(record['score'], 480)
        self.assertFalse(record['automatic_upload'])
        self.assertFalse(record['project_acceptance_confirmed'])
        self.assertEqual(len(list((self.home / 'results').glob('480-*.json'))), 1)

    def test_active_owner_status_and_stop_are_untouched(self):
        self.write('status.json', {'state': 'running', 'sentinel': 123})
        (self.runtime / 'STOP').write_text('Keep', encoding='utf-8')
        with RunLock(self.runtime / 'run.lock'):
            self.assertEqual(self.run_fake(self.timeout), 1)
        self.assertEqual(self.read(), {'state': 'running', 'sentinel': 123})
        self.assertEqual((self.runtime / 'STOP').read_text(), 'Keep')

    def test_backend_failure_releases_lock_and_monitor_thread(self):
        def solver(**kwargs):
            raise RuntimeError('Fixture backend failure')
        self.assertEqual(self.run_fake(solver), 1)
        self.assertEqual(self.read()['state'], 'error')
        with RunLock(self.runtime / 'run.lock'):
            pass
        self.assertFalse(any(t.name == 'eternity-exact-status' for t in threading.enumerate()))

    def test_fresh_and_resumed_negative_claims_are_distinct(self):
        def solver(*, checkpoint=None, on_checkpoint=None, **kwargs):
            return {'outcome': 'infeasible', 'board': None,
                    'proof_scope': 'remaining_checkpoint_frontier' if checkpoint else 'entire_supplied_problem'}
        self.assertEqual(self.run_fake(solver), 0)
        self.assertEqual(self.read()['state'], 'infeasible')
        self.write('exact-dfs-checkpoint.json', {'version': 1, 'problem_hash': self.problem_hash, 'seed': exact_worker.DEFAULT_SEED})
        self.assertEqual(self.run_fake(solver), 0)
        self.assertEqual(self.read()['state'], 'checkpoint_frontier_exhausted')
        self.assertFalse(self.read()['coverage_verified'])
        self.assertTrue(self.read()['resumed'])

    def test_incompatible_checkpoint_requires_explicit_fresh_restart(self):
        checkpoint = {'version': 1, 'problem_hash': 'incorrect', 'seed': exact_worker.DEFAULT_SEED}
        self.write('exact-dfs-checkpoint.json', checkpoint)
        def solver(*, checkpoint=None, on_checkpoint=None, **kwargs):
            self.assertIsNone(checkpoint)
            saved = {'version': 1, 'problem_hash': self.problem_hash, 'seed': exact_worker.DEFAULT_SEED, 'fixture': True}
            on_checkpoint(saved)
            return {'outcome': 'stopped', 'board': None, 'checkpoint': saved}
        self.assertEqual(self.run_fake(solver), 1)
        self.assertEqual(self.read('exact-dfs-checkpoint.json'), checkpoint)
        self.assertIn('--fresh-exact', self.read()['error'])
        self.assertEqual(self.run_fake(solver, fresh_exact=True), 0)
        self.assertEqual(self.read()['state'], 'stopped')
        self.assertTrue(self.read('exact-dfs-checkpoint.json')['fixture'])
        self.assertEqual(len(list(self.runtime.glob('exact-dfs-checkpoint.archived-*.json'))), 1)
        self.assertNotIn('checkpoint', self.read('exact-session.json')['result'])

    def test_real_dfs_tiny_checkpoint_stop_preserve_resume_and_exhaustion(self):
        import exact_dfs
        # This injection replaces only the backend's puzzle with a tiny cyclic
        # contradiction. Worker files remain temporary; no 16x16 DFS is run.
        faces = []
        for right, down in ((1, 2), (2, 1), (3, 4), (4, 3)):
            base = (0, right, down, 0)
            faces.extend(tuple(base[(s-r) % 4] for s in range(4)) for r in range(4))
        tiny = make_problem(2, faces)
        tiny_hash = problem_hash(tiny)
        calls = []
        def solver(*, problem, checkpoint=None, on_checkpoint=None, **kwargs):
            calls.append(checkpoint)
            local_stop = [False]
            first = len(calls) == 1
            def saved(value):
                on_checkpoint(value)
                if first and value['counters']['branches'] > 0:
                    (self.runtime / 'STOP').write_text('Tiny fixture stop', encoding='utf-8')
                    local_stop[0] = True
            result = exact_dfs.solve(tiny, seconds=2, seed=kwargs['seed'], checkpoint=checkpoint,
                                     on_checkpoint=saved, checkpoint_interval=0,
                                     should_stop=lambda: local_stop[0] or kwargs['should_stop'](),
                                     on_progress=kwargs['on_progress'])
            self.assertLess(result['elapsed_seconds'], 2)
            return result
        with patch.object(exact_worker, 'problem_hash', return_value=tiny_hash):
            self.assertEqual(self.run_fake(solver), 0)
            self.assertEqual(self.read()['state'], 'stopped')
            checkpoint = self.read('exact-dfs-checkpoint.json')
            self.assertGreater(checkpoint['counters']['branches'], 0)
            self.assertEqual(checkpoint['problem_hash'], tiny_hash)
            self.assertEqual(checkpoint['seed'], 20261007)
            self.assertEqual(self.run_fake(solver, preserve_stop=True), 0)
            self.assertEqual(len(calls), 1, 'Preserved stop must not enter the native backend')
            self.assertEqual(self.read('exact-dfs-checkpoint.json'), checkpoint)
            self.assertEqual(self.run_fake(solver), 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], checkpoint)
        self.assertEqual(self.read()['state'], 'checkpoint_frontier_exhausted')
        self.assertEqual(self.read()['best_score'], 466)
        self.assertFalse(self.read()['coverage_verified'])
        self.assertTrue(self.read()['resumed'])
        self.assertFalse((self.runtime / 'exact-dfs-checkpoint.json').exists())
        self.assertEqual(len(list(self.runtime.glob('exact-dfs-checkpoint.completed-*.json'))), 1)

    def test_real_dfs_rejects_corrupt_checkpoint_without_replacing_it(self):
        import exact_dfs
        from exact_model import generated_problem
        tiny, _ = generated_problem(2, 2, 71)
        checkpoint = exact_dfs.solve(tiny, seconds=0, seed=exact_worker.DEFAULT_SEED)['checkpoint']
        checkpoint['checksum'] = '0' * 64
        self.write('exact-dfs-checkpoint.json', checkpoint)
        def solver(*, problem, checkpoint=None, on_checkpoint=None, **kwargs):
            return exact_dfs.solve(tiny, seconds=2, seed=kwargs['seed'], checkpoint=checkpoint,
                                   on_checkpoint=on_checkpoint, should_stop=kwargs['should_stop'])
        with patch.object(exact_worker, 'problem_hash', return_value=problem_hash(tiny)):
            self.assertEqual(self.run_fake(solver), 1)
        self.assertEqual(self.read('exact-dfs-checkpoint.json'), checkpoint)
        self.assertEqual(self.read()['state'], 'error')
        self.assertIn('checksum', self.read()['error'])
        self.assertIn('--fresh-exact', self.read()['error'])

    def test_invalid_worker_settings_fail_before_state_changes(self):
        for kwargs in ({'engine': 'dfs', 'workers': 2}, {'engine': 'sat', 'workers': 2},
                       {'engine': 'cp-sat', 'workers': 5}, {'seconds': float('inf')}, {'seed': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_fake(self.timeout, **kwargs)
        self.assertEqual(list(self.runtime.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
