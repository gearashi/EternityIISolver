"""Positive lookahead witnesses, independent constraints and saved-frontier scope."""
from itertools import product
import unittest
from unittest.mock import patch

import exact_dfs
from exact_model import check_solution, generated_problem
from exact_worker import dfs_profile_options
from test_exact_dfs import cyclic_unsat, exhaustive, reseal


class ProbeDomainOrderTests(unittest.TestCase):
    def test_detector_agrees_with_full_scan_for_every_previous_cell(self):
        # Empty, singleton (including a non-low bit), and multistate domains.
        for domains in product((0, 1, 8, 5), repeat=3):
            for previous in (None, 0, 1, 2):
                cell = exact_dfs._unresolved_cell(domains, previous)
                self.assertEqual(cell is None, all(mask.bit_count() == 1 for mask in domains))
                if cell is not None:
                    self.assertNotEqual(domains[cell].bit_count(), 1)

    def test_previous_blocker_is_rechecked_once_and_moves_between_trials(self):
        tested = []

        class CountedMask(int):
            def bit_count(self):
                tested.append(int(self))
                return super().bit_count()

        trials = (
            ((1, 2, 12, 8), [1, 2, 12], 2),
            ((1, 2, 12, 8), [12], 2),
            ((1, 3, 4, 8), [4, 1, 3], 1),
            ((1, 2, 4, 0), [2, 1, 4, 0], 3),
            ((1, 2, 4, 8), [8, 1, 2, 4], None),
        )
        previous = None
        for masks, expected_tests, expected_cell in trials:
            tested.clear()
            previous = exact_dfs._unresolved_cell(tuple(map(CountedMask, masks)), previous)
            self.assertEqual(tested, expected_tests)
            self.assertEqual(previous, expected_cell)


    def test_hint_is_local_to_a_search_and_keeps_sibling_results_and_counts(self):
        problem, _ = generated_problem(4, 3, 0)
        options = {'probe': True, 'probe_max_candidates': 1}
        hints_at_node = []
        original_probe = exact_dfs._Search.probe

        def observe(search, domains, matched):
            hints_at_node.append(search.probe_blocker)
            return original_probe(search, domains, matched)

        def run():
            return exact_dfs.solve(problem, seconds=5, seed=20261007, options=options)

        detector = exact_dfs._unresolved_cell
        with patch.object(exact_dfs, '_unresolved_cell', lambda domains, previous: detector(domains, None)):
            linear = run()
        # The fixture backtracks through a failed branch before finding a solution.
        self.assertGreater(linear['conflicts'], 0)
        self.assertTrue(check_solution(problem, linear['board']))
        with patch.object(exact_dfs._Search, 'probe', observe):
            for _ in range(2):
                hints_at_node.clear()
                cached = run()
                self.assertIsNone(hints_at_node[0])
                self.assertTrue(any(cell is not None for cell in hints_at_node[1:]))
                for name in ('outcome', 'board', *exact_dfs._COUNTERS):
                    self.assertEqual(cached[name], linear[name], name)

    def test_restored_search_starts_with_a_cold_hint_and_repeats_only_paid_work(self):
        problem, _ = generated_problem(4, 3, 0)
        options = {'probe': True, 'probe_max_candidates': 1}

        def run(**kwargs):
            return exact_dfs.solve(problem, seconds=5, seed=20261007, options=options, **kwargs)

        stop = False

        def after_third_probe(snapshot):
            nonlocal stop
            stop = snapshot['counters']['probe_checks'] >= 3

        partial = run(on_checkpoint=after_third_probe, checkpoint_interval=0, should_stop=lambda: stop)
        self.assertEqual(partial['outcome'], 'stopped')
        saved = partial['checkpoint']
        self.assertNotIn('probe_blocker', saved)
        phases = []
        original_probe = exact_dfs._Search.probe

        def observe(search, domains, matched):
            phases.append((search.restoring, search.probe_blocker))
            return original_probe(search, domains, matched)

        with patch.object(exact_dfs._Search, 'probe', observe):
            resumed = run(checkpoint=saved)
        self.assertTrue(any(restoring for restoring, _ in phases))
        self.assertTrue(all(cell is None for restoring, cell in phases if restoring))
        self.assertIsNone(next(cell for restoring, cell in phases if not restoring))
        fresh = run()
        self.assertEqual(resumed['board'], fresh['board'])
        self.assertTrue(check_solution(problem, resumed['board']))
        for name in ('branches', 'conflicts', 'probe_rejections'):
            self.assertEqual(resumed[name], fresh[name], name)
        self.assertEqual(resumed['probe_checks'], fresh['probe_checks'] + 1)

class ProbeWitnessTests(unittest.TestCase):
    def setUp(self):
        self.problem, _ = generated_problem(2, 3, 1)
        self.options = dfs_profile_options('full')

    def solve(self, **kwargs):
        return exact_dfs.solve(self.problem, seconds=5, seed=20261007,
                               options=self.options, **kwargs)

    def test_first_paid_probe_can_finish_without_a_branch(self):
        self.assertIsNotNone(exhaustive(self.problem))
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                self.options['adaptive'] = adaptive
                result = self.solve()
                self.assertEqual(result['outcome'], 'solved')
                self.assertEqual((result['probe_checks'], result['branches']), (1, 0))
                self.assertTrue(check_solution(self.problem, result['board']))
                if adaptive:
                    self.assertEqual(result['adaptive_policy']['probe']['calls'], 1)

    def test_forced_probe_uses_the_standard_validator(self):
        with patch.object(exact_dfs, 'check_solution', return_value=False) as checked:
            with self.assertRaisesRegex(AssertionError, 'failed independent exact-model validation'):
                self.solve()
        (problem, board), _ = checked.call_args
        self.assertIs(problem, self.problem)
        self.assertTrue(check_solution(problem, board))

    def test_impossible_cycle_retains_its_complete_negative_result(self):
        problem = cyclic_unsat()
        self.assertIsNone(exhaustive(problem))
        result = exact_dfs.solve(problem, seconds=5, options={'probe': True})
        self.assertEqual(result['outcome'], 'infeasible')
        self.assertTrue(result['coverage_verified'])
        self.assertIsNone(result['board'])
        self.assertEqual(result['probe_rejections'], result['probe_checks'])
        self.assertGreater(result['probe_checks'], 0)

    def test_interrupting_a_successful_probe_retries_the_pending_node(self):
        stop = False

        def after_first_probe(snapshot):
            nonlocal stop
            stop = snapshot['counters']['probe_checks'] >= 1

        interrupted = self.solve(on_checkpoint=after_first_probe, checkpoint_interval=0,
                                 should_stop=lambda: stop)
        self.assertEqual(interrupted['outcome'], 'stopped')
        checkpoint = interrupted['checkpoint']
        self.assertEqual(checkpoint['stack'], [{'phase': 'pending'}])
        resumed = self.solve(checkpoint=checkpoint)
        self.assertEqual(resumed['outcome'], 'solved')
        self.assertEqual(resumed['branches'], 0)
        self.assertEqual(resumed['probe_checks'], interrupted['probe_checks'] + 1)
        self.assertTrue(check_solution(self.problem, resumed['board']))

    def test_restore_cannot_return_a_witness_from_an_exhausted_ancestor(self):
        # v0.5.0's root split for this seed covers colors 1, 2, and 3 on edge 0.
        # Replay must reconstruct that split even though a root probe solves it.
        checkpoint = self.solve(should_stop=lambda: True)['checkpoint']
        checkpoint['stack'] = [dict(phase='branch', kind='edge', cell=0,
                                    candidates=[1, 2, 3], next=3,
                                    adaptive_before=checkpoint['adaptive_policy'])]
        with patch.object(exact_dfs, '_unresolved_cell', side_effect=AssertionError('scanning a replayed probe')):
            resumed = self.solve(checkpoint=reseal(checkpoint))
        self.assertEqual(resumed['outcome'], 'infeasible')
        self.assertEqual(resumed['proof_scope'], 'remaining_checkpoint_frontier')
        self.assertFalse(resumed['coverage_verified'])
        self.assertIsNone(resumed['board'])


if __name__ == '__main__':
    unittest.main()
