"""Positive lookahead witnesses, independent constraints and saved-frontier scope."""
import unittest
from unittest.mock import patch

import exact_dfs
from exact_model import check_solution, generated_problem
from exact_worker import dfs_profile_options
from test_exact_dfs import cyclic_unsat, exhaustive, reseal


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
        resumed = self.solve(checkpoint=reseal(checkpoint))
        self.assertEqual(resumed['outcome'], 'infeasible')
        self.assertEqual(resumed['proof_scope'], 'remaining_checkpoint_frontier')
        self.assertFalse(resumed['coverage_verified'])
        self.assertIsNone(resumed['board'])


if __name__ == '__main__':
    unittest.main()
