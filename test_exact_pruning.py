"""Tiny correctness tests for the isolated Hall-pruning ablation harness."""
import unittest

import exact_dfs
from exact_model import ExactProblem, check_solution, generated_problem
from scripts.benchmark_pruning import hall_problem, solve_variant
from test_exact_dfs import cyclic_unsat, exhaustive


class ExactPruningTests(unittest.TestCase):
    def test_both_variants_agree_with_tiny_exhaustive_cases(self):
        cases = [cyclic_unsat()]
        for seed in range(8):
            problem, answer = generated_problem(2, 3, seed)
            cases.append(problem)
            domains = list(problem.domains)
            domains[0] = tuple(s for s in domains[0] if s != answer[0])
            cases.append(ExactProblem(2, problem.faces, {}, tuple(domains)))
        for number, problem in enumerate(cases):
            expected = 'solved' if exhaustive(problem) is not None else 'infeasible'
            for variant in ('hall', 'no-hall'):
                with self.subTest(case=number, variant=variant):
                    result = solve_variant(problem, variant, seconds=2)
                    self.assertEqual(result['outcome'], expected)
                    if expected == 'solved':
                        self.assertTrue(check_solution(problem, result['board']))
                        self.assertTrue(result['independently_validated'])

    def test_hall_fixture_distinguishes_the_pruning_rule(self):
        problem = hall_problem(4)
        full = solve_variant(problem, 'hall', seconds=2)
        ablated = solve_variant(problem, 'no-hall', seconds=2)
        self.assertEqual(full['outcome'], 'infeasible')
        self.assertEqual(ablated['outcome'], 'infeasible')
        self.assertEqual(full['branches'], 0)
        self.assertGreater(full['matching_failures'], 0)
        self.assertGreater(ablated['branches'], full['branches'])
        self.assertEqual(ablated['matching_checks'], 0)
        self.assertEqual(ablated['matching_seconds'], 0)
        self.assertGreater(ablated['matching_skips'], 0)
        self.assertEqual(full['root_before'], ablated['root_before'])
        self.assertEqual(full['root_propagation_outcome'], 'contradiction')
        self.assertEqual(ablated['root_propagation_outcome'], 'consistent')

    def test_instrumentation_restores_original_solver_on_stop_or_error(self):
        original = exact_dfs._Search
        problem, _ = generated_problem(2, 3, 5)
        stopped = solve_variant(problem, 'no-hall', seconds=2, should_stop=lambda: True)
        self.assertEqual(stopped['outcome'], 'stopped')
        self.assertIs(exact_dfs._Search, original)
        with self.assertRaises(ValueError):
            solve_variant(problem, 'hall', seconds=-1)
        self.assertIs(exact_dfs._Search, original)
        with self.assertRaises(ValueError):
            solve_variant(problem, 'unknown')

    def test_root_candidate_observations_are_bounded_counts(self):
        problem, _ = generated_problem(3, 5, 32, clue_count=1)
        for variant in ('hall', 'no-hall'):
            result = solve_variant(problem, variant, seconds=2)
            self.assertEqual(result['outcome'], 'solved')
            self.assertLessEqual(result['root_after']['state_candidates'], result['root_before']['state_candidates'])
            self.assertLessEqual(result['root_after']['piece_candidates'], result['root_before']['piece_candidates'])
            self.assertGreaterEqual(result['matching_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
