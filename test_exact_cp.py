"""Small exactness and bounded/cancellable execution checks; no GPU/network."""
from dataclasses import replace
import itertools
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import exact_cp
from exact_model import ExactProblem, check_solution, generated_problem


class FakeSolver:
    def __init__(self, status, *, wait=False):
        self.parameters = SimpleNamespace()
        self.status = status
        self.wait = wait
        self.entered = threading.Event()
        self.stopped = threading.Event()
        self.num_branches = 17
        self.num_conflicts = 3

    def solve(self, model):
        self.entered.set()
        if self.wait:
            if not self.stopped.wait(1):
                raise AssertionError('Cooperative solver cancellation was not delivered')
        return self.status

    def stop_search(self):
        self.stopped.set()

    def status_name(self, status):
        return 'UNKNOWN'


class ExactCpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            raise unittest.SkipTest('Optional ortools dependency is not installed')
        cls.cp_model = cp_model

    def test_known_solvable_generated_puzzles(self):
        for size, colors, clues in ((1, 1, 0), (2, 3, 0), (3, 4, 1)):
            with self.subTest(size=size):
                problem, known = generated_problem(size, colors, 20261009, clue_count=clues)
                self.assertTrue(check_solution(problem, known))
                result = exact_cp.solve(problem, seconds=3, seed=42, workers=1)
                self.assertEqual(result['outcome'], 'solved', result)
                self.assertTrue(check_solution(problem, result['board']))
                self.assertLessEqual(result['build_seconds'], result['elapsed_seconds'])
                self.assertGreaterEqual(result['branches'], 0)

    def test_fixed_symmetric_rotation_is_preserved_exactly(self):
        # All four rotations project to the same (piece,N,E,S,W) tuple. The
        # explicit fixed state must still select rotation3, not a default0.
        problem = ExactProblem(1, ((0, 0, 0, 0),) * 4, {0: 3}, ((0, 1, 2, 3),))
        result = exact_cp.solve(problem, seconds=2)
        self.assertEqual(result['outcome'], 'solved')
        self.assertEqual(result['board'], [3])
        self.assertTrue(check_solution(problem, result['board']))
        restricted = replace(problem, fixed={}, domains=((1, 3),))
        self.assertIn(exact_cp.solve(restricted, seconds=2)['board'][0], (1, 3))

    def test_all_different_is_required_even_without_edges(self):
        problem, board = generated_problem(2, 3, 7)
        duplicate = list(board)
        duplicate[1] = next(state for state in problem.domains[1] if state // 4 == board[0] // 4)
        problem = replace(problem, domains=tuple((state,) for state in duplicate), edges=())
        self.assertFalse(check_solution(problem, duplicate))
        result = exact_cp.solve(problem, seconds=2)
        self.assertEqual(result['outcome'], 'infeasible')
        self.assertIsNone(result['board'])
        self.assertEqual(result['solver_status'], 'INFEASIBLE')

    def test_explicit_edge_subset_is_respected(self):
        problem, board = generated_problem(2, 11, 42)
        for a, b in itertools.combinations(range(4), 2):
            candidate = list(board)
            candidate[a] = next(state for state in problem.domains[a] if state // 4 == board[b] // 4)
            candidate[b] = next(state for state in problem.domains[b] if state // 4 == board[a] // 4)
            if not check_solution(problem, candidate):
                break
        else:
            self.fail('Expected a mismatched but frame-legal fixture')
        frozen = replace(problem, domains=tuple((state,) for state in candidate))
        relaxed = replace(frozen, edges=())
        self.assertTrue(check_solution(relaxed, candidate))
        good = exact_cp.solve(relaxed, seconds=2)
        self.assertEqual(good['outcome'], 'solved')
        self.assertEqual(good['board'], candidate)
        self.assertEqual(exact_cp.solve(frozen, seconds=2)['outcome'], 'infeasible')

    def test_empty_domain_and_fixed_conflict_are_infeasible(self):
        problem, _ = generated_problem(1, 1, 4)
        for changed in (replace(problem, domains=((),)), replace(problem, fixed={0: 2}, domains=((0, 1),))):
            with self.subTest(problem=changed):
                result = exact_cp.solve(changed, seconds=2)
                self.assertEqual(result['outcome'], 'infeasible')
                self.assertIsNone(result['board'])

    def test_zero_budget_and_immediate_stop_do_not_start_native_solver(self):
        problem, _ = generated_problem(2, 2, 4)
        with patch.object(self.cp_model, 'CpSolver', side_effect=AssertionError('Solver must not start')):
            self.assertEqual(exact_cp.solve(problem, seconds=0)['outcome'], 'timeout')
            self.assertEqual(exact_cp.solve(problem, seconds=1, should_stop=lambda: True)['outcome'], 'stopped')

    def test_model_construction_consumes_the_same_wall_budget(self):
        problem, _ = generated_problem(2, 2, 4)
        original = self.cp_model.CpModel.add_allowed_assignments
        def slow_table(model, variables, tuples):
            time.sleep(.03)  # Represent one indivisible native construction call.
            return original(model, variables, tuples)
        with patch.object(self.cp_model.CpModel, 'add_allowed_assignments', slow_table), \
             patch.object(self.cp_model, 'CpSolver', side_effect=AssertionError('No second search budget')):
            result = exact_cp.solve(problem, seconds=.01)
        self.assertEqual(result['outcome'], 'timeout')
        self.assertGreaterEqual(result['build_seconds'], .03)
        self.assertIn('building', result['reason'])

    def test_stop_during_construction_prevents_native_search(self):
        problem, _ = generated_problem(2, 2, 4)
        stop = threading.Event()
        original = self.cp_model.CpModel.add_allowed_assignments
        def stop_after_table(model, variables, tuples):
            result = original(model, variables, tuples)
            stop.set()
            return result
        with patch.object(self.cp_model.CpModel, 'add_allowed_assignments', stop_after_table), \
             patch.object(self.cp_model, 'CpSolver', side_effect=AssertionError('Stopped during build')):
            result = exact_cp.solve(problem, seconds=1, should_stop=stop.is_set)
        self.assertEqual(result['outcome'], 'stopped')

    def test_stop_during_search_is_delivered_cooperatively(self):
        problem, _ = generated_problem(2, 2, 4)
        fake = FakeSolver(self.cp_model.UNKNOWN, wait=True)
        with patch.object(self.cp_model, 'CpSolver', return_value=fake):
            result = exact_cp.solve(problem, seconds=2, should_stop=fake.entered.is_set)
        self.assertEqual(result['outcome'], 'stopped')
        self.assertTrue(fake.stopped.is_set())
        self.assertIsNone(result['board'])
        self.assertEqual((result['branches'], result['conflicts']), (17, 3))
        self.assertLess(result['elapsed_seconds'], .5)

    def test_unknown_is_never_reported_as_infeasible(self):
        problem, _ = generated_problem(2, 2, 4)
        fake = FakeSolver(self.cp_model.UNKNOWN)
        with patch.object(self.cp_model, 'CpSolver', return_value=fake):
            result = exact_cp.solve(problem, seconds=2)
        self.assertEqual(result['outcome'], 'timeout')
        self.assertEqual(result['solver_status'], 'UNKNOWN')
        self.assertIsNone(result['board'])

    def test_progress_and_parameter_validation(self):
        problem, _ = generated_problem(1, 1, 4)
        updates = []
        result = exact_cp.solve(problem, seconds=2, on_progress=updates.append)
        self.assertEqual(result['outcome'], 'solved')
        self.assertEqual(updates[0]['phase'], 'building')
        self.assertEqual(updates[-1]['phase'], 'finished')
        self.assertEqual(updates[-1]['outcome'], 'solved')
        for kwargs in ({'seconds': -1}, {'seconds': float('nan')}, {'seconds': float('inf')},
                       {'workers': 0}, {'workers': True}, {'seed': -1}, {'seed': 2**32}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                exact_cp.solve(problem, **kwargs)


if __name__ == '__main__':
    unittest.main()
