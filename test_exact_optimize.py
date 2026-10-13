"""Tiny bounded tests for full-board optimization and genuine assumption cores.

No full Eternity II benchmark, network, GPU or runtime-state changes.
"""
from dataclasses import replace
import importlib.util
import itertools
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from exact_model import check_solution, generated_problem
from exact_optimize import core_guided_repair, evaluate_board, optimize
from test_exact_sat import tiny_problem
import exact_sat


def legal_boards(problem):
    for board in itertools.product(*problem.domains):
        checked = evaluate_board(problem, board)
        if checked['valid']:
            yield checked['score'], list(board)


class WaitingCpSolver:
    def __init__(self, status):
        self.parameters = SimpleNamespace()
        self.entered = threading.Event()
        self.stopped = threading.Event()
        self.status = status
        self.num_branches = 9
        self.num_conflicts = 2

    def solve(self, model, callback):
        self.entered.set()
        if not self.stopped.wait(1):
            raise AssertionError('Optimizer did not cancel the native search')
        return self.status

    def stop_search(self):
        self.stopped.set()

    def status_name(self, status):
        return 'UNKNOWN'


@unittest.skipUnless(importlib.util.find_spec('ortools'), 'ortools unavailable')
class ScoreOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ortools.sat.python import cp_model
        cls.cp_model = cp_model

    def test_matches_exhaustive_tiny_maximum_without_freezing_reference(self):
        for impossible in (False, True):
            with self.subTest(impossible=impossible):
                problem = tiny_problem(impossible)
                candidates = sorted(legal_boards(problem))
                baseline, reference = candidates[0]
                optimum = candidates[-1][0]
                self.assertLess(baseline, optimum)
                improvements = []
                result = optimize(problem, reference, seconds=2, on_improvement=improvements.append)
                self.assertEqual(result['score'], optimum, result)
                self.assertTrue(evaluate_board(problem, result['board'])['valid'])
                self.assertEqual(result['outcome'], 'improved' if impossible else 'solved')
                self.assertNotEqual(result['board'], reference)
                self.assertTrue(improvements)
                self.assertTrue(all(evaluate_board(problem, item['board'])['score'] == item['score']
                                    for item in improvements))
                self.assertEqual([item['score'] for item in improvements],
                                 sorted(set(item['score'] for item in improvements)))

    def test_infeasible_score_target_keeps_validated_baseline(self):
        problem = tiny_problem(True)
        baseline, reference = max(legal_boards(problem))
        self.assertEqual(baseline, 3)
        result = optimize(problem, reference, seconds=2)
        self.assertEqual(result['outcome'], 'infeasible_for_target')
        self.assertEqual(result['score'], 3)
        self.assertEqual(result['target_score'], 4)
        self.assertEqual(result['board'], reference)
        self.assertIn('supplied-domains', result['proof_scope'])

    def test_fixed_placement_rotation_and_frame_are_hard(self):
        problem = tiny_problem(fixed={0: 0})
        reference = min(legal_boards(problem))[1]
        result = optimize(problem, reference, seconds=2)
        self.assertEqual(result['outcome'], 'solved')
        self.assertEqual(result['board'][0], 0)
        self.assertTrue(check_solution(problem, result['board']))
        symmetric, reference = generated_problem(1, 1, 3)
        symmetric = replace(symmetric, fixed={0: 3})
        self.assertEqual(optimize(symmetric, [3], seconds=2)['board'], [3])
        with self.assertRaises(ValueError):
            optimize(symmetric, [0], seconds=2)

    def test_partial_grid_problem_is_rejected(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        with self.assertRaises(ValueError):
            optimize(replace(problem, edges=()), reference, seconds=2)
        with self.assertRaises(ValueError):
            core_guided_repair(replace(problem, edges=()), reference, seconds=2)

    def test_duplicate_piece_and_illegal_clue_reference_rejected(self):
        problem = tiny_problem(fixed={0: 0})
        reference = [0, 4, 8, 12]
        duplicate = list(reference)
        duplicate[1] = next(state for state in problem.domains[1] if state // 4 == 0)
        self.assertFalse(evaluate_board(problem, duplicate)['valid'])
        with self.assertRaises(ValueError):
            optimize(problem, duplicate, seconds=2)

    def test_zero_budget_and_existing_stop_preserve_reference_without_native_search(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        with patch.object(self.cp_model, 'CpSolver', side_effect=AssertionError('Native solver should not start')):
            for kwargs, expected in (({'seconds': 0}, 'timeout'),
                                     ({'seconds': 2, 'should_stop': lambda: True}, 'stopped')):
                result = optimize(problem, reference, **kwargs)
                self.assertEqual(result['outcome'], expected)
                self.assertEqual(result['board'], reference)
                self.assertTrue(evaluate_board(problem, result['board'])['valid'])

    def test_build_and_search_share_one_deadline(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        original = self.cp_model.CpModel.add_allowed_assignments
        def slow_table(model, variables, rows):
            time.sleep(.03)
            return original(model, variables, rows)
        with patch.object(self.cp_model.CpModel, 'add_allowed_assignments', slow_table), \
             patch.object(self.cp_model, 'CpSolver', side_effect=AssertionError('No second search budget')):
            result = optimize(problem, reference, seconds=.01)
        self.assertEqual(result['outcome'], 'timeout')
        self.assertIn('building', result['reason'])
        self.assertEqual(result['board'], reference)

    def test_native_stop_and_timeout_are_not_score_infeasibility(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        for stop in (False, True):
            fake = WaitingCpSolver(self.cp_model.UNKNOWN)
            with self.subTest(stop=stop), patch.object(self.cp_model, 'CpSolver', return_value=fake):
                result = optimize(problem, reference, seconds=2 if stop else .08,
                                  should_stop=fake.entered.is_set if stop else None)
            self.assertEqual(result['outcome'], 'stopped' if stop else 'timeout')
            self.assertEqual(result['board'], reference)
            self.assertEqual(result['branches'], 9)
            self.assertTrue(fake.stopped.is_set())
            self.assertFalse(any(thread.name == 'eternity-optimize-cancel' for thread in threading.enumerate()))

    def test_independent_callback_validation_rejects_bad_native_witness(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        calls = 0
        def validate(puzzle, board):
            nonlocal calls
            calls += 1
            return evaluate_board(puzzle, board) if calls == 1 else {'valid': False, 'score': None}
        with patch('exact_optimize.evaluate_board', side_effect=validate), self.assertRaises(RuntimeError):
            optimize(problem, reference, seconds=2)

    def test_parameters_are_checked(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        for kwargs in ({'seconds': -1}, {'seconds': True}, {'seconds': float('nan')},
                       {'seed': -1}, {'workers': 0}, {'workers': True}, {'target_score': 5},
                       {'on_improvement': 5}, {'should_stop': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                optimize(problem, reference, **kwargs)


@unittest.skipUnless(importlib.util.find_spec('pysat'), 'python-sat unavailable')
class CoreRepairTests(unittest.TestCase):
    def test_core_unfreezing_reaches_tiny_solution_and_keeps_clue(self):
        problem = tiny_problem(fixed={0: 0})
        reference = min(legal_boards(problem))[1]
        self.assertFalse(check_solution(problem, reference))
        improvements = []
        result = core_guided_repair(problem, reference, seconds=2, free_cells=[],
                                    on_improvement=improvements.append)
        self.assertEqual(result['outcome'], 'solved', result)
        self.assertTrue(check_solution(problem, result['board']))
        self.assertEqual(result['board'][0], 0)
        self.assertTrue(result['cores'])
        self.assertEqual(result['model_builds'], 1)
        self.assertEqual(result['assumption_queries'], result['rounds'])
        self.assertGreater(result['rounds'], 1)
        self.assertTrue(result['incremental_sat'])
        self.assertNotIn(0, result['free_cells'])
        self.assertTrue(all(0 not in core for core in result['cores']))
        self.assertEqual(len(improvements), 1)

    def test_timeout_never_creates_or_uses_a_core(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        unknown = {'outcome': 'timeout', 'branches': 0, 'conflicts': 0, 'build_seconds': 0,
                   'core_cells': [0], 'board': None, 'model_builds': 1}
        with patch.object(exact_sat.AssumptionSession, 'solve', return_value=unknown):
            result = core_guided_repair(problem, reference, seconds=2, free_cells=[])
        self.assertEqual(result['outcome'], 'timeout')
        self.assertEqual(result['cores'], [])
        self.assertEqual(result['free_cells'], [])
        self.assertEqual(result['board'], reference)

    def test_base_infeasibility_scope_and_round_limit(self):
        impossible = tiny_problem(True)
        reference = min(legal_boards(impossible))[1]
        result = core_guided_repair(impossible, reference, seconds=2, free_cells=range(4))
        self.assertEqual(result['outcome'], 'infeasible_for_target')
        self.assertEqual(result['cores'], [])
        self.assertIn('supplied-domains', result['proof_scope'])
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        bounded = core_guided_repair(problem, reference, seconds=2, free_cells=[], max_rounds=1)
        self.assertEqual(bounded['outcome'], 'timeout')
        self.assertEqual(bounded['termination'], 'round_limit')
        self.assertEqual(bounded['board'], reference)


if __name__ == '__main__':
    unittest.main()
