"""Exhaustive tiny oracles and bounded lifecycle checks for regional repair."""
from dataclasses import replace
import importlib.util
import itertools
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from exact_model import generated_problem, grid_edges
from exact_optimize import evaluate_board
from repair_regions import region_repair, select_region
from test_exact_sat import tiny_problem


def legal_boards(problem):
    return [(checked['score'], list(board)) for board in itertools.product(*problem.domains)
            if (checked := evaluate_board(problem, board))['valid']]


def region_oracle(problem, board, cells):
    return max((score, candidate) for score, candidate in legal_boards(problem)
               if all(candidate[c] == board[c] for c in range(len(board)) if c not in cells))


class RegionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.problem, self.board = generated_problem(4, 7, 84, clue_count=1)
        # Rotate a mutable interior piece without changing any physical identity.
        cell = next(c for c in (5, 6, 9, 10) if c not in self.problem.fixed)
        self.board[cell] = (self.board[cell] // 4) * 4 + (self.board[cell] + 1) % 4
        self.assertTrue(evaluate_board(self.problem, self.board)['valid'])

    def test_core_connected_and_anchored_at_mismatch(self):
        core = select_region(self.problem, self.board, seed=9, region_size=5, donor_count=0)
        self.assertEqual(len(core), 5)
        self.assertFalse(set(core) & self.problem.fixed.keys())
        visited = {core[0]}
        while True:
            expanded = visited | {b for a, b, _, _ in grid_edges(4) if a in visited and b in core} | {
                a for a, b, _, _ in grid_edges(4) if b in visited and a in core}
            if expanded == visited:
                break
            visited = expanded
        self.assertEqual(visited, set(core))
        self.assertTrue(any((a in core or b in core) and self.problem.faces[self.board[a]][sa] !=
                            self.problem.faces[self.board[b]][sb] for a, b, sa, sb in grid_edges(4)))

    def test_donors_are_distinct_compatible_and_deterministic(self):
        kwargs = dict(seed=53, region_size=3)
        core = select_region(self.problem, self.board, donor_count=0, **kwargs)
        cells = select_region(self.problem, self.board, donor_count=4, **kwargs)
        self.assertEqual(cells, select_region(self.problem, self.board, donor_count=4, **kwargs))
        self.assertTrue(set(core) < set(cells))
        self.assertLessEqual(len(cells), 7)
        self.assertEqual(len(cells), len(set(cells)))
        self.assertFalse(set(cells) & self.problem.fixed.keys())
        core_pieces = {self.board[c] // 4 for c in core}
        for donor in set(cells) - set(core):
            piece = self.board[donor] // 4
            self.assertTrue(any(s // 4 == piece for c in core for s in self.problem.domains[c]))
            self.assertTrue(any(s // 4 in core_pieces for s in self.problem.domains[donor]))

    def test_perfect_fixed_and_cancelled_boards_choose_no_cells(self):
        problem, perfect = generated_problem(3, 5, 22)
        self.assertEqual(select_region(problem, perfect), [])
        fixed = replace(self.problem, fixed=dict(enumerate(self.board)))
        self.assertEqual(select_region(fixed, self.board), [])
        self.assertEqual(select_region(self.problem, self.board, should_stop=lambda: True), [])

    def test_selection_rejects_invalid_arguments(self):
        for kwargs in ({'region_size': 0}, {'region_size': True}, {'region_size': 65},
                       {'donor_count': -1}, {'donor_count': True}, {'region_size': 64, 'donor_count': 1},
                       {'seed': -1}, {'seed': True}, {'should_stop': 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_region(self.problem, self.board, **kwargs)
        with self.assertRaises(ValueError):
            select_region(replace(self.problem, edges=()), self.board)
        with self.assertRaises(ValueError):
            select_region(self.problem, self.board[:-1])


class WaitingSolver:
    def __init__(self, status):
        self.parameters = SimpleNamespace()
        self.entered = threading.Event()
        self.stopped = threading.Event()
        self.status = status
        self.num_branches, self.num_conflicts = 11, 3

    def solve(self, model, callback):
        self.entered.set()
        if not self.stopped.wait(1):
            raise AssertionError('Native regional solver was not cancelled')
        return self.status

    def stop_search(self):
        self.stopped.set()

    def status_name(self, status):
        return 'UNKNOWN'


@unittest.skipUnless(importlib.util.find_spec('ortools'), 'ortools unavailable')
class RegionRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ortools.sat.python import cp_model
        cls.cp_model = cp_model

    def test_adjacent_region_score_agrees_with_exhaustive_oracle(self):
        for impossible in (False, True):
            problem = tiny_problem(impossible)
            reference = min(legal_boards(problem))[1]
            for cells in ([0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 3]):
                with self.subTest(impossible=impossible, cells=cells):
                    expected, _ = region_oracle(problem, reference, cells)
                    result = region_repair(problem, reference, cells, seconds=2, seed=2)
                    self.assertEqual(result['score'], expected, result)
                    self.assertEqual(result['target_score'], evaluate_board(problem, reference)['score'] + 1)
                    self.assertEqual(evaluate_board(problem, result['board'])['score'], result['score'])
                    self.assertEqual({result['board'][c] // 4 for c in cells}, {reference[c] // 4 for c in cells})
                    self.assertTrue(all(result['board'][c] == reference[c] for c in range(4) if c not in cells))
                    self.assertEqual(result['workers'], 1)
                    self.assertIn('frozen-complement', result['proof_scope'])

    def test_internal_and_boundary_edges_counted_once(self):
        problem = tiny_problem(True)
        reference = min(legal_boards(problem))[1]
        result = region_repair(problem, reference, [0, 1], seconds=2)
        self.assertEqual(result['changing_edges'], 3)
        expected_fixed = int(problem.faces[reference[2]][1] == problem.faces[reference[3]][3])
        self.assertEqual(result['fixed_score'], expected_fixed)
        self.assertEqual(result['score'], region_oracle(problem, reference, [0, 1])[0])

    def test_fixed_oriented_clues_and_cell_domains_preserved(self):
        problem = tiny_problem(fixed={0: 0})
        reference = min(legal_boards(problem))[1]
        domains = tuple(tuple(state for state in domain if state == reference[cell] or state % 4 == 0)
                        for cell, domain in enumerate(problem.domains))
        problem = replace(problem, domains=domains)
        result = region_repair(problem, reference, [1, 2, 3], seconds=2)
        self.assertEqual(result['score'], region_oracle(problem, reference, [1, 2, 3])[0])
        self.assertEqual(result['board'][0], 0)
        self.assertTrue(evaluate_board(problem, result['board'])['valid'])
        with self.assertRaises(ValueError):
            region_repair(problem, reference, [0, 1])

    def test_impossible_local_target_is_scoped_and_keeps_parent(self):
        problem = tiny_problem(True)
        score, reference = max(legal_boards(problem))
        result = region_repair(problem, reference, [0, 1], seconds=2)
        self.assertEqual(result['termination'], 'no_improvement_in_region')
        self.assertEqual(result['board'], reference)
        self.assertEqual(result['score'], score)
        self.assertEqual(result['target_score'], 4)
        self.assertEqual(result['proof_scope'], 'selected-cells-and-piece-pool-with-frozen-complement-only')

    def test_noop_timeout_stop_and_already_perfect_do_not_start_solver(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        with patch.object(self.cp_model, 'CpSolver', side_effect=AssertionError('Unexpected native solve')):
            for kwargs, expected in (({'seconds': 0}, 'timeout'),
                                     ({'seconds': 1, 'should_stop': lambda: True}, 'stopped')):
                result = region_repair(problem, reference, [0, 1], **kwargs)
                self.assertEqual(result['termination'], expected)
                self.assertEqual(result['board'], reference)
            self.assertEqual(region_repair(problem, reference, [], seconds=1)['termination'], 'empty_region')
            perfect = max(legal_boards(problem))[1]
            self.assertEqual(region_repair(problem, perfect, [0, 1], seconds=1)['termination'], 'already_perfect')

    def test_build_and_native_search_share_one_budget(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        original = self.cp_model.CpModel.add_allowed_assignments
        def slow_table(model, variables, rows):
            time.sleep(.03)
            return original(model, variables, rows)
        with patch.object(self.cp_model.CpModel, 'add_allowed_assignments', slow_table), \
             patch.object(self.cp_model, 'CpSolver', side_effect=AssertionError('No second budget')):
            result = region_repair(problem, reference, [0, 1], seconds=.01)
        self.assertEqual(result['termination'], 'timeout')
        self.assertEqual(result['phase'], 'building')
        self.assertEqual(result['board'], reference)

    def test_native_cancellation_and_timeout_never_claim_infeasibility(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        for stop in (False, True):
            fake = WaitingSolver(self.cp_model.UNKNOWN)
            with self.subTest(stop=stop), patch.object(self.cp_model, 'CpSolver', return_value=fake):
                result = region_repair(problem, reference, [0, 1], seconds=2 if stop else .08,
                                       should_stop=fake.entered.is_set if stop else None)
            self.assertEqual(result['termination'], 'stopped' if stop else 'timeout')
            self.assertEqual(result['board'], reference)
            self.assertEqual(result['branches'], 11)
            self.assertTrue(fake.stopped.is_set())
            self.assertFalse(any(thread.name == 'eternity-region-cancel' for thread in threading.enumerate()))

    def test_adjacent_interiors_and_donor_pool_repair_to_valid_perfect_board(self):
        problem, perfect = generated_problem(4, 7, 318)
        problem = replace(problem, fixed={0: perfect[0]})
        reference = perfect.copy()
        cells = [5, 6, 9, 10]
        for cell in cells:
            state = reference[cell]
            reference[cell] = (state // 4) * 4 + (state + 1) % 4
        self.assertLess(evaluate_board(problem, reference)['score'], 24)
        result = region_repair(problem, reference, cells, seconds=2)
        self.assertEqual(result['score'], 24, result)
        self.assertEqual(result['termination'], 'perfect_board_validated')
        self.assertTrue(result['optimal'])
        self.assertTrue(evaluate_board(problem, result['board'])['valid'])
        self.assertTrue(all(result['board'][cell] == reference[cell] for cell in range(16) if cell not in cells))

    def test_cancellation_callback_error_cleans_up_native_worker(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        fake = WaitingSolver(self.cp_model.UNKNOWN)
        def broken_stop():
            if fake.entered.is_set():
                raise ValueError('test callback error')
            return False
        with patch.object(self.cp_model, 'CpSolver', return_value=fake), self.assertRaises(RuntimeError):
            region_repair(problem, reference, [0, 1], seconds=2, should_stop=broken_stop)
        self.assertTrue(fake.stopped.is_set())
        self.assertFalse(any(thread.name == 'eternity-region-cancel' for thread in threading.enumerate()))
    def test_unbounded_cell_iterators_are_rejected_after_bounded_consumption(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        consumed = 0
        def repeating_cells():
            nonlocal consumed
            while True:
                consumed += 1
                if consumed > 65:
                    raise AssertionError('Region consumed more than its bounded cell limit')
                yield 0
        with self.assertRaises(ValueError):
            region_repair(problem, reference, repeating_cells())
        self.assertEqual(consumed, 65)
    def test_bad_reference_cells_and_search_arguments_rejected(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        for cells in ([0, 0], [True], [-1], [4], None):
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                region_repair(problem, reference, cells)
        for kwargs in ({'seconds': -1}, {'seconds': True}, {'seconds': float('inf')},
                       {'seconds': float('nan')}, {'seed': 2**32}, {'seed': True}, {'should_stop': False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                region_repair(problem, reference, [0, 1], **kwargs)
        with self.assertRaises(ValueError):
            region_repair(replace(problem, edges=()), reference, [0, 1])
        with self.assertRaises(ValueError):
            region_repair(problem, [reference[0]] * 4, [0, 1])

    def test_independent_validation_rejects_corrupt_native_witness(self):
        problem = tiny_problem()
        reference = min(legal_boards(problem))[1]
        calls = 0
        def checked(puzzle, candidate):
            nonlocal calls
            calls += 1
            if calls > 1:
                return {'valid': False, 'score': None}
            return evaluate_board(puzzle, candidate)
        with patch('repair_regions.evaluate_board', side_effect=checked), self.assertRaises(RuntimeError):
            region_repair(problem, reference, [0, 1, 2, 3], seconds=2)


if __name__ == '__main__':
    unittest.main()
