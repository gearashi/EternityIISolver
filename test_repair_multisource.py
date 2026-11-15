"""Independent tiny oracles for bounded multi-source search improvements."""
from dataclasses import replace
import importlib.util
import itertools
import unittest
from unittest.mock import patch

from exact_model import check_solution, generated_problem, grid_edges
from exact_optimize import evaluate_board
from repair_multisource import ExplainedRegionExpansion, SquareSupportSession, multi_parent_recombine
from test_exact_sat import tiny_problem


def legal_boards(problem):
    return [list(board) for board in itertools.product(*problem.domains)
            if evaluate_board(problem, board)['valid']]


def three_parents():
    problem, solution = generated_problem(4, 20, 84, clue_count=1)
    cells = [c for c in (5, 6, 9, 10) if c not in problem.fixed][:3]
    parents = []
    for correct in cells:
        board = solution.copy()
        for cell in cells:
            if cell != correct:
                board[cell] = 4 * (board[cell] // 4) + (board[cell] + 1) % 4
        parents.append(board)
    return problem, solution, cells, parents


@unittest.skipUnless(importlib.util.find_spec('ortools'), 'OR-Tools unavailable')
class MultiParentTests(unittest.TestCase):
    def test_three_parents_recover_witness_missing_from_every_pair_union(self):
        problem, solution, cells, parents = three_parents()
        for pair in itertools.combinations(parents, 2):
            scores = []
            for states in itertools.product(*[sorted({parent[c] for parent in pair}) for c in cells]):
                board = solution.copy()
                for cell, state in zip(cells, states):
                    board[cell] = state
                scores.append(evaluate_board(problem, board)['score'])
            self.assertLess(max(scores), 24)
        answer = multi_parent_recombine(problem, parents, cells=cells, seconds=5, deterministic_budget=.1)
        self.assertEqual(answer['score'], 24)
        self.assertTrue(check_solution(problem, answer['board']))
        self.assertEqual(answer['distinct_parents'], 3)
        self.assertLessEqual(answer['table_rows'], 3 * len(cells))
        self.assertEqual(answer['workers'], 1)
        self.assertGreaterEqual(answer['deterministic_work'], 0)
        for cell, state in problem.fixed.items():
            self.assertEqual(answer['board'][cell], state)

    def test_native_optimum_matches_exhaustive_observed_state_union(self):
        problem = tiny_problem()
        boards = legal_boards(problem)
        for indices in ((1, 4, 9), (0, 7, 15), (3, 11, 22)):
            parents = [boards[i] for i in indices]
            reference = max(parents, key=lambda b: evaluate_board(problem, b)['score'])
            best = max(evaluate_board(problem, b)['score'] for b in boards
                       if all(b[c] in {parent[c] for parent in parents} for c in range(4)))
            answer = multi_parent_recombine(problem, parents, cells=range(4), seconds=2,
                                             deterministic_budget=.1, first_improvement=False)
            self.assertEqual(answer['score'], best)
            self.assertGreaterEqual(answer['score'], evaluate_board(problem, reference)['score'])
            self.assertTrue(answer['optimal'])
            self.assertTrue(evaluate_board(problem, answer['board'])['valid'])

    def test_selected_pool_and_frozen_complement_are_preserved(self):
        problem, _, cells, parents = three_parents()
        cells = cells[:2]
        answer = multi_parent_recombine(problem, parents, cells=cells, seconds=2,
                                         deterministic_budget=.1, first_improvement=False)
        reference = parents[answer['reference_parent']]
        self.assertTrue(all(answer['board'][c] == reference[c] for c in range(16) if c not in cells))
        self.assertEqual({answer['board'][c] // 4 for c in cells}, {reference[c] // 4 for c in cells})
        self.assertTrue(all(answer['board'][c] in {p[c] for p in parents} for c in cells))

    def test_cancel_zero_work_and_zero_wall_budget_return_best_parent(self):
        problem, _, cells, parents = three_parents()
        best = max(evaluate_board(problem, b)['score'] for b in parents)
        for kwargs, outcome in (({'seconds': 0}, 'timeout'),
                                ({'deterministic_budget': 0}, 'work_limit'),
                                ({'should_stop': lambda: True}, 'stopped')):
            answer = multi_parent_recombine(problem, parents, cells=cells, **kwargs)
            self.assertEqual(answer['score'], best)
            self.assertEqual(answer['outcome'], outcome)
            self.assertEqual(answer['branches'], 0)
            self.assertFalse(answer['optimal'])

    def test_input_validation_and_automatic_region_cap(self):
        problem, _, cells, parents = three_parents()
        for kwargs in ({'seconds': -1}, {'seconds': True}, {'deterministic_budget': float('nan')},
                       {'seed': True}, {'should_stop': 4}, {'max_cells': 0},
                       {'cells': [cells[0], cells[0]]}, {'cells': list(problem.fixed)},
                       {'first_improvement': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                multi_parent_recombine(problem, parents, **kwargs)
        with self.assertRaises(ValueError):
            multi_parent_recombine(problem, parents[:2])
        with self.assertRaises(ValueError):
            multi_parent_recombine(replace(problem, edges=()), parents)
        invalid = [b.copy() for b in parents]
        invalid[0][cells[0]] = invalid[0][cells[1]]
        with self.assertRaises(ValueError):
            multi_parent_recombine(problem, invalid)
        answer = multi_parent_recombine(problem, parents, max_cells=1, seconds=0)
        self.assertLessEqual(len(answer['region_cells']), 1)


@unittest.skipUnless(importlib.util.find_spec('pysat'), 'python-sat unavailable')
class SquareSupportTests(unittest.TestCase):
    def test_every_structurally_legal_tiny_assignment_matches_independent_oracle(self):
        for unsatisfiable in (False, True):
            problem = tiny_problem(unsatisfiable)
            with SquareSupportSession(problem, seconds=10) as session:
                for board in legal_boards(problem):
                    answer = session.solve(dict(enumerate(board)), seconds=1)
                    expected = 'solved' if check_solution(problem, board) else 'infeasible'
                    self.assertEqual(answer['outcome'], expected)
                    self.assertEqual(answer['model_builds'], 1)
                    self.assertEqual(answer['square_support_encoded'], 1)
                    if expected == 'solved':
                        self.assertEqual(answer['board'], board)
                    else:
                        core = answer['core_placements']
                        self.assertIsNotNone(core)
                        # The alleged core alone must rule out every exact solution.
                        self.assertFalse(any(check_solution(problem, candidate)
                                             and all(candidate[c] == state for c, state in core.items())
                                             for candidate in legal_boards(problem)))

    def test_incomplete_tuple_enumeration_is_skipped_without_pruning(self):
        problem = tiny_problem()
        for kwargs in ({'square_work_budget': 1}, {'square_tuple_cap': 1}):
            with SquareSupportSession(problem, seconds=5, **kwargs) as session:
                answer = session.solve()
                self.assertEqual(answer['outcome'], 'solved')
                self.assertTrue(check_solution(problem, answer['board']))
                self.assertEqual(answer['square_support_encoded'], 0)
                self.assertEqual(answer['square_support_skipped'], 1)
                self.assertEqual(answer['square_support_tuples'], 0)

    def test_partial_edge_problem_does_not_gain_missing_constraints(self):
        problem = replace(tiny_problem(True), edges=tuple(grid_edges(2)[:2]))
        with SquareSupportSession(problem, seconds=5) as session:
            answer = session.solve()
            self.assertEqual(answer['outcome'], 'solved')
            self.assertTrue(check_solution(problem, answer['board']))
            self.assertEqual(answer['square_support_encoded'], 0)

    def test_interrupted_encoding_is_discarded_then_can_rebuild(self):
        problem = tiny_problem()
        with SquareSupportSession(problem, seconds=5) as session:
            original = session._add
            armed = [True]
            def interrupt_extension(clause):
                # Base model sets _built before the extension resets it.
                if armed[0] and session.placements and not session._built:
                    armed[0] = False
                    from exact_sat import _Cancelled
                    raise _Cancelled('timeout')
                return original(clause)
            with patch.object(session, '_add', side_effect=interrupt_extension):
                failed = session.solve()
            self.assertEqual(failed['outcome'], 'timeout')
            self.assertIsNone(failed['core_cells'])
            self.assertIsNone(session.native)
            answer = session.solve()
            self.assertEqual(answer['outcome'], 'solved')
            self.assertEqual(answer['model_builds'], 2)
            self.assertTrue(check_solution(problem, answer['board']))

    def test_bad_limits_rejected(self):
        for kwargs in ({'square_work_budget': -1}, {'square_work_budget': True}, {'square_tuple_cap': -1}):
            with self.assertRaises(ValueError):
                SquareSupportSession(tiny_problem(), **kwargs)


@unittest.skipUnless(importlib.util.find_spec('pysat'), 'python-sat unavailable')
class ExplainedExpansionTests(unittest.TestCase):
    def test_real_core_releases_a_frozen_cell_and_preserves_fixed_clue(self):
        problem = tiny_problem(fixed={0: 0})
        reference = next(board for board in legal_boards(problem) if not check_solution(problem, board))
        with ExplainedRegionExpansion(problem, seconds=5) as helper:
            answer = helper.expand(reference, [], seconds=1)
            self.assertEqual(answer['outcome'], 'expanded')
            self.assertEqual(len(answer['released_cells']), 1)
            self.assertTrue(set(answer['released_cells']) <= set(answer['core_cells']))
            self.assertFalse(set(answer['region_cells']) & problem.fixed.keys())
            core = answer['core_placements']
            self.assertFalse(any(check_solution(problem, b) and all(b[c] == state for c, state in core.items())
                                 for b in legal_boards(problem)))
            second = helper.expand(reference, answer['region_cells'], seconds=1)
            self.assertEqual(second['sat_result']['model_builds'], 1)
            self.assertEqual(second['queries'], 2)

    def test_full_mutable_region_returns_independently_validated_perfect_witness(self):
        problem = tiny_problem(fixed={0: 0})
        reference = next(b for b in legal_boards(problem) if not check_solution(problem, b))
        with ExplainedRegionExpansion(problem, seconds=5, square_support=True) as helper:
            answer = helper.expand(reference, [1, 2, 3], seconds=1)
        self.assertEqual(answer['outcome'], 'solved')
        self.assertTrue(check_solution(problem, answer['board']))
        self.assertEqual(answer['board'][0], 0)

    def test_unknown_query_limit_and_region_limit_make_no_new_proof(self):
        problem = tiny_problem()
        reference = next(b for b in legal_boards(problem) if not check_solution(problem, b))
        with ExplainedRegionExpansion(problem, seconds=5, max_queries=1) as helper:
            unknown = helper.expand(reference, [], seconds=0)
            self.assertEqual(unknown['outcome'], 'timeout')
            self.assertIsNone(unknown['core_cells'])
            limited = helper.expand(reference, [], seconds=1)
            self.assertEqual(limited['outcome'], 'query_limit')
            self.assertIsNone(limited['core_cells'])
        with ExplainedRegionExpansion(problem, seconds=5) as helper:
            limited = helper.expand(reference, [0], max_cells=1, seconds=1)
            self.assertEqual(limited['outcome'], 'region_limit')
            self.assertEqual(limited['released_cells'], [])
            self.assertTrue(limited['core_cells'])

    def test_base_unsat_scope_is_explicit(self):
        problem = tiny_problem(True)
        reference = legal_boards(problem)[0]
        with ExplainedRegionExpansion(problem, seconds=5, square_support=True) as helper:
            answer = helper.expand(reference, range(4), seconds=1)
        self.assertEqual(answer['outcome'], 'base_infeasible')
        self.assertEqual(answer['core_cells'], [])
        self.assertEqual(answer['proof_target'], 'all-shared-edges-match')


if __name__ == '__main__':
    unittest.main()
