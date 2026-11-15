"""Production CP-region proof reuse, isolated finite cases and cancellation."""
from dataclasses import replace
import importlib.util
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from exact_model import generated_problem
from exact_optimize import evaluate_board
from repair_regions import RegionRepairSession
from test_exact_sat import tiny_problem


def legal_boards(problem):
    return [(checked['score'], list(board)) for board in itertools.product(*problem.domains)
            if (checked := evaluate_board(problem, board))['valid']]


class RegionSessionSignatureTests(unittest.TestCase):
    def test_semantic_key_binds_every_scoped_input(self):
        problem = tiny_problem(True); boards = legal_boards(problem); score, board = max(boards)
        session = RegionRepairSession(problem); cells = [0, 1]; target = score + 1
        base = session._key(board, cells, target)
        self.assertNotEqual(base, session._key(board, cells, target - 1))
        self.assertNotEqual(base, session._key(board, [0, 2], target))
        changed = next(b for _, b in boards if b[2:] != board[2:] and {b[c] // 4 for c in cells} != {board[c] // 4 for c in cells})
        self.assertNotEqual(base, session._key(changed, cells, target))
        same_pool = next(b for _, b in boards if b[2:] != board[2:] and {b[c] // 4 for c in cells} == {board[c] // 4 for c in cells})
        self.assertNotEqual(base, session._key(same_pool, cells, target))
        domains = tuple((board[c],) if c == 0 else d for c, d in enumerate(problem.domains))
        different = RegionRepairSession(replace(problem, domains=domains))
        self.assertNotEqual(base, different._key(board, cells, target))
        different = RegionRepairSession(replace(problem, fixed={2: board[2]}))
        self.assertNotEqual(base, different._key(board, cells, target))

    def test_session_owns_immutable_problem_snapshot(self):
        source = tiny_problem(True)
        source = replace(source, faces=[list(f) for f in source.faces], domains=[list(d) for d in source.domains])
        session = RegionRepairSession(source); before = session._problem_hash
        source.faces[0][0] = 255; source.domains[0].clear(); source.fixed[0] = 1
        self.assertEqual(session._problem_hash, before)
        self.assertNotEqual(session.problem.faces[0][0], 255)
        self.assertTrue(session.problem.domains[0]); self.assertEqual(dict(session.problem.fixed), {})
        with self.assertRaises(TypeError): session.problem.fixed[0] = 1

    def test_argument_validation_and_bounded_cell_consumption(self):
        p = tiny_problem(True); board = max(legal_boards(p))[1]; session = RegionRepairSession(p)
        for capacity in (0, 4097, True, 1.5):
            with self.assertRaises(ValueError): RegionRepairSession(p, capacity)
        for cells in ([0, 0], [True], [-1], [4], None):
            with self.assertRaises(ValueError): session.solve(board, cells, seconds=0)
        for kwargs in ({'seconds': True}, {'seconds': -1}, {'seconds': float('inf')},
                       {'seed': True}, {'should_stop': 1}):
            with self.assertRaises(ValueError): session.solve(board, [0], **kwargs)
        calls = [0]
        def many():
            while True:
                calls[0] += 1
                if calls[0] > 65: raise AssertionError('Unbounded consumption')
                yield 0
        with self.assertRaises(ValueError): session.solve(board, many(), seconds=0)
        self.assertEqual(calls[0], 65)

    def test_session_supports_existing_64_cell_limit(self):
        p, answer = generated_problem(8, 3, 1)
        result = RegionRepairSession(p).solve(answer, range(64), seconds=0)
        self.assertEqual(len(result['region_cells']), 64)
        self.assertEqual(result['termination'], 'timeout')


@unittest.skipUnless(importlib.util.find_spec('ortools'), 'ortools unavailable')
class RegionSessionProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ortools.sat.python import cp_model
        cls.cp_model = cp_model

    def test_real_completed_proof_replays_without_native_search(self):
        p = tiny_problem(True); boards = legal_boards(p); score, reference = max(boards)
        self.assertEqual(score, 3)
        session = RegionRepairSession(p)
        first = session.solve(reference, range(4), seconds=2)
        self.assertEqual(first['solver_status'], 'INFEASIBLE')
        self.assertEqual(first['termination'], 'no_improvement_in_region')
        self.assertEqual(session.cache_size, 1)
        other = next(b for s, b in boards if s == score and b != reference)
        with patch('repair_regions.region_repair', side_effect=AssertionError('Repeated native search')):
            second = session.solve(other, reversed(range(4)), seconds=1)
        self.assertTrue(second['cache_hit']); self.assertTrue(second['proof_reused'])
        self.assertEqual(second['board'], other); self.assertEqual(second['score'], score)
        self.assertEqual(second['branches'], 0); self.assertEqual(second['conflicts'], 0)
        self.assertEqual(second['phase'], 'cached_proof')
        self.assertEqual(second['proof_scope'], first['proof_scope'])
        second['board'][0] = -1
        self.assertTrue(evaluate_board(p, session.solve(other, range(4), seconds=1)['board'])['valid'])
        # Changed target on the same physical pool must be searched again.
        low = min(boards)[1]
        changed = session.solve(low, range(4), seconds=2)
        self.assertFalse(changed.get('cache_hit', False)); self.assertTrue(changed['improved'])

    def test_stop_and_timeout_never_create_or_reuse_proofs(self):
        p = tiny_problem(True); board = max(legal_boards(p))[1]
        session = RegionRepairSession(p)
        for kwargs, termination in (({'seconds': 0}, 'timeout'),
                                    ({'seconds': 1, 'should_stop': lambda: True}, 'stopped')):
            result = session.solve(board, [0, 1], **kwargs)
            self.assertEqual(result['termination'], termination); self.assertEqual(session.cache_size, 0)
        session.solve(board, [0, 1], seconds=2)
        self.assertEqual(session.cache_size, 1)
        for kwargs, termination in (({'seconds': 0}, 'timeout'),
                                    ({'seconds': 1, 'should_stop': lambda: True}, 'stopped')):
            result = session.solve(board, [0, 1], **kwargs)
            self.assertEqual(result['termination'], termination); self.assertFalse(result.get('cache_hit', False))

    def test_expired_or_stopped_native_status_is_not_cached(self):
        p = tiny_problem(True); board = max(legal_boards(p))[1]
        for stopped in (False, True):
            session = RegionRepairSession(p); clock = [0.]; native_done = [False]
            def native_solve(model, callback):
                native_done[0] = True
                if not stopped: clock[0] = 2.
                return self.cp_model.INFEASIBLE
            fake = SimpleNamespace(parameters=SimpleNamespace(), num_branches=4, num_conflicts=2,
                                   solve=native_solve, stop_search=lambda: None,
                                   status_name=lambda status: 'INFEASIBLE')
            with patch.object(self.cp_model, 'CpSolver', return_value=fake), \
                 patch('repair_regions.time.monotonic', side_effect=lambda: clock[0]):
                result = session.solve(board, [0, 1], seconds=1,
                                       should_stop=(lambda: native_done[0]) if stopped else None)
            self.assertEqual(result['termination'], 'stopped' if stopped else 'timeout')
            self.assertEqual(result['solver_status'], 'INFEASIBLE'); self.assertEqual(session.cache_size, 0)

    def test_model_invalid_and_unknown_are_not_proofs(self):
        p = tiny_problem(True); board = max(legal_boards(p))[1]
        for status in (self.cp_model.MODEL_INVALID, self.cp_model.UNKNOWN):
            session = RegionRepairSession(p)
            fake = SimpleNamespace(parameters=SimpleNamespace(), num_branches=0, num_conflicts=0,
                                   solve=lambda model, callback: status, stop_search=lambda: None,
                                   status_name=lambda status: 'MODEL_INVALID' if status == self.cp_model.MODEL_INVALID else 'UNKNOWN',
                                   response_stats=lambda: 'test-only invalid response')
            with patch.object(self.cp_model, 'CpSolver', return_value=fake):
                if status == self.cp_model.MODEL_INVALID:
                    with self.assertRaises(RuntimeError): session.solve(board, [0, 1], seconds=1)
                else:
                    self.assertEqual(session.solve(board, [0, 1], seconds=1)['termination'], 'timeout')
            self.assertEqual(session.cache_size, 0)

    def test_lru_eviction_and_changed_cells_do_not_reuse_other_proof(self):
        p = tiny_problem(True); board = max(legal_boards(p))[1]
        session = RegionRepairSession(p, capacity=1)
        for cells in ([0], [1], [0]):
            result = session.solve(board, cells, seconds=2)
            self.assertEqual(result['termination'], 'no_improvement_in_region')
            self.assertFalse(result.get('cache_hit', False)); self.assertEqual(session.cache_size, 1)
        self.assertTrue(session.solve(board, [0], seconds=1)['cache_hit'])


if __name__ == '__main__': unittest.main()
