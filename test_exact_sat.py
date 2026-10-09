"""Tiny exact SAT checks; no GPU, official BOINC files or networking."""
import importlib.util
import itertools
import random
import threading
import time
import unittest
from unittest.mock import patch

from exact_model import ExactProblem, check_solution, generated_problem, make_problem
import exact_sat


def tiny_problem(unsatisfiable=False, *, edges=None, fixed=None):
    pieces = [(0, 1, 2, 0), (0, 0, 3, 1), (2, 4, 0, 0),
              (3, 0, 0, 9 if unsatisfiable else 4)]
    faces = tuple(tuple(piece[(side - rotation) % 4] for side in range(4))
                  for piece in pieces for rotation in range(4))
    domains = tuple(tuple(state for state, face in enumerate(faces)
                          if all((color == 0) == boundary for color, boundary in zip(
                              face, (cell < 2, cell % 2 == 1, cell >= 2, cell % 2 == 0))))
                    for cell in range(4))
    return ExactProblem(size=2, faces=faces, fixed={} if fixed is None else fixed,
                        domains=domains, edges=edges)


@unittest.skipUnless(importlib.util.find_spec('pysat'), 'python-sat unavailable')
class ExactSATTests(unittest.TestCase):
    def test_shared_generated_puzzles(self):
        for size, colors, seed, clues in ((1, 1, 0, 0), (2, 3, 42, 1), (3, 4, 91, 2)):
            with self.subTest(size=size):
                problem, known = generated_problem(size, colors, seed, clue_count=clues)
                self.assertTrue(check_solution(problem, known))
                answer = exact_sat.solve(problem, seconds=5, seed=seed, workers=2)
                self.assertEqual(answer['outcome'], 'solved')
                self.assertTrue(check_solution(problem, answer['board']))
                self.assertEqual(answer['workers'], 1)
                self.assertEqual(answer['requested_workers'], 2)

    def test_tiny_domain_restrictions_agree_with_exhaustive_oracle(self):
        rng = random.Random(765)
        for seed in range(8):
            with self.subTest(seed=seed):
                base, _ = generated_problem(2, 3, seed)
                domains = tuple(tuple(s for s in d if rng.random() < 0.65) for d in base.domains)
                problem = ExactProblem(size=base.size, faces=base.faces, fixed=base.fixed,
                                       domains=domains, edges=base.edges)
                exists = any(check_solution(problem, board) for board in itertools.product(*domains))
                answer = exact_sat.solve(problem, seconds=5, seed=seed)
                self.assertEqual(answer['outcome'], 'solved' if exists else 'infeasible')
                if exists:
                    self.assertTrue(check_solution(problem, answer['board']))

    def test_solves_tiny_board_and_respects_fixed_state(self):
        problem = tiny_problem(fixed={0: 0})
        answer = exact_sat.solve(problem, seconds=5, seed=123)
        self.assertEqual(answer['outcome'], 'solved')
        self.assertTrue(check_solution(problem, answer['board']))
        self.assertEqual(answer['board'][0], 0)
        self.assertEqual(sorted(state // 4 for state in answer['board']), [0, 1, 2, 3])
        self.assertEqual(answer['workers'], 1)
        self.assertGreater(answer['clauses'], 0)

    def test_contradictory_edge_colors_are_proved_infeasible(self):
        answer = exact_sat.solve(tiny_problem(True), seconds=5)
        self.assertEqual(answer['outcome'], 'infeasible')
        self.assertIsNone(answer['board'])

    def test_omitted_edge_constraints_are_not_reintroduced(self):
        problem = tiny_problem(True, edges=(), fixed={0: 0, 1: 4, 2: 8, 3: 12})
        answer = exact_sat.solve(problem, seconds=5)
        self.assertEqual(answer['outcome'], 'solved')
        self.assertTrue(check_solution(problem, answer['board']))
        self.assertEqual(answer['board'], [0, 4, 8, 12])

    def test_explicit_edge_subset_still_enforces_included_edges(self):
        base = tiny_problem(True)
        fixed = {0: 0, 1: 4, 2: 8, 3: 12}
        for edges, expected in ((((0, 1, 1, 3),), 'solved'), (((2, 3, 1, 3),), 'infeasible')):
            with self.subTest(edges=edges):
                problem = make_problem(base.size, base.faces, fixed=fixed, edges=edges)
                answer = exact_sat.solve(problem, seconds=5)
                self.assertEqual(answer['outcome'], expected)

    def test_piece_cannot_be_used_twice(self):
        problem = tiny_problem()
        problem = ExactProblem(size=2, faces=problem.faces, fixed={},
                               domains=tuple(tuple(s for s in d if s // 4 == 0) for d in problem.domains),
                               edges=())
        answer = exact_sat.solve(problem, seconds=5)
        self.assertEqual(answer['outcome'], 'infeasible')

    def test_immediate_timeout_is_not_infeasible(self):
        answer = exact_sat.solve(tiny_problem(), seconds=0)
        self.assertEqual(answer['outcome'], 'timeout')
        self.assertIsNone(answer['board'])

    def test_preexisting_stop_is_not_infeasible(self):
        answer = exact_sat.solve(tiny_problem(True), seconds=5, should_stop=lambda: True)
        self.assertEqual(answer['outcome'], 'stopped')

    def test_stop_during_encoding_is_cooperative(self):
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls > 5
        answer = exact_sat.solve(tiny_problem(), seconds=5, should_stop=cancel)
        self.assertEqual(answer['outcome'], 'stopped')
        self.assertIsNone(answer['board'])

    def test_interrupt_none_and_false_never_mean_unsat(self):
        # Native waiting is simulated so this test deterministically exercises
        # the monitor race without requiring an unpredictably difficult puzzle.
        for stopped, native_result in ((False, None), (False, False), (True, None), (True, False)):
            with self.subTest(stopped=stopped, native_result=native_result):
                event = threading.Event()
                stop = threading.Event()
                class WaitingSolver:
                    def __init__(self): self.deleted = False
                    def add_clause(self, _): pass
                    def set_phases(self, _): pass
                    def solve_limited(self, expect_interrupt):
                        if stopped: stop.set()
                        self_outer.assertTrue(event.wait(1), 'interrupt did not arrive')
                        return native_result
                    def interrupt(self): event.set()
                    def accum_stats(self): return {'decisions': 7, 'conflicts': 3}
                    def delete(self): self.deleted = True
                self_outer = self
                fake = WaitingSolver()
                with patch('pysat.solvers.Glucose4', return_value=fake):
                    answer = exact_sat.solve(tiny_problem(), seconds=0.08,
                                             should_stop=stop.is_set if stopped else None)
                self.assertEqual(answer['outcome'], 'stopped' if stopped else 'timeout')
                self.assertIsNone(answer['board'])
                self.assertTrue(fake.deleted)
                self.assertEqual(answer['conflicts'], 3)
                self.assertFalse(any(t.name == 'eternity-sat-interrupt' for t in threading.enumerate()))

    def test_progress_callback_time_is_in_total_deadline(self):
        def progress(_): time.sleep(0.02)
        answer = exact_sat.solve(tiny_problem(), seconds=0.005, on_progress=progress)
        self.assertEqual(answer['outcome'], 'timeout')
        self.assertIsNone(answer['board'])


if __name__ == '__main__':
    unittest.main()
