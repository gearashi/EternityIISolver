"""Cross-engine root deductions preserve original solutions and bounded status."""
from dataclasses import replace
from itertools import product
import unittest

from exact_model import check_solution, generated_problem
from exact_preprocess import solve
from test_exact_sat import tiny_problem


class ExactPreprocessTests(unittest.TestCase):
    def test_tiny_cross_engine_controls_and_edge_scopes(self):
        cases = [tiny_problem(), tiny_problem(True), tiny_problem(True, edges=())]
        for seed in (31, 71, 127):
            p, _ = generated_problem(2, 3, seed)
            cases.append(replace(p, domains=tuple(tuple(s for s in d if (s+c+seed)%3)
                                                  for c, d in enumerate(p.domains))))
        for p in cases:
            expected = {board for board in product(*p.domains) if check_solution(p, board)}
            for engine, profile in product(('sat','cp-sat','regional-master'), ('matching','rules')):
                with self.subTest(engine=engine, profile=profile, case=p):
                    r = solve(p, backend=engine, seconds=4, profile=profile)
                    self.assertEqual(r['outcome'], 'solved' if expected else 'infeasible')
                    if expected:
                        self.assertIn(tuple(r['board']), expected)
                    self.assertGreater(r['preprocessing']['work_ticks'], 0)

    def test_stopped_or_expired_has_no_proof_and_fresh_retry_works(self):
        problem = tiny_problem()
        for kwargs, expected in (({'seconds': 0}, 'timeout'), ({'should_stop': lambda: True}, 'stopped')):
            result = solve(problem, backend='sat', **kwargs)
            self.assertEqual(result['outcome'], expected)
            self.assertFalse(result['native_started'])
        self.assertTrue(check_solution(problem, solve(problem, backend='sat', seconds=3)['board']))

    def test_progress_callback_cannot_change_the_frozen_input(self):
        p=tiny_problem(fixed={0:0})
        def progress(_):
            p.fixed[0]=15
        r=solve(p,backend='sat',seconds=3,on_progress=progress)
        self.assertEqual(r['outcome'],'solved')
        self.assertEqual(r['board'][0],0)

    def test_soft_score_and_assumption_engines_rejected(self):
        with self.assertRaises(ValueError):
            solve(tiny_problem(True), backend='sat-optimize')
        with self.assertRaises(ValueError):
            solve(tiny_problem(True), backend='sat', assumptions={})
