from dataclasses import replace
import itertools
import random
import unittest

from exact_model import check_solution, generated_problem, grid_edges
from test_exact_sat import tiny_problem
from test_refined_ordering import independent_oracle
import refined_diagrams
import regional_master


class ScopedRegionalReasoning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ortools.sat.python import cp_model

    def test_sound_across_relaxed_edges_and_nonrotational_tables(self):
        for seed in range(18):
            p, known = generated_problem(2, 1 + seed % 3, seed, clue_count=seed%2)
            rng = random.Random(2711+seed)
            if seed%3==0:
                # ExactProblem accepts arbitrary state faces for each piece.
                states=list(p.faces)
                state=known[0]
                states[state]=tuple(7 if color else 0 for color in states[state])
                p=replace(p, faces=tuple(states))
            p=replace(p, domains=tuple(tuple(s for s in d if rng.random()<.78) for d in p.domains),
                      edges=tuple(e for e in grid_edges(2) if rng.random()<.8))
            expected=independent_oracle(p)
            result=regional_master.solve(p,seconds=3,seed=seed,minimize_interfaces=True,
                        color_conservation=True,allocation_cores=True,hall_capacity=True,cache_witnesses=True,
                        regions=((0,3),(1,2)),max_master_queries=1000)
            self.assertEqual(result['outcome'],'solved' if expected else 'infeasible',(seed,result))
            if expected:self.assertTrue(check_solution(p,result['board']))
            for mode in ('exact','refine'):
                for lifetime in ('project','require'):
                    result=refined_diagrams.solve(p,seconds=2,abstraction=mode,identity_lifetime=lifetime)
                    self.assertEqual(result['outcome'],'solved' if expected else 'infeasible',(seed,result))
                    if expected:self.assertTrue(check_solution(p,result['board']))

    def test_stopped_runs_retry_fresh(self):
        p=tiny_problem(True)
        for engine,options in [(regional_master,dict(color_conservation=True,hall_capacity=True,
                      allocation_cores=True,minimize_interfaces=True,cache_witnesses=True)),
                      (refined_diagrams,dict(identity_lifetime='require'))]:
            self.assertEqual(engine.solve(p,should_stop=lambda:True,**options)['outcome'],'stopped')
            self.assertEqual(engine.solve(p,max_work=3,**options)['termination'],'work_limit')
            self.assertEqual(engine.solve(p,seconds=3,**options)['outcome'],'infeasible')

    def test_missing_piece_has_no_full_board(self):
        p,_=generated_problem(2,1,11)
        q=replace(p,domains=tuple(tuple(s for s in d if s//4!=3) for d in p.domains))
        self.assertIsNone(independent_oracle(q))
        result=refined_diagrams.solve(q,identity_lifetime='require')
        self.assertEqual(result['termination'],'piece_missing')

    def test_subset_edges_allow_matching_with_unpaired_side(self):
        p=tiny_problem(True,edges=())
        result=regional_master.solve(p,color_conservation=True)
        self.assertTrue(check_solution(p,result['board']))
        self.assertGreater(result['conservation_equations'],0)
