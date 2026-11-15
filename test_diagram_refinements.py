from dataclasses import replace
import random
import unittest
from exact_model import generated_problem,check_solution,grid_edges
from test_refined_ordering import independent_oracle
from refined_diagrams import solve


class DiagramRefinementControls(unittest.TestCase):
    def test_all_inferred_restrictions_allow_every_exact_solution(self):
        for seed in range(20):
            p,_=generated_problem(2,1+seed%3,seed,clue_count=seed%2)
            rng=random.Random(seed+971)
            p=replace(p,domains=tuple(tuple(s for s in d if rng.random()<.81) for d in p.domains),
                      edges=tuple(e for e in grid_edges(2) if rng.random()<.9))
            expected=independent_oracle(p)
            for options in (dict(groups=True),dict(refinement='local'),
                            dict(refinement='local',groups=True)):
                r=solve(p,seed=seed,identity_lifetime='require',refine_batch=1,**options)
                self.assertEqual(r['outcome'],'solved' if expected is not None else 'infeasible',(seed,options))
                if expected:self.assertTrue(check_solution(p,r['board']))
    def test_local_counterexample_never_concludes_a_cutoff_is_a_proof(self):
        p,_=generated_problem(2,1,91)
        p=replace(p,domains=tuple(tuple(s for s in d if s//4<3) for d in p.domains))
        r=solve(p,refinement='local',refine_batch=1,max_refinements=0)
        self.assertEqual(r['termination'],'refinement_limit')
        r=solve(p,refinement='local',refine_batch=1)
        self.assertEqual(r['outcome'],'infeasible')
        self.assertGreater(r['reasoning_counters_session']['local_pair_failures'],0)
