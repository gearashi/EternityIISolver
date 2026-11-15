from dataclasses import replace
import random
import unittest
from exact_model import generated_problem,grid_edges,check_solution
from test_refined_ordering import independent_oracle
from regional_master import solve


class RegionalSATOracleControls(unittest.TestCase):
    def test_complete_boards_and_cuts_across_mixed_domains(self):
        for seed in range(24):
            q,_=generated_problem(2,1+seed%3,seed,clue_count=seed%2)
            rng=random.Random(5511+seed)
            q=replace(q,domains=tuple(tuple(s for s in d if rng.random()<.86) for d in q.domains),
                      edges=tuple(e for e in grid_edges(2) if rng.random()<.8))
            expected=independent_oracle(q)
            for regions in (((0,1),(2,3)),((0,3),(1,2))):
                r=solve(q,seconds=3,regions=regions,local_oracle='sat',allocation_cores=True,
                        minimize_interfaces=True,hall_capacity=True,color_conservation=True)
                self.assertEqual(r['outcome'],'solved' if expected else 'infeasible',(seed,regions))
                if expected:self.assertTrue(check_solution(q,r['board']))
    def test_cap_remains_unknown(self):
        p,_=generated_problem(3,2,797)
        r=solve(p,local_oracle='sat',max_work=4)
        self.assertFalse(r['complete']);self.assertIsNone(r['board'])
