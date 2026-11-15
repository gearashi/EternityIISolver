from dataclasses import replace
from itertools import product
import unittest
from exact_model import generated_problem,check_solution
from perfect_distance import DistanceBounds,verify_packing,packing


class DistanceTests(unittest.TestCase):
    def test_certificates_against_every_tiny_board_and_all_solutions(self):
        for seed in range(8):
            p,_=generated_problem(2,3,seed,clue_count=seed%2)
            boards=[b for b in product(*p.domains) if len({s//4 for s in b})==4]
            valid=[b for b in boards if check_solution(p,b)]
            self.assertTrue(valid)
            for parent in boards:
                oracle=min(sum(s!=t for s,t in zip(parent,b)) for b in valid)
                d=DistanceBounds(p,parent)
                for region in ((0,1),(0,1,2),(0,1,2,3)):
                    v=d.local_minimum(region)
                    certified=verify_packing(p,parent,[(region,v,1)],1)
                    self.assertLessEqual(certified['lower_bound'],oracle)
                    self.assertLessEqual(v,min(sum(parent[i]!=b[i] for i in region) for b in valid))

    def test_bad_packing_or_overstated_cut_rejected(self):
        p,b=generated_problem(2,2,3)
        with self.assertRaises(ValueError):verify_packing(p,b,[((0,1),1,1)],1)
        with self.assertRaises(ValueError):verify_packing(p,b,[((0,1),0,1),((1,2),0,1)],1)
        with self.assertRaises(ValueError):DistanceBounds(p,[b[0]]*4)
        q=replace(p,edges=())
        self.assertEqual(DistanceBounds(q,b).local_minimum(range(4)),0)
