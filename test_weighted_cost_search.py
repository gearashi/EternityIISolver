from dataclasses import replace
from itertools import product
from random import Random
import unittest
from exact_model import generated_problem, grid_edges
from exact_optimize import evaluate_board
from test_exact_sat import tiny_problem
from weighted_cost_search import optimize


class WeightedTests(unittest.TestCase):
    def test_all_targets_and_bounds_against_independent_exhaustion(self):
        cases=[tiny_problem(),tiny_problem(True),tiny_problem(fixed={0:0})]
        for seed in range(10):
            p,known=generated_problem(3,3,seed,clue_count=1)
            rng=Random(seed)
            d=tuple(tuple(dict.fromkeys((s,rng.choice(d0)))) for s,d0 in zip(known,p.domains))
            cases.append(replace(p,domains=d))
        for p in cases:
            valid=[(b,evaluate_board(p,b)['score']) for b in product(*p.domains) if evaluate_board(p,b)['valid']]
            ref,base=min(valid,key=lambda x:x[1])
            for target in range(base,len(grid_edges(p.size))+1):
                for policy in (True,False):
                    r=optimize(p,ref,seconds=2,target_score=target,directional=policy)
                    self.assertEqual(r['outcome'],'solved' if any(n>=target for _,n in valid) else 'infeasible')
                    self.assertIn((tuple(r['board']),r['score']),valid)

    def test_caps_preserve_reference(self):
        p=tiny_problem(True)
        for opts,outcome in (({'seconds':0},'timeout'),({'max_work':0},'timeout'),({'should_stop':lambda:True},'stopped')):
            r=optimize(p,[0,4,8,12],target_score=4,**opts)
            self.assertEqual(r['outcome'],outcome)
            self.assertEqual(r['score'],3)
