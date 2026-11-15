from dataclasses import replace
import itertools
import random
import unittest
from exact_model import generated_problem,constraint_edges
from test_regional_elimination import score_loss
from scripts.regional_marginals import compile_costs


class DiagramSupportChecks(unittest.TestCase):
    def test_every_returned_minimum_and_threshold_on_tiny_physical_inputs(self):
        for seed in range(18):
            p,_=generated_problem(2,1+seed%3,seed,clue_count=seed%2);rng=random.Random(57+seed)
            p=replace(p,domains=tuple(tuple(s for s in d if rng.random()<.83) for d in p.domains),
                      edges=tuple(e for e in constraint_edges(p) if rng.random()<.9))
            best={}
            for board in itertools.product(*p.domains):
                loss=score_loss(p,board)
                if loss is None:continue
                for cell,state in enumerate(board):best[cell,state]=min(loss,best.get((cell,state),loss))
            for tracked in ((),(1,),range(4)):
                r=compile_costs(p,tracked=tracked)
                self.assertTrue(r['complete'])
                if r['table'] is None:
                    self.assertFalse(best)
                    continue
                table=r['table']
                for (cell,state),cost in best.items():
                    self.assertLessEqual(table.by_state[cell][state],cost)
                    if len(tracked)==4:self.assertEqual(table.by_state[cell][state],cost)
                if len(tracked)==4:
                    self.assertEqual({(c,s):v for c,row in enumerate(table.by_state) for s,v in row.items()},best)
                for threshold in range(len(constraint_edges(p))+1):
                    supported=table.supported(threshold)
                    for (cell,state),v in best.items():
                        if v<=threshold:self.assertIn(state,supported[cell])
    def test_limits_return_unknown(self):
        p,_=generated_problem(3,2,101)
        for options in (dict(max_work=1),dict(max_arcs=1),dict(should_stop=lambda:True)):
            r=compile_costs(p,**options)
            self.assertFalse(r['complete']);self.assertIsNone(r['table'])
