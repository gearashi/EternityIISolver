import itertools
from dataclasses import replace
import unittest

from exact_model import generated_problem, grid_edges
from exact_optimize import evaluate_board
from repair_components import screen
from repair_transfer_bound import certificate


class ComponentTests(unittest.TestCase):
    def test_every_small_improvement_has_a_screened_component(self):
        witnesses=0
        for seed in range(25):
            p,answer=generated_problem(5,3,seed)
            cells=(6,7,8,11,12)
            ds=tuple((s,(s//4)*4+(s+1)%4) if i in cells else (s,) for i,s in enumerate(answer))
            p=replace(p,domains=ds)
            states=[(b,evaluate_board(p,b)['score']) for b in itertools.product(*ds)]
            for b,score in states:
                dual=certificate(p,b)
                if not dual['certified']:
                    continue
                result=screen(p,b,dual['potentials'],radius=5,keep=128)
                self.assertEqual('complete',result['status'])
                samples={tuple(x['cells']) for x in result['samples']}
                self.assertEqual(len(samples),sum(result['viable_by_size'].values()))
                for board,value in states:
                    if value<=score:
                        continue
                    changed={i for i,(s,t) in enumerate(zip(b,board)) if s!=t}
                    cs=[]
                    while changed:
                        comp={changed.pop()}
                        while True:
                            near={x for a,c,_,_ in grid_edges(5) for y,x in ((a,c),(c,a))
                                  if y in comp and x in changed}
                            if not near:
                                break
                            changed-=near
                            comp.update(near)
                        cs.append(tuple(sorted(comp)))
                    self.assertTrue(samples.intersection(cs))
                    witnesses+=1
        self.assertGreater(witnesses,0)

    def test_cap_produces_no_complete_claim(self):
        p,b=generated_problem(5,3,2)
        result=screen(p,b,[0]*len(b),radius=4,max_shapes=1)
        self.assertEqual('incomplete',result['status'])
