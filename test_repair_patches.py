import itertools
import unittest
from exact_model import generated_problem
from exact_optimize import evaluate_board
from repair_patches import compose


class Patches(unittest.TestCase):
    def test_exact_disjoint_combination_of_overlapping_candidates(self):
        for seed in range(8):
            p,solution=generated_problem(5,5,seed,clue_count=1)
            movable=[c for c in (6,7,8,11,12,13) if c not in p.fixed]
            parent=solution.copy()
            for c in movable:
                parent[c]=parent[c]//4*4+(parent[c]+1)%4
            candidates=[]
            for group in [(movable[0],),(movable[1],),(movable[2],),(movable[3],),tuple(movable[:3]),tuple(movable[2:])]:
                b=parent.copy()
                for c in group:
                    b[c]=solution[c]
                candidates.append(b)
            maximum=evaluate_board(p,parent)['score']
            for bits in itertools.product((0,1),repeat=len(candidates)):
                b=parent.copy()
                seen=set()
                for patch,bit in zip(candidates,bits):
                    if not bit:
                        continue
                    diff={i for i,s in enumerate(patch) if s!=parent[i]}
                    if seen & diff:
                        break
                    seen.update(diff)
                    for i in diff:
                        b[i]=patch[i]
                else:
                    maximum=max(maximum,evaluate_board(p,b)['score'])
            r=compose(p,parent,candidates)
            self.assertTrue(r['optimal'])
            self.assertEqual(maximum,r['score'])
            self.assertTrue(evaluate_board(p,r['board'])['valid'])

    def test_cutoff_input_and_empty(self):
        p,b=generated_problem(4,3,17)
        self.assertTrue(compose(p,b,[])['optimal'])
        self.assertEqual('stopped',compose(p,b,[b],should_stop=lambda:True)['termination'])
        with self.assertRaises(ValueError):
            compose(p,b,[b]*65)
        with self.assertRaises(ValueError):
            compose(p,b,[[0]*16])
