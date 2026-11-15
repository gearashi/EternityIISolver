import itertools
import random
import unittest

from exact_model import generated_problem
from exact_optimize import evaluate_board
from repair_multisource import multi_parent_recombine
from repair_rotations import lift_rows


def families(seed,cells):
    p,answer=generated_problem(5,7,seed)
    rng=random.Random(seed)
    parents=[]
    for _ in range(3):
        order=list(cells)
        rng.shuffle(order)
        b=answer.copy()
        for c,d in zip(cells,order):
            b[c]=(answer[d]//4)*4+rng.randrange(4)
        parents.append(b)
    return p,parents


class LiftChecks(unittest.TestCase):
    def test_actual_optimum_cannot_decrease_when_lifting(self):
        for seed in range(10):
            for cells in ((6,7,8),(6,8,16,18),(6,7,11,12)):
                p,parents=families(seed,cells)
                ref=max(parents,key=lambda b:evaluate_board(p,b)['score'])
                original={c:sorted({b[c] for b in parents}) for c in cells}
                lifted,warm,counts=lift_rows(p,ref,cells,original)
                def optimum(allow):
                    maximum=0
                    for combo in itertools.product(*(allow[c] for c in cells)):
                        b=ref.copy()
                        for c,s in zip(cells,combo):
                            b[c]=s
                        value=evaluate_board(p,b)
                        if value['valid']:
                            maximum=max(maximum,value['score'])
                    return maximum
                self.assertLessEqual(counts['rows_after'],counts['rows_before'])
                self.assertGreaterEqual(optimum(lifted),optimum(original))
                self.assertTrue(evaluate_board(p,warm)['valid'])
                self.assertGreaterEqual(evaluate_board(p,warm)['score'],evaluate_board(p,ref)['score'])

    def test_unseen_legal_rotations_repair_consensus_errors(self):
        p,answer=generated_problem(5,20,84)
        cells=(6,8,16)
        b=answer.copy()
        for c in cells:
            b[c]=b[c]//4*4+(b[c]+1)%4
        run=multi_parent_recombine(p,[b,b,b],cells=cells,seconds=10,lift_rotations=True)
        self.assertEqual(run['score'],40)
        self.assertEqual(run['branches'],0)
        self.assertGreater(run['lift_work']['orientation_tests'],0)
        self.assertTrue(run['improved'])
        limited=multi_parent_recombine(p,[b,b,b],cells=cells,lift_rotations=True,seconds=0)
        self.assertEqual(limited['termination'],'timeout')
        self.assertEqual(limited['board'],b)
