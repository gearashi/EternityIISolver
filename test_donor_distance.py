import itertools
import unittest
from exact_model import generated_problem,check_solution
from regional_distance import DonorDistance


class DonorScopeChecks(unittest.TestCase):
    def test_rejects_out_of_board_indices_and_invalid_limits(self):
        problem,parent=generated_problem(2,2,13)
        oracle=DonorDistance(problem,parent)
        for cells in ((-1,),(4,),(),(0,0),(False,),range(100)):
            with self.assertRaises(ValueError):
                oracle.improve(cells,0)
        for options in ({'extra':-1},{'max_work':-1},{'max_work':True}):
            with self.assertRaises(ValueError):
                oracle.improve([0],0,**options)

    def test_tiny_physical_bounds_and_exhaustion(self):
        for seed in range(12):
            p,known=generated_problem(2,2,seed)
            boards=[b for b in itertools.product(*p.domains) if len({s//4 for s in b})==4]
            parent=boards[seed%len(boards)]
            valid=[b for b in boards if check_solution(p,b)]
            model=DonorDistance(p,parent)
            for cells in ((0,1,2),(0,1,2,3)):
                true=min(sum(parent[c]!=b[c] for c in cells) for b in valid)
                bounds=[]
                for mode in ('matching','edges'):
                    v,reason=model.improve(cells,0,extra=len(cells),level=mode)
                    self.assertLessEqual(v,true);bounds.append(v)
                self.assertLessEqual(bounds[0],bounds[1])
                if len(cells)==4:self.assertEqual(bounds[-1],true)
                self.assertEqual(model.improve(cells,0,max_work=0),(0,'work_limit'))
