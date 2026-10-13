import itertools
import unittest
from exact_model import generated_problem,grid_edges,make_problem
from smart_repair import structurally_valid,score
from mismatch_bounds import mismatch_lower_bound,can_beat

class MismatchBoundTests(unittest.TestCase):
    def test_relaxation_never_exceeds_exhaustive_optimum(self):
        for seed in range(8):
            p,_=generated_problem(2,3,seed)
            boards=[list(b) for b in itertools.product(*p.domains) if structurally_valid(p,b)]
            optimum=min(4-score(p,b) for b in boards)
            for cap in (0,4096):
                result=mismatch_lower_bound(p,square_limit=cap)
                self.assertLessEqual(result['lower_bound'],optimum)
                if cap: self.assertEqual(result['lower_bound'],optimum)
                ids=[i for g in result['groups'] for i in g['edges']]
                self.assertEqual(sorted(ids),list(range(4)))

    def test_fixed_mismatched_board_bound_is_exact(self):
        p,board=generated_problem(3,3,1)
        # Use a legal alternative permutation of two corners.
        a,b=0,2;pa,pb=board[a]//4,board[b]//4
        board[a]=next(s for s in p.domains[a] if s//4==pb)
        board[b]=next(s for s in p.domains[b] if s//4==pa)
        q=make_problem(3,p.faces,domains=[(s,) for s in board])
        result=can_beat(q,score(q,board))
        self.assertEqual(result['lower_bound'],12-score(q,board))
        self.assertFalse(result['can_beat_bound'])
        self.assertFalse(result['structurally_infeasible'])

    def test_impossible_inventory_and_domain_subset_guard(self):
        p,board=generated_problem(2,2,7)
        bad=make_problem(2,p.faces,domains=[p.domains[0]]*4)
        self.assertTrue(mismatch_lower_bound(bad)['structurally_infeasible'])
        with self.assertRaises(ValueError): mismatch_lower_bound(p,domains=[(999,)]*4)

    def test_square_strictly_strengthens_separate_edge_bounds(self):
        faces=[]
        for east,south in ((1,1),(1,1),(1,2),(2,2)):
            base=(0,east,south,0)
            faces.extend(tuple(base[(side-r)%4] for side in range(4)) for r in range(4))
        p=make_problem(2,faces)
        cheap=mismatch_lower_bound(p,square_limit=0)
        strong=mismatch_lower_bound(p)
        optimum=min(4-score(p,b) for b in itertools.product(*p.domains) if structurally_valid(p,b))
        self.assertEqual(cheap['lower_bound'],0)
        self.assertEqual(strong['lower_bound'],optimum)
        self.assertGreater(optimum,0)

    def test_optimizer_uses_scoped_bound_before_native_search(self):
        from exact_optimize import optimize
        faces=[]
        for east,south in ((1,1),(1,1),(1,2),(2,2)):
            base=(0,east,south,0)
            faces.extend(tuple(base[(side-r)%4] for side in range(4)) for r in range(4))
        p=make_problem(2,faces)
        reference=max((list(b) for b in itertools.product(*p.domains) if structurally_valid(p,b)),
                      key=lambda b:score(p,b))
        bounded=optimize(p,reference,seconds=1,regional_bounds=True)
        native=optimize(p,reference,seconds=1,regional_bounds=False)
        self.assertEqual(bounded['termination'],'admissible_mismatch_bound')
        self.assertEqual(bounded['outcome'],'infeasible_for_target')
        self.assertEqual(bounded['outcome'],native['outcome'])
        self.assertEqual(bounded['branches'],0)
        self.assertEqual(bounded['board'],reference)

    def test_interrupt_never_returns_partial_bound(self):
        p,_=generated_problem(3,3,8)
        calls=[0]
        def check():
            calls[0]+=1
            if calls[0]>2: raise InterruptedError()
        with self.assertRaises(InterruptedError): mismatch_lower_bound(p,check=check)

if __name__=='__main__':unittest.main()
