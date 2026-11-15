"""Exact mismatch-oracle checks for overlapping cost partitions."""
from dataclasses import replace
from fractions import Fraction
import itertools,random,unittest
from exact_model import generated_problem,grid_edges
from exact_optimize import evaluate_board
from overlap_bounds import lower_bound
from test_relational_regions import odd_cycle,masks


class OverlapBoundTests(unittest.TestCase):
    def validate_certificate(self,p,r):
        weights=[{int(e):Fraction(*v) for e,v in row.items()} for row in r['cost_partition']]
        for e in range(len(grid_edges(p.size))):self.assertLessEqual(sum(row.get(e,0) for row in weights),1)
        self.assertEqual(sum((Fraction(*v) for v in r['region_minima']),Fraction(0)),Fraction(*r['rational_bound']))

    def test_all_tiny_structural_boards_bound_true_optimum(self):
        for seed in (101,211,307,419):
            p,_=generated_problem(2,3,seed)
            optimum=min(4-v['score'] for b in itertools.product(*p.domains) if (v:=evaluate_board(p,b))['valid'])
            for strategy in ('max','disjoint','uniform','optimized'):
                r=lower_bound(p,strategy=strategy,max_work=200000)
                self.assertTrue(r['complete']);self.assertLessEqual(r['lower_bound'],optimum)
                self.validate_certificate(p,r)

    def test_physical_narrowed_three_by_three_oracle(self):
        for seed in (101,211,307,419):
            p,answer=generated_problem(3,3,seed);rng=random.Random(seed)
            ds=tuple(tuple(dict.fromkeys((s,rng.choice(p.domains[c])))) for c,s in enumerate(answer))
            p=replace(p,domains=ds)
            optimum=min(12-v['score'] for b in itertools.product(*ds) if (v:=evaluate_board(p,b))['valid'])
            scores=[]
            for strategy in ('max','disjoint','uniform','optimized'):
                r=lower_bound(p,strategy=strategy,max_work=500000)
                self.assertTrue(r['complete']);self.assertLessEqual(r['lower_bound'],optimum)
                self.validate_certificate(p,r);scores.append(r['lower_bound'])
            self.assertGreaterEqual(scores[-1],max(scores[:-1]))

    def test_overlap_never_double_counts_same_mismatch(self):
        p=odd_cycle();ds=masks(p)
        # Fix a structural board with one mismatch in an edge shared by squares.
        board=[4*c for c in range(9)]
        ds=[1<<s for s in board]
        score=evaluate_board(p,board)['score']
        for strategy in ('max','disjoint','uniform','optimized'):
            r=lower_bound(p,ds,strategy=strategy,max_work=100000)
            self.assertTrue(r['complete']);self.assertLessEqual(r['lower_bound'],12-score)
            self.validate_certificate(p,r)
        self.assertEqual(r['lower_bound'],12-score)

    def test_partition_improves_disjoint_on_physical_negative_fixture(self):
        p,answer=generated_problem(3,3,101);rng=random.Random(1101);groups={}
        for c,s in enumerate(answer):groups.setdefault(sum(v==0 for v in p.faces[s]),[]).append(c)
        board=list(answer)
        for group in groups.values():
            pieces=[answer[c]//4 for c in group];rng.shuffle(pieces)
            for c,piece in zip(group,pieces):board[c]=rng.choice([s for s in p.domains[c] if s//4==piece])
        ds=tuple(tuple(dict.fromkeys((s,rng.choice(p.domains[c])))) for c,s in enumerate(board))
        p=replace(p,domains=ds)
        optimum=min(12-v['score'] for b in itertools.product(*ds) if (v:=evaluate_board(p,b))['valid'])
        bounds=[lower_bound(p,strategy=mode,max_work=100000) for mode in ('max','disjoint','uniform','optimized')]
        self.assertEqual([r['lower_bound'] for r in bounds],[2,3,4,4]);self.assertEqual(optimum,5)
        for r in bounds:self.assertTrue(r['complete']);self.validate_certificate(p,r)
        self.assertGreater(bounds[-1]['accepted_transfers'],0)
        # Domain-fixed exhaustive optimum is independent of region tables.
        self.assertTrue(all(r['lower_bound']<=optimum for r in bounds))

    def test_caps_never_return_a_bound(self):
        p=odd_cycle()
        for kwargs in ({'max_work':0},{'max_work':15},{'max_tuples':1},{'should_stop':lambda:True}):
            r=lower_bound(p,**kwargs)
            self.assertFalse(r['complete']);self.assertIsNone(r['lower_bound']);self.assertFalse(r['structurally_infeasible'])

    def test_empty_domain_is_structural_failure(self):
        p=odd_cycle();ds=masks(p);ds[0]=0
        r=lower_bound(p,ds)
        self.assertTrue(r['complete']);self.assertTrue(r['structurally_infeasible']);self.assertIsNone(r['lower_bound'])

    def test_invalid_strategy_and_empty_regions(self):
        p=odd_cycle()
        with self.assertRaises(ValueError):lower_bound(p,strategy='untrusted')
        r=lower_bound(p,regions=[])
        self.assertTrue(r['complete']);self.assertEqual(r['lower_bound'],0)

if __name__=='__main__':unittest.main()
