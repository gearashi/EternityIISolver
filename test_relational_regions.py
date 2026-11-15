"""Independent finite oracles for relational overlap propagation."""
from dataclasses import replace
import itertools,random,unittest
from exact_model import make_problem,generated_problem,check_solution,grid_edges
from relational_regions import propagate


def masks(p):return [sum(1<<s for s in d) for d in p.domains]


def odd_cycle():
    """Constructed binary-face CSP, not physical rotations: two squares disagree."""
    size=3;faces=[]
    for c in range(9):
        for bit in (0,1,0,1):
            outside=(c<3,c%3==2,c>=6,c%3==0)
            faces.append([0 if b else 1 for b in outside])
    ring=((0,1,1,3),(1,2,1,3),(2,5,2,0),(5,4,3,1),(4,3,3,1),(3,0,0,2))
    for index,(a,b,sa,sb) in enumerate(ring):
        for bit in range(2):
            faces[4*a+bit][sa]=bit+1
            faces[4*b+bit][sb]=(1-bit if index==2 else bit)+1
    ds=[(4*c,4*c+1) if c<6 else (4*c,) for c in range(9)]
    return make_problem(3,faces,{c:4*c for c in range(6,9)},ds)


class RelationalRegionTests(unittest.TestCase):
    def test_relations_reject_what_unary_tables_cannot(self):
        p=odd_cycle();groups=((0,1,3,4),(1,2,4,5))
        unary=propagate(p,regions=groups,relational=False)
        relational=propagate(p,regions=groups)
        self.assertTrue(unary['complete']);self.assertTrue(unary['consistent'])
        self.assertEqual(unary['removed_states'],0)
        self.assertTrue(relational['complete']);self.assertFalse(relational['consistent'])
        self.assertEqual(relational['termination'],'relational_contradiction')
        self.assertFalse(any(check_solution(p,board) for board in itertools.product(*p.domains)))

    def test_all_tiny_solutions_survive_both_routes(self):
        for seed in (101,211,307,419):
            p,answer=generated_problem(2,3,seed,clue_count=seed%2)
            solutions=[board for board in itertools.product(*p.domains) if check_solution(p,board)]
            for relational in (False,True):
                result=propagate(p,relational=relational)
                self.assertTrue(result['complete']);self.assertTrue(result['consistent'])
                for board in solutions:self.assertTrue(all(result['domains'][c]>>s&1 for c,s in enumerate(board)))

    def test_physical_three_by_three_domain_oracles(self):
        for seed in (101,211,307,419):
            p,answer=generated_problem(3,3,seed,clue_count=1);rng=random.Random(seed)
            ds=tuple(tuple(dict.fromkeys((s,rng.choice(p.domains[c])))) for c,s in enumerate(answer))
            p=replace(p,domains=ds);solutions=[b for b in itertools.product(*ds) if check_solution(p,b)]
            r=propagate(p)
            self.assertTrue(r['complete']);self.assertTrue(r['consistent']);self.assertTrue(solutions)
            for board in solutions:self.assertTrue(all(r['domains'][c]>>s&1 for c,s in enumerate(board)))

    def test_caps_and_stop_return_original_unmodified_domains(self):
        p=odd_cycle();ds=masks(p)
        for kwargs in ({'max_work':0},{'max_work':30},{'max_tuples':1},{'should_stop':lambda:True}):
            r=propagate(p,ds,**kwargs)
            self.assertFalse(r['complete']);self.assertTrue(r['consistent']);self.assertEqual(r['domains'],ds)
        self.assertEqual(ds,masks(p))

    def test_frames_fixed_pieces_and_edge_subset(self):
        p,answer=generated_problem(2,2,101,clue_count=1)
        r=propagate(p)
        for c,s in p.fixed.items():self.assertEqual(r['domains'][c],1<<s)
        q=replace(p,edges=())
        r=propagate(q)
        self.assertTrue(r['complete']);self.assertTrue(r['consistent'])
        self.assertTrue(all(r['domains'][c]>>s&1 for c,s in enumerate(answer)))

    def test_invalid_and_bounded_inputs(self):
        p,_=generated_problem(2,2,101)
        for kwargs in ({'regions':[[0,0]]},{'regions':[[4]]},{'regions':itertools.repeat([0])},
                       {'max_tuples':0},{'max_work':True},{'relational':1},{'should_stop':5}):
            with self.assertRaises(ValueError):propagate(p,**kwargs)
        for ds in ([1],[-1]*4,[True]*4,[1<<100]*4):
            with self.assertRaises(ValueError):propagate(p,ds)

if __name__=='__main__':unittest.main()
