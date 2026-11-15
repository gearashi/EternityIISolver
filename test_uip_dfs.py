from dataclasses import replace
from itertools import product
from random import Random
import unittest
import exact_sat
from exact_model import generated_problem, check_solution, constraint_edges
from test_exact_sat import tiny_problem
from uip_dfs import solve


def holds(atom,b,p):
    if len(atom)==2:
        c,s=atom
        return b[c]==s
    kind,x,y=atom
    if kind=='row':return any(b[c]//4==y for c in range(x*p.size,(x+1)*p.size))
    if kind=='piece':return b[x]//4==y
    a,_,s,_=constraint_edges(p)[x]
    return p.faces[b[a]][s]==y


class UIPTests(unittest.TestCase):
    def test_every_learned_clause_holds_on_every_exact_board(self):
        cases=[tiny_problem(),tiny_problem(True),tiny_problem(True,edges=())]
        for seed in range(36):
            p,known=generated_problem(3,2+(seed%2),seed,clue_count=seed%3)
            rng=Random(213+seed)
            d=tuple(tuple(dict.fromkeys((k,rng.choice(dom),rng.choice(dom)))) for k,dom in zip(known,p.domains))
            if seed%3==2:d=tuple(tuple(s for s in d0 if (s+c+seed)%4) for c,d0 in enumerate(d))
            cases.append(replace(p,domains=d))
        for p in cases:
            answers=[b for b in product(*p.domains) if check_solution(p,b)]
            for learning,channels,ordering in ((False,False,'baseline'),(True,False,'baseline'),
                    (True,True,'baseline'),(True,False,'density'),(True,True,'adaptive')):
                r=solve(p,seconds=3,learning=learning,channels=channels,ordering=ordering,include_proofs=True)
                self.assertEqual(r['outcome'],'solved' if answers else 'infeasible')
                if answers:self.assertIn(tuple(r['board']), answers)
                for clause in r['clauses']:
                    for b in answers:
                        self.assertTrue(any(holds(r['atoms'][abs(l)],b,p)==(l>0) for l in clause))

    def test_nonvacuous_learned_clauses_verified_by_independent_sat_oracle(self):
        checked=mixed=0
        for seed in (313,321,263,3821):
            p,_=generated_problem(4,2,seed,clue_count=1)
            r=solve(p,seconds=3,channels=True,ordering='adaptive',include_proofs=True)
            self.assertEqual(r['outcome'],'solved')
            for clause in r['clauses']:
                ds=[set(d) for d in p.domains]
                for literal in clause:
                    atom=r['atoms'][abs(literal)]
                    if len(atom)==2:
                        c,s=atom
                        ds[c]={v for v in ds[c] if (v==s)==(literal<0)}
                        continue
                    kind,x,y=atom
                    if kind=='piece':
                        ds[x]={v for v in ds[x] if (v//4==y)==(literal<0)}
                    elif kind=='row':
                        for c,d in enumerate(ds):
                            if (c//p.size==x)==(literal>0):
                                ds[c]={v for v in d if v//4!=y}
                    elif kind=='color':
                        a,_,side,_=constraint_edges(p)[x]
                        ds[a]={v for v in ds[a] if (p.faces[v][side]==y)==(literal<0)}
                q=replace(p,domains=tuple(tuple(sorted(d)) for d in ds))
                self.assertEqual(exact_sat.solve(q,seconds=2)['outcome'],'infeasible')
                checked+=1
                mixed+=any(len(r['atoms'][abs(l)])>2 for l in clause)
        self.assertGreater(checked,0)
        self.assertGreater(mixed,0)

    def test_real_learning_and_capped_unknown(self):
        actual=0
        for seed in (321,313,263):
            p,_=generated_problem(4,3,seed,clue_count=1)
            r=solve(p,seconds=4,include_proofs=True)
            self.assertEqual(r['outcome'],'solved')
            actual+=r['learned_clauses']
        self.assertGreater(actual,0)
        self.assertEqual(solve(p,seconds=0)['outcome'],'timeout')
        self.assertEqual(solve(p,max_work=1)['outcome'],'timeout')
