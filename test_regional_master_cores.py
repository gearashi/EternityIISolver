"""Proof and resource-bound checks for smaller regional interface cuts."""
from dataclasses import replace
import itertools
import random
import unittest
from unittest.mock import patch

from exact_model import generated_problem, check_solution, constraint_edges
from test_exact_sat import tiny_problem
import regional_master


class RegionalMasterCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ortools.sat.python import cp_model
        cls.cp_model=cp_model

    def test_all_complete_boards_survive_every_generated_cut(self):
        rng=random.Random(8291)
        original=self.cp_model.CpModel.add_forbidden_assignments
        for seed in range(8):
            p,_=generated_problem(2,3,seed)
            if seed%2:p=replace(p,domains=tuple(tuple(s for s in d if rng.random()<.8) for d in p.domains))
            solutions=[b for b in itertools.product(*p.domains) if check_solution(p,b)]
            regions=((0,2),(1,3));where={c:r for r,g in enumerate(regions) for c in g}
            cross=[e for e in constraint_edges(p) if where[e[0]]!=where[e[1]]]
            cuts=[]
            def capture(model,variables,rows):
                cuts.extend((tuple(v.name for v in variables),tuple(row)) for row in rows)
                return original(model,variables,rows)
            with patch.object(self.cp_model.CpModel,'add_forbidden_assignments',capture):
                result=regional_master.solve(p,seconds=5,seed=seed,regions=regions,
                    max_master_queries=1024,max_work=400000,minimize_interfaces=True)
            self.assertEqual(result['outcome'],'solved' if solutions else 'infeasible')
            for board in solutions:
                assignment={f'p{s//4}':where[c] for c,s in enumerate(board)}
                assignment.update({f'e{i}':p.faces[board[a]][sa] for i,(a,b,sa,sb) in enumerate(cross)})
                for names,values in cuts:
                    self.assertNotEqual(tuple(assignment[name] for name in names),values)
            if result['board'] is not None:self.assertTrue(check_solution(p,result['board']))
            self.assertLessEqual(result['explanation_work'],result['oracle_work'])
            self.assertGreaterEqual(result['interface_literals_removed'],0)
            self.assertEqual(result['work'],result['setup_work']+result['oracle_work']+result['cut_work'])

    def test_each_extracted_core_independently_proves_local_infeasibility(self):
        original=self.cp_model.CpModel.add_forbidden_assignments
        for seed in (101,211,307,419):
            p,_=generated_problem(3,2,seed,clue_count=1)
            regions=((0,1,2),(3,4,5),(6,7,8));where={c:r for r,g in enumerate(regions) for c in g}
            cross=[e for e in constraint_edges(p) if where[e[0]]!=where[e[1]]]
            cuts=[]
            def capture(model,variables,rows):
                cuts.extend((tuple(v.name for v in variables),tuple(row)) for row in rows)
                return original(model,variables,rows)
            with patch.object(self.cp_model.CpModel,'add_forbidden_assignments',capture):
                result=regional_master.solve(p,seconds=5,seed=seed,regions=regions,
                    max_master_queries=1024,max_work=500000,minimize_interfaces=True)
            self.assertEqual(result['outcome'],'solved')
            self.assertTrue(check_solution(p,result['board']))
            for names,values in cuts:
                literals=dict(zip(names,values));pool={int(name[1:]) for name in names if name.startswith('p')}
                region=next(value for name,value in literals.items() if name.startswith('p'))
                cells=regions[region]
                # An independent finite enumeration of precisely the cut's
                # allocated pool, retained boundaries, and internal edges.
                domains=[tuple(s for s in p.domains[c] if s//4 in pool and
                    (c not in p.fixed or p.fixed[c]==s)) for c in cells]
                for states in itertools.product(*domains):
                    if len({s//4 for s in states})!=len(cells):continue
                    board=dict(zip(cells,states))
                    if any(p.faces[board[a]][sa]!=p.faces[board[b]][sb] for a,b,sa,sb in constraint_edges(p)
                           if a in board and b in board):continue
                    boundary_ok=True
                    for name,color in literals.items():
                        if not name.startswith('e'):continue
                        a,b,sa,sb=cross[int(name[1:])]
                        c,side=(a,sa) if a in board else (b,sb)
                        if p.faces[board[c]][side]!=color:boundary_ok=False;break
                    self.assertFalse(boundary_ok,(seed,literals,board))

    def test_minimization_actually_drops_irrelevant_interface_conditions(self):
        result=regional_master.solve(tiny_problem(True),seconds=5,max_master_queries=1024,
                                    max_work=400000,minimize_interfaces=True)
        self.assertEqual(result['outcome'],'infeasible')
        self.assertGreater(result['interface_literals_removed'],0)
        self.assertGreater(result['extracted_cores'],0)

    def test_cancel_inside_local_proof_does_not_add_an_unproved_cut(self):
        phase=[''];checks=[0]
        def progress(row):phase[0]=row['phase']
        def stop():
            if phase[0]=='regional_oracles':checks[0]+=1
            return checks[0]>=7
        result=regional_master.solve(tiny_problem(True),seconds=5,max_work=400000,
            minimize_interfaces=True,on_progress=progress,should_stop=stop)
        self.assertEqual(result['outcome'],'stopped')
        self.assertGreater(result['local_queries'],0)
        self.assertEqual(result['regional_cuts_added'],0)
        self.assertFalse(result['complete']);self.assertIsNone(result['board'])
        self.assertLessEqual(result['explanation_work'],result['oracle_work'])

    def test_clues_domains_and_empty_edge_scope(self):
        p,answer=generated_problem(3,2,211,clue_count=2)
        p=replace(p,domains=tuple((s,) for s in answer))
        result=regional_master.solve(p,seconds=5,minimize_interfaces=True)
        self.assertEqual(result['outcome'],'solved');self.assertEqual(result['board'],answer)
        p=replace(tiny_problem(True),edges=())
        result=regional_master.solve(p,seconds=5,regions=((0,),(1,2),(3,)),minimize_interfaces=True)
        self.assertEqual(result['outcome'],'solved');self.assertTrue(check_solution(p,result['board']))

    def test_option_validation(self):
        for settings in ({'minimize_interfaces':1},{'minimize_interfaces':True,'regional_cuts':False}):
            with self.assertRaises(ValueError):regional_master.solve(tiny_problem(),**settings)


if __name__=='__main__':unittest.main()
