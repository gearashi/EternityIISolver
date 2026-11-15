"""Independent finite oracles for piece-allocation/interface master cuts."""
from dataclasses import replace
import itertools
import random
import unittest
from unittest.mock import patch
from exact_model import generated_problem,check_solution,make_problem
from test_exact_sat import tiny_problem
import regional_master


def bad_corner_problem(seed=101):
    p,answer=generated_problem(3,1,seed)
    piece=answer[0]//4
    faces=list(p.faces)
    for s in range(4*piece,4*piece+4):faces[s]=tuple(9 if value else 0 for value in faces[s])
    return make_problem(3,faces)


class RegionalMasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Correctness budgets cover solving; native cold import is measured
        # separately by the benchmark and can vary with concurrent disk load.
        from ortools.sat.python import cp_model

    def test_tiny_complete_oracle_with_both_cut_scopes(self):
        for unsat in (False,True):
            p=tiny_problem(unsat)
            expected=any(check_solution(p,b) for b in itertools.product(*p.domains))
            for cuts in (False,True):
                r=regional_master.solve(p,seconds=5,max_master_queries=512,regional_cuts=cuts)
                self.assertEqual(r['outcome'],'solved' if expected else 'infeasible')
                if r['board'] is not None:self.assertTrue(check_solution(p,r['board']))
                self.assertEqual(r['branches'],r['local_placements'])
                self.assertEqual(r['reasoning_counters_session']['master_branches'],r['master_branches'])
                self.assertGreater(r['regional_cuts_added' if cuts else 'full_cuts_added'],0)

    def test_random_tiny_domains_agree_with_enumeration(self):
        rng=random.Random(8123)
        for seed in range(12):
            p,_=generated_problem(2,3,seed)
            p=replace(p,domains=tuple(tuple(s for s in d if rng.random()<.7) for d in p.domains))
            expected=any(check_solution(p,b) for b in itertools.product(*p.domains))
            r=regional_master.solve(p,seconds=3,max_master_queries=512)
            self.assertEqual(r['outcome'],'solved' if expected else 'infeasible',seed)
            if r['board'] is not None:self.assertTrue(check_solution(p,r['board']))

    def test_clues_orientation_and_domain_restrictions(self):
        p,answer=generated_problem(3,3,211,clue_count=3)
        # Retain a restricted known completion; all master allocations still
        # represent this full valid solution.
        p=replace(p,domains=tuple((answer[c],) for c in range(9)))
        r=regional_master.solve(p,seconds=3)
        self.assertEqual(r['outcome'],'solved');self.assertEqual(r['board'],answer)
        for c,s in p.fixed.items():self.assertEqual(r['board'][c],s)

    def test_edge_subset_and_nonrow_regions(self):
        base=tiny_problem(True)
        p=replace(base,edges=())
        r=regional_master.solve(p,seconds=3,regions=[[0,2],[1,3]])
        self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(p,r['board']))
        self.assertFalse(check_solution(base,r['board']))
        self.assertEqual(r['proof_scope'],'supplied-problem-edges-and-domains')

    def test_allocation_does_not_reuse_piece_between_regions(self):
        p=tiny_problem()
        # Each singleton region can independently place piece0, but globally
        # no allocation can give that physical piece to all four cells.
        p=replace(p,domains=tuple(tuple(s for s in d if s//4==0) for d in p.domains),edges=())
        r=regional_master.solve(p,seconds=3,regions=[[0],[1],[2],[3]])
        self.assertEqual(r['outcome'],'infeasible');self.assertIsNone(r['board'])

    def test_caps_and_cancellation_are_unknown_without_failure_cuts(self):
        p=tiny_problem()
        for kwargs,termination in (({'seconds':0},'timeout'),({'max_work':0},'work_limit'),
                                   ({'should_stop':lambda:True},'stopped')):
            r=regional_master.solve(p,**kwargs)
            self.assertEqual(r['termination'],termination);self.assertFalse(r['complete'])
            self.assertEqual(r['regional_cuts_added'],0);self.assertIsNone(r['board'])
        r=regional_master.solve(tiny_problem(True),seconds=3,max_master_queries=1)
        self.assertEqual(r['termination'],'query_limit');self.assertFalse(r['complete'])
        self.assertLessEqual(r['regional_cuts_added'],1)

    def test_unknown_inside_oracle_never_cuts(self):
        p=tiny_problem()
        r=regional_master.solve(p,seconds=3,max_work=63,max_master_queries=5)
        self.assertEqual(r['termination'],'work_limit');self.assertEqual(r['regional_cuts_added'],0)
        self.assertGreater(r['local_queries'],0)

    def test_verified_solution_at_stop_boundary_is_retained(self):
        p,answer=generated_problem(2,2,113)
        p=replace(p,domains=tuple((s,) for s in answer));stopped=[False]
        real=regional_master.check_solution
        def validate(problem,board):
            valid=real(problem,board);stopped[0]=True;return valid
        with patch.object(regional_master,'check_solution',side_effect=validate):
            r=regional_master.solve(p,seconds=3,should_stop=lambda:stopped[0])
        self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(p,r['board']))

    def test_input_validation_and_callback_error(self):
        p=tiny_problem()
        for kwargs in ({'regions':[]},{'regions':[[0,1],[2,2]]},{'regions':[[0,1],[2,True]]},
                       {'seconds':True},{'max_work':-1},{'max_master_queries':0},{'seed':True},
                       {'regional_cuts':1},{'should_stop':False},{'on_progress':False}):
            with self.assertRaises(ValueError):regional_master.solve(p,**kwargs)
        def bad(_):raise RuntimeError('caller failed')
        with self.assertRaisesRegex(RuntimeError,'caller failed'):regional_master.solve(p,on_progress=bad)

if __name__=='__main__':unittest.main()
