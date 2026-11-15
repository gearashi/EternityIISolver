"""Independent finite oracles for explained pruning, learning and backjumping."""
from dataclasses import replace
import itertools
import random
import unittest
from unittest.mock import patch

from exact_model import ExactProblem,check_solution,generated_problem,make_problem
import explained_dfs
from test_exact_sat import tiny_problem
from test_residual_components import binary_components


def oracle(p):
    return [list(b) for b in itertools.product(*p.domains) if check_solution(p,b)]


def satisfiable_reason_fixture():
    p,squares=binary_components(True,1)
    c=squares[0][-1];piece=p.domains[c][0]//4
    faces=list(p.faces);face=list(faces[4*piece+1]);face[0]=3
    faces[4*piece+2]=tuple(face)
    ds=list(p.domains);ds[c]=ds[c]+(4*piece+2,)
    return replace(p,faces=tuple(faces),domains=tuple(ds))


class ExplainedDfsTests(unittest.TestCase):
    def test_tiny_exhaustive_domain_and_edge_oracles(self):
        rng=random.Random(51491)
        for seed in range(24):
            p,_=generated_problem(2,3,seed)
            ds=tuple(tuple(s for s in d if rng.random()<.7) for d in p.domains)
            edges=tuple(e for e in __import__('exact_model').grid_edges(2) if rng.random()<.7)
            p=replace(p,domains=ds,edges=edges)
            wanted=oracle(p)
            for learning in (False,True):
                r=explained_dfs.solve(p,seconds=3,seed=seed,learning=learning,include_nogoods=True)
                self.assertEqual(r['outcome'],'solved' if wanted else 'infeasible')
                self.assertTrue(r['coverage_verified'])
                if wanted:self.assertTrue(check_solution(p,r['board']))
                for clause in r['learned_nogoods']:
                    self.assertFalse(any(all(b[c]==s for c,s in clause) for b in wanted))

    def test_real_rotations_clues_and_frame(self):
        for size,colors,seed in ((1,1,1),(2,3,73),(3,5,91),(4,5,17)):
            p,known=generated_problem(size,colors,seed,clue_count=min(size,3))
            r=explained_dfs.solve(p,seconds=5,seed=seed,max_work=5000000)
            self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(p,r['board']))
            self.assertTrue(all(r['board'][c]==s for c,s in p.fixed.items()))

    def test_retained_nogoods_exclude_no_actual_solution(self):
        p=satisfiable_reason_fixture();solutions=oracle(p)
        self.assertTrue(solutions)
        r=explained_dfs.solve(p,seconds=4,seed=101,include_nogoods=True)
        self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(p,r['board']))
        self.assertGreater(r['conflicts'],0);self.assertGreater(len(r['learned_nogoods']),0)
        for clause in r['learned_nogoods']:
            self.assertFalse(any(all(b[c]==s for c,s in clause) for b in solutions),clause)

    def test_backjump_skips_unrelated_decisions(self):
        p,_=binary_components(True,4)
        off=explained_dfs.solve(p,seconds=5,seed=101,learning=False,max_work=5000000)
        on=explained_dfs.solve(p,seconds=5,seed=101,learning=True,max_work=5000000)
        self.assertEqual(off['outcome'],'infeasible');self.assertEqual(on['outcome'],'infeasible')
        self.assertLess(on['branches'],off['branches'])
        self.assertGreater(on['reasoning_counters_session']['backjumped_levels'],0)
        self.assertLess(on['work'],off['work'])

    def test_omitted_edges_are_not_reintroduced(self):
        p=tiny_problem(True,edges=(),fixed={0:0,1:4,2:8,3:12})
        r=explained_dfs.solve(p,seconds=2)
        self.assertEqual(r['outcome'],'solved');self.assertEqual(r['board'],[0,4,8,12])
        for edge,outcome in ((((0,1,1,3),),'solved'),(((2,3,1,3),),'infeasible')):
            r=explained_dfs.solve(replace(p,edges=edge),seconds=2)
            self.assertEqual(r['outcome'],outcome)

    def test_duplicate_piece_and_missing_piece_fail(self):
        p=tiny_problem(edges=(),fixed={0:0,1:1})
        r=explained_dfs.solve(p,seconds=2);self.assertEqual(r['outcome'],'infeasible')
        p=tiny_problem(edges=())
        ds=tuple(tuple(s for s in d if s//4!=3) for d in p.domains)
        r=explained_dfs.solve(replace(p,domains=ds),seconds=2)
        self.assertEqual(r['outcome'],'infeasible')

    def test_temporary_cap_and_stop_are_never_proofs(self):
        p=tiny_problem()
        for kwargs in ({'seconds':0},{'max_work':0},{'max_work':55},{'should_stop':lambda:True}):
            r=explained_dfs.solve(p,**kwargs)
            self.assertIn(r['outcome'],('timeout','stopped'))
            self.assertFalse(r['coverage_verified']);self.assertIsNone(r['board'])
            self.assertLessEqual(r['work'],r['work_limit'])
        self.assertEqual(explained_dfs.solve(p,seconds=2)['outcome'],'solved')

    def test_memory_caps_do_not_change_proof_and_no_cross_problem_cache(self):
        p,_=binary_components(True,3)
        for cap in (0,1,4):
            r=explained_dfs.solve(p,seconds=5,max_work=5000000,max_nogoods=cap,max_nogood_literals=1)
            self.assertEqual(r['outcome'],'infeasible');self.assertLessEqual(r['retained_nogoods'],cap)
        r=explained_dfs.solve(satisfiable_reason_fixture(),seconds=4)
        self.assertEqual(r['outcome'],'solved')

    def test_root_resolution_includes_previously_removed_values(self):
        # Random restricted relation tables stress multi-step reasons and branch
        # resolution. Every base removal, forced piece and cycle is oracle checked.
        rng=random.Random(6301)
        for seed in range(20):
            p,squares=binary_components(False,1);cells=squares[0]
            faces=list(p.faces);ds=list(p.domains)
            for c in cells:
                piece=ds[c][0]//4;ds[c]=tuple(4*piece+i for i in range(3))
                for s in ds[c]:
                    face=[1]*4
                    for a,b,sa,sb in __import__('exact_model').grid_edges(8):
                        if a==c and b in cells:face[sa]=rng.randint(2,3)
                        if b==c and a in cells:face[sb]=rng.randint(2,3)
                    faces[s]=tuple(face)
            p=replace(p,faces=tuple(faces),domains=tuple(ds));wanted=oracle(p)
            r=explained_dfs.solve(p,seconds=3,seed=seed,include_nogoods=True,max_work=2000000)
            self.assertEqual(r['outcome'],'solved' if wanted else 'infeasible')
            for clause in r['learned_nogoods']:
                self.assertFalse(any(all(b[c]==s for c,s in clause) for b in wanted))

    def test_snapshot_and_callback_propagation(self):
        p=tiny_problem();mutable=replace(p,faces=[list(f) for f in p.faces],domains=[list(d) for d in p.domains],fixed=dict(p.fixed))
        search=explained_dfs._Search(mutable,2,0,1,None,None,True,100000,256,64,False)
        mutable.faces[0][0]=255;mutable.domains[0].clear();mutable.fixed[0]=13
        result=search.run();self.assertEqual(result['outcome'],'solved');self.assertTrue(check_solution(p,result['board']))
        def broken():raise RuntimeError('stop failure')
        with self.assertRaisesRegex(RuntimeError,'stop failure'):explained_dfs.solve(p,should_stop=broken)
        def progress(r):raise RuntimeError('progress failure')
        with self.assertRaisesRegex(RuntimeError,'progress failure'):explained_dfs.solve(p,on_progress=progress)

    def test_invalid_parameters_and_final_progress_counters(self):
        p=tiny_problem()
        for kwargs in ({'seconds':True},{'seconds':float('inf')},{'seed':True},{'workers':0},
                       {'max_work':-1},{'learning':1},{'max_nogoods':-1},{'max_nogood_literals':0},
                       {'should_stop':3},{'on_progress':3},{'include_nogoods':1}):
            with self.assertRaises(ValueError):explained_dfs.solve(p,**kwargs)
        progress=[];r=explained_dfs.solve(p,on_progress=progress.append,workers=4)
        self.assertEqual(r,progress[-1]);self.assertEqual(r['workers'],1)
        self.assertEqual(r['requested_workers'],4)
        self.assertEqual(r['branches'],r['reasoning_counters_session']['decision_placements'])
        numeric=r['reasoning_counters_session'];self.assertTrue(all(type(v) is int for v in numeric.values()))
        self.assertEqual(r['work'],sum(v for k,v in numeric.items() if k in ('setup_state_visits','domain_copies','reason_copies','arc_state_checks','piece_state_checks','reason_literals','reason_joins','removed_states','nogood_literal_checks','decision_placements','propagation_rounds')))

if __name__=='__main__':unittest.main()
