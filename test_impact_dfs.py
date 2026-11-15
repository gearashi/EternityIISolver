from dataclasses import replace
import itertools
import random
import unittest
from exact_model import generated_problem,check_solution
from exact_dfs import solve as baseline
from impact_dfs import solve, _ImpactSearch
from exact_dfs import _Interrupted


def oracle(p):
    return next((list(b) for b in itertools.product(*p.domains) if check_solution(p,b)),None)


class ImpactSearchTests(unittest.TestCase):
    def test_exhaustive_physical_cases_and_complete_negative_proofs(self):
        outcomes=set()
        for seed in range(24):
            p,_=generated_problem(2,2+seed%2,seed,clue_count=seed%2)
            rng=random.Random(800+seed)
            p=replace(p,domains=tuple(tuple(s for s in d if rng.random()<.65) for d in p.domains))
            expected=oracle(p)
            for ordering in ('baseline','impact'):
                for discrepancy in (None,0,2):
                    r=solve(p,seconds=3,seed=seed,ordering=ordering,discrepancy_limit=discrepancy)
                    outcomes.add(r['outcome'])
                    self.assertEqual(r['outcome'],'solved' if expected is not None else 'infeasible',(seed,ordering,discrepancy,r))
                    self.assertEqual(r['coverage_verified'],expected is None)
                    if r['board'] is not None:self.assertTrue(check_solution(p,r['board']))
        self.assertEqual(outcomes,{'solved','infeasible'})

    def test_ablation_preserves_established_baseline(self):
        for size in (2,3,4):
            p,_=generated_problem(size,3,71,clue_count=1)
            a=baseline(p,seed=13,seconds=3)
            b=solve(p,seed=13,seconds=3,ordering='baseline',discrepancy_limit=None)
            for key in ('outcome','board','branches','conflicts','propagations','matching_checks'):
                self.assertEqual(a[key],b[key],(size,key))
            self.assertEqual(b['reasoning_counters_session']['impact_operations'],0)

    def test_scoped_edges_and_fixed_orientations(self):
        p,_=generated_problem(3,3,29,clue_count=2)
        for edges in (None,()):
            p=replace(p,edges=edges)
            r=solve(p,seconds=3)
            self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(p,r['board']))
            self.assertTrue(all(r['board'][c]==s for c,s in p.fixed.items()))

    def test_limits_never_claim_infeasibility(self):
        p,_=generated_problem(4,3,41)
        for kwargs in ({'seconds':0},{'max_work':0},{'max_work':10},{'should_stop':lambda:True}):
            r=solve(p,**kwargs)
            self.assertIn(r['outcome'],('timeout','stopped'))
            self.assertFalse(r['coverage_verified']);self.assertIsNone(r['board']);self.assertIsNone(r['checkpoint'])

    def test_reserved_work_limit_and_optional_probe_accounting(self):
        p,_=generated_problem(3,3,51)
        search=_ImpactSearch(p,3,0,None,None,None,None,'impact',2,10)
        search.work_ticks=1
        with self.assertRaises(_Interrupted):search.charge(20)
        self.assertTrue(search.limit_hit);self.assertEqual(search.impact_work,0)
        a=baseline(p,seconds=3,options={'probe':True})
        b=solve(p,seconds=3,options={'probe':True},ordering='baseline',discrepancy_limit=None)
        self.assertEqual(a['board'],b['board'])
        self.assertEqual(a['branches']+a['probe_checks'],b['branches'])

    def test_invalid_configuration(self):
        p,_=generated_problem(2,2,10)
        for kwargs in ({'ordering':'bad'},{'discrepancy_limit':True},{'discrepancy_limit':-1},
                       {'workers':True},{'workers':2},{'max_work':True},{'seconds':float('nan')},
                       {'seed':True},{'should_stop':1}):
            with self.assertRaises(ValueError):solve(p,**kwargs)

    def test_history_work_visible(self):
        p,_=generated_problem(4,2,13)
        r=solve(p,seconds=3)
        c=r['reasoning_counters_session']
        self.assertEqual(r['work'],c['dfs_check_calls']+c['impact_operations'])
        self.assertGreater(c['impact_operations'],0);self.assertGreater(c['observed_branches'],0)
        self.assertGreaterEqual(r['branches'],c['observed_branches'])

if __name__=='__main__':unittest.main()
