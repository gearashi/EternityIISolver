"""Exhaustive tiny oracles for generalized score failures; no real puzzle run."""
from dataclasses import replace
import itertools
import unittest
from unittest.mock import patch
from exact_model import check_solution, generated_problem
from exact_optimize import evaluate_board
from repair_certificates import FixedTargetScoreSession,GeneralizedFailureCache
from test_exact_sat import tiny_problem
from test_exact_sat_optimize import boards


class CertificateTests(unittest.TestCase):
    def fixture(self):
        p=tiny_problem();legal=sorted(boards(p));return p,legal,legal[0][1]

    def assert_proof(self,legal,proof):
        matching=[s for s,b in legal if all(b[c]==v for c,v in proof['placements'].items())]
        self.assertTrue(all(s<proof['target_score'] for s in matching))

    def test_score_threshold_is_hard_and_frozen_cells_are_temporary(self):
        p,legal,ref=self.fixture()
        with FixedTargetScoreSession(p,4,seconds=3) as s:
            no=s.solve_frozen(dict(enumerate(ref)))
            self.assertEqual(no['outcome'],'infeasible')
            self.assertTrue(no['core_placements'])
            yes=s.solve_frozen({})
            self.assertEqual(yes['outcome'],'solved');self.assertTrue(check_solution(p,yes['board']))
            s.target_score=1
            with self.assertRaises(ValueError):s.solve_frozen({})

    def test_generalized_hit_with_different_region_and_exact_ablation_miss(self):
        p,legal,ref=self.fixture()
        with GeneralizedFailureCache(p) as g,GeneralizedFailureCache(p,generalize=False) as exact:
            first=g.solve(ref,[],target_score=4,seconds=3)
            exact.solve(ref,[],target_score=4,seconds=3)
            unused=next(c for c in range(4) if c not in first['core_placements'])
            hit=g.solve(ref,[unused],target_score=4,seconds=3)
            miss=exact.solve(ref,[unused],target_score=4,seconds=3)
            self.assertTrue(hit['cache_hit']);self.assertFalse(miss['cache_hit'])
            self.assertEqual(hit['native_queries'],0);self.assertGreater(miss['native_queries'],0)
            self.assert_proof(legal,g.certificates()[0])

    def test_all_tiny_contexts_match_exhaustive_target_oracle(self):
        p,legal,ref=self.fixture()
        with GeneralizedFailureCache(p,capacity=64) as g:
            for target in (2,4):
                for _,reference in legal:
                    for free in ([],[0],[1],[2],[3],[0,1],[2,3],list(range(4))):
                        frozen={c:s for c,s in enumerate(reference) if c not in free}
                        maximum=max(s for s,b in legal if all(b[c]==v for c,v in frozen.items()))
                        out=g.solve(reference,free,target_score=target,seconds=2,seed=0)
                        if out['proven_unreachable']:
                            self.assertLess(maximum,target)
                        else:
                            self.assertGreaterEqual(out['score'],target)
                            self.assertTrue(evaluate_board(p,out['board'])['valid'])
                            self.assertTrue(all(out['board'][c]==v for c,v in frozen.items()))
            for proof in g.certificates():self.assert_proof(legal,proof)

    def test_unreachable_high_target_does_not_claim_no_lower_improvement(self):
        p=tiny_problem(True);legal=sorted(boards(p));ref=legal[0][1]
        self.assertLess(evaluate_board(p,ref)['score'],3)
        with GeneralizedFailureCache(p) as g:
            high=g.solve(ref,list(range(4)),target_score=4,seconds=3)
            self.assertTrue(high['proven_unreachable'])
            self.assertEqual(high['termination'],'target_unreachable')
            self.assertEqual(high['certificate_target'],4);self.assertFalse(high['optimal'])
            hit=g.solve(ref,list(range(4)),target_score=4,seconds=3)
            self.assertTrue(hit['cache_hit']);self.assertFalse(hit['optimal'])
            self.assertEqual(hit['termination'],'target_unreachable')
            lower=g.solve(ref,list(range(4)),target_score=3,seconds=3)
            self.assertEqual(lower['score'],3);self.assertTrue(lower['improved'])
            next_score=g.solve(lower['board'],list(range(4)),target_score=4,seconds=3)
            self.assertTrue(next_score['cache_hit']);self.assertTrue(next_score['optimal'])
            self.assertEqual(next_score['termination'],'no_improvement_in_region')

    def test_cached_lower_threshold_can_prove_optimality_for_a_higher_request(self):
        p,_,ref=self.fixture()
        with GeneralizedFailureCache(p) as g:
            first=g.solve(ref,[],target_score=1,seconds=3)
            self.assertEqual(first['certificate_target'],1)
            self.assertTrue(first['optimal'])
            higher=g.solve(ref,[],target_score=4,seconds=3)
            self.assertTrue(higher['cache_hit']);self.assertTrue(higher['optimal'])
            self.assertEqual(higher['target_score'],4)
            self.assertEqual(higher['certificate_target'],1)
            self.assertEqual(higher['termination'],'no_improvement_in_region')

    def test_empty_core_retains_global_target_scope(self):
        p=tiny_problem(True);legal=sorted(boards(p));ref=legal[0][1]
        with GeneralizedFailureCache(p) as g:
            out=g.solve(ref,list(range(4)),target_score=4,seconds=3)
            self.assertTrue(out['proven_unreachable'])
            self.assertEqual(out['core_placements'],{})
            hit=g.solve(ref,[],target_score=4,seconds=3)
            self.assertTrue(hit['cache_hit'])
            relaxed=g.solve(ref,list(range(4)),target_score=3,seconds=3)
            self.assertFalse(relaxed['cache_hit']);self.assertEqual(relaxed['score'],3)
            self.assert_proof(legal,g.certificates()[0])

    def test_cannot_reuse_at_weaker_target_or_released_core(self):
        p,legal,ref=self.fixture()
        with GeneralizedFailureCache(p) as g:
            first=g.solve(ref,[],target_score=4,seconds=3)
            lower=g.solve(ref,[],target_score=2,seconds=3)
            self.assertFalse(lower['cache_hit'])
            released=g.solve(ref,list(range(4)),target_score=4,seconds=3)
            self.assertFalse(released['cache_hit']);self.assertEqual(released['score'],4)
        with GeneralizedFailureCache(p) as g:
            g.solve(ref,[],target_score=2,seconds=3)
            stronger=g.solve(ref,[],target_score=4,seconds=3)
            self.assertTrue(stronger['cache_hit'])

    def test_optional_core_deletions_each_preserve_proved_threshold(self):
        p,legal,ref=self.fixture()
        with GeneralizedFailureCache(p) as g:
            out=g.solve(ref,[],target_score=4,seconds=3,minimize_checks=8)
            self.assertTrue(out['proven_unreachable'])
            self.assertGreater(out['minimization_queries'],0)
            self.assertFalse(out['core_is_minimum'])
            self.assert_proof(legal,g.certificates()[0])
            self.assertEqual(out['native_queries'],1+out['minimization_queries'])

    def test_unknown_stop_model_limit_never_create_a_certificate(self):
        p,legal,ref=self.fixture()
        with GeneralizedFailureCache(p) as g:
            with patch('pysat.solvers.Glucose4',side_effect=AssertionError('No native allocation')):
                self.assertEqual(g.solve(ref,[],seconds=0)['termination'],'timeout')
                self.assertEqual(g.solve(ref,[],seconds=3,should_stop=lambda:True)['termination'],'stopped')
            self.assertEqual(g.cache_size,0)
        with GeneralizedFailureCache(p,max_variables=1) as g:
            out=g.solve(ref,[],target_score=4,seconds=3)
            self.assertEqual(out['termination'],'model_limit');self.assertEqual(g.cache_size,0)
        with GeneralizedFailureCache(p) as g:
            with patch('pysat.solvers.Glucose4.solve_limited',return_value=None),patch('pysat.solvers.Glucose4.get_core',side_effect=AssertionError('No stale core')):
                out=g.solve(ref,[],target_score=4,seconds=3)
            self.assertEqual(out['termination'],'timeout');self.assertEqual(g.cache_size,0)
            self.assertTrue(g.solve(ref,[],target_score=4,seconds=3)['proven_unreachable'])

    def test_cached_proof_does_not_override_stop_or_zero_budget(self):
        p,_,ref=self.fixture()
        with GeneralizedFailureCache(p) as g:
            g.solve(ref,[],target_score=4,seconds=3)
            self.assertEqual(g.solve(ref,[],target_score=4,seconds=0)['termination'],'timeout')
            self.assertEqual(g.solve(ref,[],target_score=4,seconds=3,should_stop=lambda:True)['termination'],'stopped')
            self.assertEqual(g.cache_hits,0)

    def test_problem_snapshot_and_detached_proof_descriptions(self):
        p,legal,ref=self.fixture()
        with GeneralizedFailureCache(p) as g:
            g.solve(ref,[],target_score=4,seconds=3)
            description=g.certificates()[0];description['placements'].clear();description['target_score']=0
            p.fixed[0]=ref[0]
            self.assertTrue(g.certificates()[0]['placements'])
            self.assertFalse(g.problem.fixed)
            with self.assertRaises(TypeError):g.problem.fixed[0]=ref[0]
            self.assertTrue(g.solve(ref,[],target_score=4,seconds=3)['cache_hit'])

    def test_clues_domains_frame_and_physical_identity_preserved(self):
        p,answer=generated_problem(3,4,1709,clue_count=2)
        from repair_diversify import Diversifier
        ref=Diversifier(p).perturb(answer,seed=51,moves=3)['board']
        free=[c for c in range(9) if c not in p.fixed]
        with GeneralizedFailureCache(p) as g:
            out=g.solve(ref,free,target_score=12,seconds=3)
            self.assertEqual(out['score'],12);self.assertTrue(check_solution(p,out['board']))
            with self.assertRaises(ValueError):g.solve(ref,[next(iter(p.fixed))],seconds=1)
            illegal=list(ref);illegal[0]=illegal[1]
            with self.assertRaises(ValueError):g.solve(illegal,free,seconds=1)

    def test_bad_core_and_reentry_do_not_poison_cache(self):
        p,_,ref=self.fixture()
        g=GeneralizedFailureCache(p)
        rejected=[]
        def callback():
            with self.assertRaises(RuntimeError):g.solve(ref,[],seconds=1)
            rejected.append(True);return False
        out=g.solve(ref,[],target_score=4,seconds=3,should_stop=callback)
        self.assertTrue(rejected);self.assertTrue(out['proven_unreachable']);g.close()
        with self.assertRaises(RuntimeError):g.solve(ref,[],seconds=1)
        bad=dict(out,outcome='infeasible',board=None,core_placements={0:999},model_limit_reached=False)
        with GeneralizedFailureCache(p) as g,patch.object(FixedTargetScoreSession,'solve_frozen',return_value=bad):
            with self.assertRaises(RuntimeError):g.solve(ref,[],target_score=4,seconds=3)
            self.assertEqual(g.cache_size,0)

    def test_capacity_validation_and_single_native_target_session(self):
        p,legal,ref=self.fixture()
        for kwargs in ({'capacity':0},{'capacity':True},{'generalize':1},{'conflict_limit':0}):
            with self.assertRaises(ValueError):GeneralizedFailureCache(p,**kwargs)
        with GeneralizedFailureCache(p,capacity=1,generalize=False) as g:
            g.solve(ref,[],target_score=4,seconds=3,seed=1);old=g._session
            g.solve(ref,[],target_score=2,seconds=3,seed=1)
            self.assertTrue(old.closed);self.assertIsNone(old.native);self.assertEqual(g.cache_size,1)
            for kwargs in ({'target_score':True},{'seed':True},{'minimize_checks':-1},{'seconds':float('nan')}):
                with self.assertRaises(ValueError):g.solve(ref,[],**kwargs)

if __name__=='__main__':unittest.main()
