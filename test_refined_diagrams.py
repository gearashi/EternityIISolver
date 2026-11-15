"""Finite independent oracles for the relaxed/refined diagram engine."""
from dataclasses import replace
import itertools,random,unittest
from unittest.mock import patch
from exact_model import check_solution,generated_problem
from test_exact_sat import tiny_problem
from refined_diagrams import solve


def oracle(p):
    return next((list(b) for b in itertools.product(*p.domains) if check_solution(p,b)),None)

class RefinedDiagramTests(unittest.TestCase):
    def test_exhaustive_small_domain_oracle(self):
        for seed in range(24):
            p,answer=generated_problem(2,2+(seed%2),seed,clue_count=seed%2)
            rng=random.Random(seed+100)
            p=replace(p,domains=tuple(tuple(s for s in d if rng.random()<.7) for d in p.domains))
            expected=oracle(p)
            for mode in ('exact','refine'):
                r=solve(p,seconds=3,seed=seed,abstraction=mode)
                self.assertEqual(r['outcome'],'solved' if expected is not None else 'infeasible',(seed,mode,r))
                if r['board'] is not None:self.assertTrue(check_solution(p,r['board']))

    def test_counterexample_refinement_is_real(self):
        p,_=generated_problem(2,1,8)
        # Three physical pieces cannot fill all four cells, even though local
        # colors can all match; relaxation must learn missing identity checks.
        p=replace(p,domains=tuple(tuple(s for s in d if s//4<3) for d in p.domains))
        r=solve(p,seconds=3,seed=4,refine_batch=1)
        self.assertEqual(r['outcome'],'infeasible');self.assertIsNone(oracle(p))
        self.assertGreater(r['reasoning_counters_session']['refinements'],0)
        self.assertGreater(r['reasoning_counters_session']['diagrams'],1)

    def test_clues_frame_identity_and_restricted_edges(self):
        for size in (2,3):
            p,answer=generated_problem(size,4,451,clue_count=1)
            for edges in (None,()):
                scoped=replace(p,edges=edges)
                r=solve(scoped,seconds=5,max_work=1000000)
                self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(scoped,r['board']))
                self.assertTrue(all(r['board'][c]==s for c,s in p.fixed.items()))

    def test_caps_are_unknown_not_proofs(self):
        p,_=generated_problem(2,1,8)
        for options in ({'seconds':0},{'max_work':0},{'max_nodes':1}):
            r=solve(p,**options);self.assertEqual(r['outcome'],'timeout')
            self.assertFalse(r['coverage_verified']);self.assertIsNone(r['board'])
        with patch('refined_diagrams.initial_domains',side_effect=AssertionError('Unbudgeted preprocessing')):
            self.assertEqual(solve(p,max_work=0)['termination'],'work_limit')
        r=solve(p,should_stop=lambda:True);self.assertEqual(r['outcome'],'stopped')
        p=replace(p,domains=tuple(tuple(s for s in d if s//4<3) for d in p.domains))
        r=solve(p,max_refinements=0)
        self.assertEqual(r['termination'],'refinement_limit');self.assertIsNone(r['board'])

    def test_invalid_limits_and_snapshot(self):
        p,answer=generated_problem(2,3,4)
        for options in ({'seconds':True},{'seconds':float('nan')},{'seed':True},{'workers':2},
                        {'max_nodes':0},{'max_work':False},{'refine_batch':0},{'abstraction':'bad'},
                        {'should_stop':1},{'on_progress':1}):
            with self.assertRaises(ValueError):solve(p,**options)
        source=replace(p,faces=[list(f) for f in p.faces],domains=[list(d) for d in p.domains])
        def mutate():source.faces[0][0]=255;source.domains[0].clear();return False
        r=solve(source,should_stop=mutate)
        self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(p,r['board']))

    def test_counted_work_is_reported_without_hidden_macro_move(self):
        p,_=generated_problem(2,1,8);r=solve(p)
        c=r['reasoning_counters_session']
        self.assertEqual(r['branches'],c['candidate_placements'])
        self.assertEqual(r['work'],sum(c[k] for k in ('candidate_placements','edge_checks','state_operations','witness_checks','setup_work')))
        self.assertGreater(c['merged_paths'],0)

if __name__=='__main__':unittest.main()
