"""Optional regional-policy integration and bounded adapter lifecycle."""
from dataclasses import replace
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dashboard_server
import launcher
import repair_reasoning
import smart_repair
from exact_model import generated_problem
from exact_optimize import evaluate_board
from test_exact_sat import tiny_problem
from test_region_failure_cache import legal_boards


class ReasoningAdapterTests(unittest.TestCase):
    def setUp(self):
        self.problem=tiny_problem(True)
        self.score,self.board=max(legal_boards(self.problem))

    def test_bounded_proof_cache_and_frozen_complement(self):
        session=repair_reasoning.ReasoningRegionSession(self.problem,'bounded')
        result=session.solve(self.board,[0,1],seconds=2)
        self.assertTrue(result['proven_unreachable'])
        self.assertEqual(result['board'][2:],self.board[2:])
        again=session.solve(self.board,[1,0],seconds=2)
        self.assertTrue(again['cache_hit'])
        self.assertEqual(session.cache_hits,1)
        self.assertGreaterEqual(session.stats()['region_reasoning_proved_targets'],2)
        self.assertTrue(evaluate_board(self.problem,result['board'])['valid'])
        session.close()
        with self.assertRaises(RuntimeError):session.solve(self.board,[0])

    def test_zero_budget_never_creates_a_failure_proof(self):
        session=repair_reasoning.ReasoningRegionSession(self.problem,'bounded')
        result=session.solve(self.board,[0,1],seconds=0)
        self.assertFalse(result['proven_unreachable'])
        self.assertEqual(session.cache_hits,0)
        self.assertEqual(session.stats()['region_reasoning_unknown_queries'],1)
        self.assertEqual(result['board'],self.board)

    def test_snapshot_and_bounded_invalid_input(self):
        problem=replace(self.problem,faces=[list(f) for f in self.problem.faces],
                        domains=[list(d) for d in self.problem.domains])
        session=repair_reasoning.ReasoningRegionSession(problem,'bounded')
        problem.faces[0][0]=255;problem.domains[0].clear();problem.fixed[0]=1
        self.assertNotEqual(session.problem.faces[0][0],255)
        self.assertTrue(session.problem.domains[0]);self.assertFalse(session.problem.fixed)
        for cells in (None,[0,0],[True],[-1],[4],itertools.repeat(0)):
            with self.assertRaises(ValueError):session.solve(self.board,cells,seconds=0)
        for kwargs in ({'seconds':True},{'seconds':float('nan')},{'seed':True},{'should_stop':5}):
            with self.assertRaises(ValueError):session.solve(self.board,[0],**kwargs)

    def test_large_regions_keep_existing_cp_path(self):
        p,board=generated_problem(6,3,2)
        session=repair_reasoning.ReasoningRegionSession(p,'bounded')
        with patch.object(session.fallback,'solve',return_value={'board':board,'termination':'timeout','branches':3,'conflicts':2}) as fallback:
            session.solve(board,range(25),seconds=0)
        self.assertEqual(fallback.call_count,1)
        self.assertEqual(session.stats()['region_reasoning_native_branches'],3)
        self.assertEqual(session.stats()['region_reasoning_native_queries'],0)

    def test_learning_keeps_seed_and_closes_native_session(self):
        with patch.object(repair_reasoning,'GeneralizedFailureCache') as factory:
            native=factory.return_value
            native.solve.return_value={'board':self.board,'termination':'timeout',
                'native_queries':1,'branches':2,'conflicts':3,'propagations':5,
                'encoding_clauses':7,'model_builds':1,'build_seconds':.25,'lookup_literal_checks':4}
            session=repair_reasoning.ReasoningRegionSession(self.problem,'learned')
            session.solve(self.board,[0],seed=91)
            session.solve(self.board,[1],seed=17)
            self.assertEqual([c.kwargs['seed'] for c in native.solve.call_args_list],[91,91])
            self.assertEqual(session.stats()['region_reasoning_native_propagations'],10)
            self.assertEqual(session.stats()['region_reasoning_encoding_clauses'],14)
            self.assertEqual(session.stats()['region_reasoning_lookup_literal_checks'],8)
            session.close();native.close.assert_called_once()

    def test_dashboard_and_launcher_forward_policies_only_to_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            for policy in ('bounded','learned'):
                for engine in ('matching','gpu-matching','dfs'):
                    args=launcher.parser().parse_args(['run','--method','exact','--exact-engine',engine,
                        '--repair-quality',policy,'--state-dir',directory])
                    settings=dashboard_server.validate_settings({'method':'exact','exact_engine':engine,'repair_quality':policy})
                    for command in (launcher.run_arguments(args),dashboard_server.worker_command(Path(directory),8765,settings)):
                        self.assertEqual('--repair-quality' in command,engine!='dfs')
                        if engine!='dfs':self.assertEqual(command[command.index('--repair-quality')+1],policy)

    def test_matching_session_closed_on_completion_and_exception(self):
        instances=[]
        class Stub:
            cache_hits=0
            def __init__(self,p,mode):self.closed=False;self.mode=mode;instances.append(self)
            def stats(self):return {}
            def close(self):self.closed=True
            def solve(self,board,cells,**kwargs):
                if fail[0]:raise RuntimeError('region fixture failure')
                return {'board':list(board)}
        def unchanged(problem,board,cells,**kwargs):return {'board':list(board),'optimal':True}
        # A virtual monotonic clock makes the portfolio allowance deterministic.
        class Clock:
            def __init__(self):self.value=0
            def __call__(self):self.value+=.0001;return self.value+(1 if self.value>.0001 else 0)
        for mode in ('bounded','learned','partition-bounds'):
            for should_fail in (False,True):
                fail=[should_fail];clock=Clock()
                with patch.object(repair_reasoning,'ReasoningRegionSession',Stub), \
                     patch.object(smart_repair,'assignment_repair',side_effect=unchanged), \
                     patch('repair_regions.select_region',return_value=[0,1]), \
                     patch.object(smart_repair.time,'monotonic',side_effect=clock):
                    kwargs=dict(seconds=20,max_batches=5,region_every=1,region_seconds=1,
                                diversify=False,repair_quality=mode)
                    if should_fail:
                        with self.assertRaisesRegex(RuntimeError,'region fixture failure'):
                            smart_repair.solve(self.problem,self.board,**kwargs)
                    else:smart_repair.solve(self.problem,self.board,**kwargs)
                self.assertTrue(instances);self.assertTrue(instances[-1].closed)
                self.assertEqual(instances[-1].mode,mode)

if __name__=='__main__':unittest.main()
