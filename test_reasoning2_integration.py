"""Optional second-round reasoning integration, independent finite oracles."""
from pathlib import Path
import json,tempfile,unittest,itertools
from unittest.mock import patch
import exact_dfs,exact_rules,exact_worker,launcher,dashboard_server,smart_repair
from exact_model import generated_problem,check_solution
from exact_optimize import evaluate_board
from repair_exact_regions import solve_region
from repair_reasoning import ReasoningRegionSession
from test_exact_sat import tiny_problem
from test_repair_bounds import completions
from test_repair_exact_regions import scramble
from test_region_failure_cache import legal_boards

class Reasoning2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Algorithm checks exclude a variable first DLL import; packaged startup
        # separately verifies that a short total budget ends inconclusively.
        from ortools.sat.python import cp_model

    def test_disabled_relational_options_keep_all_previous_fingerprints(self):
        for prior in ({},{'joint_colors':True}):
            self.assertEqual(exact_rules.normalize_options(prior),exact_rules.normalize_options(dict(prior,relational=False,relational_work=0,relational_tuples=1)))
        p,_=generated_problem(2,3,6)
        previous=exact_dfs.solve(p,seconds=0,options={'extra_rules':{}})
        continued=exact_dfs.solve(p,seconds=2,options={'extra_rules':{'relational':False}},checkpoint=previous['checkpoint'])
        self.assertEqual(continued['outcome'],'solved')
        with self.assertRaisesRegex(ValueError,'options hash mismatch'):
            exact_dfs.solve(p,seconds=1,options={'extra_rules':{'relational':True}},checkpoint=previous['checkpoint'])

    def test_real_relational_rule_preserves_known_solutions_and_reports_work(self):
        for size in (2,3,4):
            p,_=generated_problem(size,4,101,clue_count=1)
            options={'extra_rules':{'relational':True,'relational_work':3000}}
            r=exact_dfs.solve(p,seconds=3,options=options)
            self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(p,r['board']))
            self.assertIn('relational_work',r['rule_counters_session'])

    def test_unknown_relational_filter_cannot_remove_solutions(self):
        p,_=generated_problem(3,3,211)
        options={'extra_rules':{'relational':True,'relational_work':0}}
        r=exact_dfs.solve(p,seconds=3,options=options)
        self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(p,r['board']))
        self.assertGreater(r['rule_counters_session']['relational_skips'],0)

    def test_cost_bound_all_tiny_region_targets_match_complete_oracle(self):
        checked=0
        for p in (tiny_problem(False),tiny_problem(True)):
            boards=legal_boards(p)
            for _,board in (min(boards),max(boards)):
                for cells in ([0,1],[0,3],list(range(4))):
                    allowed=completions(p,board,cells,{})
                    maximum=max(score for score,_ in allowed)
                    for target in range(1,6):
                        r=solve_region(p,board,cells,target_score=target,algorithm='dfs',max_work=100000,cost_partition_bound=True)
                        self.assertEqual(r['target_reached'],maximum>=target,(maximum,target,r))
                        if maximum<target:self.assertTrue(r['proven_unreachable'])
                        self.assertTrue(evaluate_board(p,r['board'])['valid']);checked+=1
        self.assertEqual(checked,60)

    def test_partition_bound_never_double_counts_uncovered_frozen_edges(self):
        for seed in (101,211):
            p,known=generated_problem(4,3,seed)
            board=scramble(p,known)
            for cells in ([0,3],[5,6],[0,3,12]):
                pool={board[c]//4 for c in cells};scores=[]
                options=[tuple(s for s in p.domains[c] if s//4 in pool) for c in cells]
                for states in itertools.product(*options):
                    candidate=list(board)
                    for cell,state in zip(cells,states):candidate[cell]=state
                    checked=evaluate_board(p,candidate)
                    if checked['valid']:scores.append(checked['score'])
                optimum=max(scores)
                for target in (optimum,optimum+1):
                    r=solve_region(p,board,cells,target_score=target,algorithm='dfs',max_work=100000,cost_partition_bound=True)
                    self.assertEqual(r['target_reached'],target<=optimum,(seed,cells,target,r))
                    if target>optimum:self.assertTrue(r['proven_unreachable'])

    def test_matching_controller_accepts_policy_and_resumes_its_population(self):
        p=tiny_problem(True);_,board=max(legal_boards(p))
        settings=smart_repair.checkpoint_settings(repair_quality='partition-bounds')
        self.assertEqual(settings['repair_quality'],'partition-bounds')
        options=dict(seconds=2,seed=91,repair_quality='partition-bounds',region_repair_enabled=False,
                     on_checkpoint=lambda payload:None)
        first=smart_repair.solve(p,board,max_batches=1,**options)
        continued=smart_repair.solve(p,board,max_batches=1,checkpoint=first['checkpoint'],**options)
        self.assertEqual(continued['repair_quality'],'partition-bounds')
        self.assertTrue(continued['resumed']);self.assertTrue(evaluate_board(p,continued['board'])['valid'])

    def test_new_region_policy_is_bounded_and_leaves_reference_on_cancel(self):
        p=tiny_problem(True);_,board=max(legal_boards(p))
        session=ReasoningRegionSession(p,'partition-bounds')
        r=session.solve(board,[0,1],seconds=2)
        self.assertTrue(r['proven_unreachable']);self.assertTrue(r['cost_partition_bound'])
        self.assertGreater(session.stats()['region_reasoning_bound_work'],0)
        session.close()
        session=ReasoningRegionSession(p,'partition-bounds')
        r=session.solve(board,[0,1],seconds=0)
        self.assertFalse(r['proven_unreachable']);self.assertEqual(r['board'],board)
        session.close()

    def test_new_backend_routes_are_real_and_enforce_single_worker(self):
        p,_=generated_problem(2,3,17)
        for engine in ('learning-dfs','regional-master','refined-dd'):
            run=exact_worker._backend(engine)
            r=run(p,seconds=3,seed=17)
            self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(p,r['board']))
            self.assertIsInstance(r['reasoning_counters_session'],dict)
            with self.assertRaises(ValueError):dashboard_server.validate_settings({'method':'exact','exact_engine':engine,'workers':2})

    def test_app_controls_forward_engine_pruning_and_repair_choices(self):
        with tempfile.TemporaryDirectory() as tmp:
            for engine in ('learning-dfs','regional-master','refined-dd','dfs','matching','gpu-matching'):
                args=launcher.parser().parse_args(['run','--method','exact','--exact-engine',engine,
                    '--pruning','relational','--repair-quality','partition-bounds','--state-dir',tmp])
                settings=dashboard_server.validate_settings({'method':'exact','exact_engine':engine,
                    'pruning':'relational','repair_quality':'partition-bounds'})
                for command in (launcher.run_arguments(args),dashboard_server.worker_command(Path(tmp),8765,settings)):
                    self.assertIn(engine,command);self.assertIn('relational',command)
                    self.assertEqual('partition-bounds' in command,engine in ('matching','gpu-matching'))

    def test_worker_reports_work_limits_as_inconclusive_and_keeps_saved_board(self):
        for engine in ('learning-dfs','regional-master','refined-dd'):
            with tempfile.TemporaryDirectory() as tmp:
                def fake(**kwargs):return dict(outcome='timeout',termination='work_limit',board=None,
                    branches=3,conflicts=1,reasoning_counters_session={'work':27,'master_branches':11})
                with patch.object(exact_worker,'peak_memory_mb',return_value=32.):
                    code=exact_worker.run_task(tmp,engine=engine,seconds=1,solver_factory=lambda _:fake)
                self.assertEqual(code,0)
                s=json.loads((Path(tmp)/'runtime/status.json').read_text())
                self.assertEqual(s['state'],'finished_bounded_run');self.assertEqual(s['exact_phase'],'work_limit')
                self.assertEqual(s['best_score'],466);self.assertFalse(s['external_network_enabled'])
                self.assertEqual(s['reasoning_counters_session']['work'],27)
                self.assertIn('fresh search',s['resume_capability'])

if __name__=='__main__':unittest.main()
