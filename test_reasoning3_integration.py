"""App routing, preservation and reporting for the four retained improvements."""
from pathlib import Path
import json,tempfile,unittest
from unittest.mock import patch
import dashboard_server,exact_worker,launcher
from exact_model import generated_problem,check_solution


class ReasoningIntegrationTests(unittest.TestCase):
    def test_worker_selects_tested_configuration_and_preserves_reference(self):
        routes={'learning-dfs':{'matching':True,'max_work':100_000_000},'regional-master':{'minimize_interfaces':True,'max_work':10_000_000},
                'refined-dd':{'ordering':'domain','max_work':10_000_000},'impact-dfs':{'ordering':'impact','discrepancy_limit':None}}
        for engine,expected in routes.items():
            with tempfile.TemporaryDirectory() as tmp:
                observed={}
                def fake(**kwargs):
                    observed.update(kwargs)
                    return dict(outcome='timeout',termination='work_limit',board=None,branches=7,conflicts=1,
                                reasoning_counters_session={'work':41,'impact_operations':19})
                with patch.object(exact_worker,'peak_memory_mb',return_value=32.):
                    code=exact_worker.run_task(tmp,engine=engine,seconds=1,solver_factory=lambda _:fake)
                self.assertEqual(code,0)
                for key,value in expected.items():self.assertEqual(observed[key],value)
                self.assertNotIn('checkpoint',observed)
                s=json.loads((Path(tmp)/'runtime/status.json').read_text())
                self.assertEqual(s['best_score'],466);self.assertFalse(s['external_network_enabled'])
                self.assertEqual(s['exact_phase'],'work_limit');self.assertEqual(s['branches'],7)
                self.assertEqual(s['reasoning_counters_session']['work'],41)
                self.assertIn('fresh search',s['resume_capability'])

    def test_impact_engine_reaches_real_solver(self):
        p,_=generated_problem(3,3,19,clue_count=1)
        r=exact_worker._backend('impact-dfs')(p,seconds=3,ordering='impact',discrepancy_limit=None)
        self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(p,r['board']))
        self.assertEqual(r['reasoning_counters_session']['work'],r['work'])
        self.assertFalse(r['checkpoint_supported'])

    def test_controls_packaging_and_single_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=launcher.parser().parse_args(['run','--method','exact','--exact-engine','impact-dfs','--state-dir',tmp])
            settings=dashboard_server.validate_settings({'method':'exact','exact_engine':'impact-dfs'})
            for command in (launcher.run_arguments(args),dashboard_server.worker_command(Path(tmp),8765,settings)):
                self.assertIn('impact-dfs',command)
            with self.assertRaises(ValueError):dashboard_server.validate_settings({'method':'exact','exact_engine':'impact-dfs','workers':2})
        root=Path(__file__).resolve().parent
        self.assertIn('value="impact-dfs"',(root/'Dashboard.html').read_text())
        self.assertIn('This experimental engine starts fresh each time',(root/'Dashboard.html').read_text())
        for name in ('pyproject.toml','EternityIISolver.spec'):
            self.assertIn('"impact_dfs"',(root/name).read_text())
        self.assertEqual(launcher.parser().parse_args(['run']).exact_engine,'dfs')

if __name__=='__main__':unittest.main()
