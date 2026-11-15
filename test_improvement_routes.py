from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import dashboard_server
import exact_worker
import launcher
from exact_model import generated_problem, check_solution
from validator import load_bundle

ROOT=Path(__file__).resolve().parent


class ImprovementRouting(unittest.TestCase):
    def test_opt_in_sat_and_strengthened_regional_settings_are_forwarded(self):
        routes={'learning-rules':{'max_work':100_000_000},'sat-channels':dict(piece_channels=True,small_pairwise=True),
                'regional-master':dict(minimize_interfaces=True,allocation_cores=True,
                                       color_conservation=True,color_coupling='edge_colors',hall_capacity=True,cache_witnesses=True)}
        for name,expected in routes.items():
            with TemporaryDirectory() as d:
                args=launcher.parser().parse_args(['run','--method','exact','--exact-engine',name,'--state-dir',d])
                self.assertIn(name,launcher.run_arguments(args))
                settings=dashboard_server.validate_settings({'method':'exact','exact_engine':name})
                self.assertIn(name,dashboard_server.worker_command(Path(d),8765,settings))
                observed={}
                def solve(**kwargs):
                    observed.update(kwargs)
                    return dict(outcome='timeout',board=None,branches=1,conflicts=0)
                with patch.object(exact_worker,'peak_memory_mb',return_value=32.0):
                    self.assertEqual(exact_worker.run_task(d,engine=name,seconds=1,solver_factory=lambda _:solve),0)
                for key,val in expected.items():
                    self.assertEqual(observed[key],val)
                status=json.loads((Path(d)/'runtime/status.json').read_text())
                self.assertFalse(status['external_network_enabled'])
                self.assertEqual(status['best_score'],466)
                task,_=generated_problem(2,2,63)
                result=exact_worker._backend(name)(task,seconds=2,**expected)
                self.assertTrue(check_solution(task,result['board']))

    def test_repair_profiles_survive_dashboard_and_cli_forwarding(self):
        for name in ('guided-lift','price-bounds'):
            with self.subTest(name=name), TemporaryDirectory() as directory:
                settings=dashboard_server.validate_settings({'method':'exact','exact_engine':'matching','repair_quality':name})
                self.assertIn(name,dashboard_server.worker_command(Path(directory),8765,settings))
                args=launcher.parser().parse_args(['run','--method','exact','--exact-engine','matching','--repair-quality',name])
                self.assertIn(name,launcher.run_arguments(args))
                self.assertIn(f'value="{name}"',(ROOT/'Dashboard.html').read_text(encoding='utf-8'))

    def test_radius_command_uses_explicit_parent_and_honors_cutoff(self):
        output=StringIO()
        with redirect_stdout(output):
            self.assertEqual(launcher.main(['repair-radius',str(ROOT/'data/record466.json'),
                                            '--radius','2','--max-work','0']),0)
        result=json.loads(output.getvalue())
        self.assertEqual(result['score'],466)
        self.assertEqual(result['termination'],'work_limit')

    def test_patch_command_rechecks_candidate_and_reference(self):
        output=StringIO()
        reference=str(ROOT/'data/record466.json')
        with redirect_stdout(output):
            self.assertEqual(launcher.main(['repair-patches',reference,reference]),0)
        result=json.loads(output.getvalue())
        self.assertEqual(result['score'],466)

    def test_distance_command_checks_and_returns_certificate(self):
        output=StringIO()
        with redirect_stdout(output):
            self.assertEqual(launcher.main(['perfect-distance',str(ROOT/'data/record466.json')]),0)
        result=json.loads(output.getvalue())
        self.assertEqual(result['target_score'],480)
        self.assertEqual(result['parent_score'],466)
        self.assertGreaterEqual(result['lower_bound'],1)
        self.assertTrue(result['regional_certificates'])
