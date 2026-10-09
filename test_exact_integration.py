"""Public controls keep CPU exact search distinct from GPU board repair."""
import unittest
from pathlib import Path
from unittest.mock import patch
from dashboard_server import validate_settings,worker_command
from launcher import parser,run_arguments
from exact_model import ExactProblem,generated_problem,from_bundle,check_solution
from validator import load_bundle

class ExactIntegrationTests(unittest.TestCase):
    def test_existing_gpu_settings_remain_gpu(self):
        settings=validate_settings({'replicas':128,'backend':'cuda'})
        self.assertEqual(settings['method'],'gpu')
        self.assertIn('run',worker_command(Path('/state'),8765,settings))
        self.assertNotIn('exact',worker_command(Path('/state'),8765,settings))
    def test_exact_dispatch_has_no_gpu_replicas_or_backend(self):
        settings=validate_settings({'method':'exact','exact_engine':'cp-sat','workers':2,'seconds':1})
        command=worker_command(Path('/state'),8765,settings)
        self.assertIn('exact',command);self.assertIn('--preserve-stop',command)
        self.assertNotIn('--replicas',command);self.assertNotIn('--backend',command)
        self.assertIn('cp-sat',command);self.assertIn('2',command)
    def test_hybrid_forwards_gpu_sampling_settings(self):
        settings=validate_settings({'method':'exact','exact_engine':'hybrid','workers':1,'backend':'opencl','replicas':128})
        command=worker_command(Path('/state'),8765,settings)
        self.assertIn('hybrid',command);self.assertIn('--backend',command);self.assertIn('opencl',command)
        self.assertIn('--replicas',command);self.assertIn('128',command)
    def test_invalid_exact_settings_rejected(self):
        for values in ({'method':'remote'},{'exact_engine':'bogus'},{'workers':True},{'workers':0},{'workers':5},{'method':'exact','exact_engine':'dfs','workers':2},{'method':'exact','exact_engine':'sat','workers':2}):
            with self.subTest(values=values),self.assertRaises(ValueError):validate_settings(values)
    def test_cli_exact_route_parameters(self):
        args=parser().parse_args(['run','--method','exact','--exact-engine','sat','--workers','1','--seconds','0','--preserve-stop'])
        command=run_arguments(args)
        self.assertIn('--engine',command);self.assertIn('sat',command);self.assertIn('--preserve-stop',command)
        self.assertNotIn('--backend',command);self.assertNotIn('--replicas',command)
    def test_full_puzzle_does_not_freeze_reference_board(self):
        b=load_bundle();p=from_bundle(b)
        self.assertEqual(p.fixed,b.fixed_clues)
        self.assertIsNone(p.edges)
        self.assertFalse(check_solution(p,list(b.record_board)))
        self.assertGreater(sum(map(len,p.domains)),100000)
    def test_regions_have_explicitly_limited_scope(self):
        b=load_bundle();free={76,77,92,93};p=from_bundle(b,board=list(b.record_board),free_cells=free)
        self.assertTrue(all(a in free or c in free for a,c,_,_ in p.edges))
        self.assertEqual(p.fixed[0],b.record_board[0]);self.assertNotIn(76,p.fixed)
    def test_common_input_type_checks_are_strict(self):
        p,_=generated_problem(2,2,7)
        with self.assertRaises(ValueError):ExactProblem(p.size,p.faces,p.fixed,p.domains,((False,True,True,3),))
        for size in (True,'16',0,17):
            with self.assertRaises(ValueError):generated_problem(size,3,7)
if __name__=='__main__':unittest.main()
