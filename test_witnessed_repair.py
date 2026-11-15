from copy import deepcopy
from dataclasses import replace
import unittest

from exact_model import generated_problem
from exact_optimize import evaluate_board
from repair_witnesses import WitnessRegionSession
from repair_quality import QualityRepairController
from smart_repair import solve


class WitnessIntegrationTests(unittest.TestCase):
    def test_positive_replay_after_restart_and_reject_stale_pool(self):
        p,ans=generated_problem(5,13,17)
        parent=ans.copy()
        for c in (6,7,8):
            parent[c]=parent[c]//4*4+(parent[c]+1)%4
        original=WitnessRegionSession(p)
        r=original.solve(parent,(6,7,8),seconds=2)
        self.assertTrue(r['improved'])
        saved=original.export_state()
        restored=WitnessRegionSession(p)
        restored.restore_state(saved)
        hit=restored.solve(parent,(6,7,8),seconds=2)
        self.assertEqual(hit['termination'],'positive_witness')
        self.assertTrue(evaluate_board(p,hit['board'])['valid'])
        bad=deepcopy(saved)
        bad['witnesses'][0][1][0]=bad['witnesses'][0][1][1]
        with self.assertRaises(ValueError):
            restored.restore_state(bad)
        self.assertEqual(restored.stats()['positive_patch_hits'],1)
        self.assertNotEqual(restored.solve(parent,(6,7,8),seconds=0)['termination'],'positive_witness')

    def test_lifted_controller_checks_selected_domains_and_restores_counters(self):
        from test_repair_rotations import families
        p,parents=families(42,(6,8,16,18))
        controller=QualityRepairController(p,lift_rotations=True)
        r=controller.recombine(parents,seed=11,seconds=3)
        self.assertGreater(r['quality_stats']['quality_lift_orientation_tests'],0)
        checked=evaluate_board(p,r['board'])
        self.assertTrue(checked['valid'])
        self.assertEqual(checked['score'],r['score'])
        again=QualityRepairController(p,lift_rotations=True)
        again.restore_state(controller.export_state())
        self.assertEqual(again.stats(),controller.stats())

    def test_new_profile_roundtrip_and_legacy_separation(self):
        p,board=generated_problem(5,8,123)
        # Fix every legal assignment to the same imperfect physical board.
        board[6]=board[6]//4*4+(board[6]+1)%4
        p=replace(p,domains=tuple((s,) for s in board))
        opts=dict(seconds=2,seed=13,repair_quality='guided-lift',on_checkpoint=lambda _:None,
                  region_repair_enabled=False)
        one=solve(p,board,max_batches=1,**opts)
        self.assertIn('witnesses',one['checkpoint']['state']['policy_state'])
        two=solve(p,board,max_batches=1,checkpoint=one['checkpoint'],**opts)
        self.assertEqual(two['score'] if 'score' in two else two['best_score'],evaluate_board(p,board)['score'])
        with self.assertRaises(ValueError):
            solve(p,board,max_batches=1,checkpoint=one['checkpoint'],**dict(opts,repair_quality='guided'))
