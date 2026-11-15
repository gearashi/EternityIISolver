import unittest
from itertools import product
from unittest.mock import patch
from exact_model import generated_problem
from exact_optimize import evaluate_board
from repair_exact_regions import solve_region,RegionFailureCache
from repair_reasoning import ReasoningRegionSession


class CertifiedRegionIntegration(unittest.TestCase):
    def test_every_tiny_target_preserves_structural_best_and_frozen_pool(self):
        for seed in range(7):
            p,witness=generated_problem(2,3,seed,clue_count=1)
            free=tuple(c for c in range(4) if c not in p.fixed)
            boards=[b for b in product(*p.domains) if evaluate_board(p,b)['valid']]
            parent=min(boards,key=lambda b:evaluate_board(p,b)['score'])
            maximum=evaluate_board(p,witness)['score']
            for target in (maximum,maximum+1):
                r=solve_region(p,parent,free,target_score=target,certified_prices=True,algorithm='dfs')
                self.assertEqual(r['target_reached'],target<=maximum)
                self.assertTrue(evaluate_board(p,r['board'])['valid'])
                if target>maximum:self.assertTrue(r['proven_unreachable'])
            session=ReasoningRegionSession(p,'price-bounds')
            try:
                with patch('repair_reasoning.solve_region', wraps=solve_region) as scoped:
                    r=session.solve(parent,free,seconds=3)
                    self.assertTrue(scoped.call_args.kwargs['assignment_bound'])
                    self.assertTrue(scoped.call_args.kwargs['clique_bound'])
                    self.assertTrue(scoped.call_args.kwargs['certified_prices'])
            finally:session.close()
            self.assertTrue(evaluate_board(p,r['board'])['valid'])
