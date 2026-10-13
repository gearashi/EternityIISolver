"""Finite invariants and portfolio regression checks for plateau escapes."""
import unittest
from unittest.mock import patch
from dataclasses import replace
from exact_model import generated_problem, make_problem
from repair_diversify import Diversifier
from smart_repair import solve, score, structurally_valid
from test_smart_repair import scrambled


class DiversificationTests(unittest.TestCase):
    def test_legal_seeded_moves_preserve_clues_and_input(self):
        for seed in range(12):
            problem, board = generated_problem(4, 6, seed, clue_count=3)
            original = list(board)
            worker = Diversifier(problem)
            first = worker.perturb(board, seed=seed, moves=5)
            second = worker.perturb(board, seed=seed, moves=5)
            self.assertEqual(first, second)
            self.assertEqual(board, original)
            self.assertTrue(first['changed'])
            self.assertTrue(structurally_valid(problem, first['board']))
            self.assertTrue(all(first['board'][c] == s for c,s in problem.fixed.items()))
            self.assertEqual(sorted(s//4 for s in first['board']), sorted(s//4 for s in board))

    def test_restricted_domains_and_fully_fixed_boards(self):
        p, board = generated_problem(3, 5, 7)
        restricted = make_problem(p.size, p.faces, domains=[(s,) for s in board])
        for problem in (restricted, replace(p, fixed=dict(enumerate(board)))):
            result = Diversifier(problem).perturb(board, moves=8)
            self.assertFalse(result['changed'])
            self.assertEqual(result['board'], board)
            self.assertEqual(result['termination'], 'no_legal_move')

    def test_tabu_excludes_the_previous_escape(self):
        p, board = generated_problem(4, 6, 2)
        worker = Diversifier(p)
        first = worker.perturb(board, seed=42, moves=1)
        second = worker.perturb(board, seed=42, moves=1, forbidden_boards=[first['board']])
        self.assertTrue(second['changed'])
        self.assertNotEqual(second['board'], first['board'])
        self.assertTrue(structurally_valid(p, second['board']))

    def test_cancellation_retains_original_and_bad_inputs_fail(self):
        p, board = generated_problem(3, 5, 2)
        worker = Diversifier(p)
        self.assertEqual(worker.perturb(board, should_stop=lambda:True)['board'], board)
        for kwargs in ({'seed':True},{'seed':-1},{'moves':0},{'moves':33},{'moves':1.5},{'should_stop':7}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): worker.perturb(board, **kwargs)
        with self.assertRaises(ValueError): worker.perturb([0]*9)
        with self.assertRaises(ValueError): Diversifier(replace(p,edges=()))

    def fixture(self):
        p, answer = generated_problem(4, 7, 28, clue_count=1)
        reference = list(answer)
        cell = next(c for c in (5,6,9,10) if c not in p.fixed)
        reference[cell] = 4*(reference[cell]//4)+(reference[cell]+1)%4
        self.assertTrue(structurally_valid(p,reference))
        worse = scrambled(p, answer, 79)
        self.assertLess(score(p,worse), score(p,reference))
        return p, reference, worse

    def test_plateau_changes_working_board_without_lowering_saved_best(self):
        p, reference, worse = self.fixture()
        original=list(reference); escaped=[]; saved=[]
        def unchanged(problem, board, cells, **kwargs):
            return dict(board=list(board),score=score(problem,board),optimal=True)
        def escape(*args, **kwargs):
            escaped.append(True)
            return dict(board=list(worse),changed=True)
        with patch('smart_repair.assignment_repair',side_effect=unchanged), \
             patch.object(Diversifier,'perturb',side_effect=escape):
            result=solve(p,reference,seconds=2,plateau_rounds=1,region_repair_enabled=False,
                         should_stop=lambda:bool(escaped),on_improvement=saved.append)
        self.assertEqual(result['diversifications'],1)
        self.assertEqual(result['board'],original)
        self.assertEqual(reference,original)
        self.assertEqual(saved,[])
        self.assertEqual(result['termination'],'stopped')

    def test_tabu_prevents_immediate_matching_return_but_keeps_incumbent(self):
        p, reference, worse=self.fixture()
        calls=[]; escapes=[]
        def unchanged_or_return(problem, board, cells, **kwargs):
            calls.append(list(board))
            return dict(board=list(reference),score=score(problem,reference),optimal=True)
        def escape(*args, **kwargs):
            escapes.append(True)
            return dict(board=list(worse),changed=True)
        with patch('smart_repair.assignment_repair',side_effect=unchanged_or_return), \
             patch.object(Diversifier,'perturb',side_effect=escape):
            result=solve(p,reference,seconds=2,plateau_rounds=1,region_repair_enabled=False,
                         should_stop=lambda:len(calls)>=2)
        self.assertEqual(calls,[reference,worse])
        self.assertEqual(result['tabu_rejections'],1)
        self.assertEqual(result['board'],reference)

    def test_region_improvement_is_validated_and_returned(self):
        p,reference,worse=self.fixture()
        _,answer=generated_problem(4,7,28,clue_count=1)
        saved=[]
        def unchanged(problem,board,cells,**kwargs):
            return dict(board=list(board),score=score(problem,board),optimal=True)
        # Give the region budget a deterministic nonzero allowance without sleeping.
        ticks=iter(i*.001 for i in range(10000))
        with patch('smart_repair.assignment_repair',side_effect=unchanged), \
             patch('smart_repair.time.monotonic',side_effect=lambda:next(ticks)), \
             patch('repair_regions.select_region',return_value=[5,6]), \
             patch('repair_regions.region_repair',return_value={'board':answer}) as regional:
            result=solve(p,reference,seconds=2,diversify=False,region_every=1,on_improvement=saved.append)
        regional.assert_called_once()
        self.assertEqual(result['region_improvements'],1)
        self.assertEqual(result['score'],24)
        self.assertEqual(result['board'],answer)
        self.assertEqual(len(saved),1)

    def test_disabled_new_operators_and_invalid_configuration(self):
        p,reference,_=self.fixture()
        with patch.object(Diversifier,'perturb',side_effect=AssertionError('Disabled')), \
             patch('repair_regions.region_repair',side_effect=AssertionError('Disabled')):
            result=solve(p,reference,seconds=.03,diversify=False,region_repair_enabled=False)
        self.assertEqual(result['diversifications'],0)
        self.assertEqual(result['region_calls'],0)
        for options in ({'diversify':1},{'region_repair_enabled':None},{'plateau_rounds':0},
                        {'region_every':True},{'region_seconds':float('inf')},{'region_seconds':0}):
            with self.subTest(options=options),self.assertRaises(ValueError): solve(p,reference,**options)


if __name__=='__main__': unittest.main()
