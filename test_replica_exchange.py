import copy
import unittest
from types import SimpleNamespace
import numpy as np
from replica_exchange import exchange_plan, exchange_engine
from validator import load_bundle, validate_board


class Array:
    def __init__(self, value): self.data = np.asarray(value).copy()
    def get(self): return self.data.copy()


class ExchangeTests(unittest.TestCase):
    def test_favorable_exchange_cold_receives_better_board(self):
        r = exchange_plan([430, 460], [.1, 1.], seed=19)
        self.assertEqual(r['permutation'], [1, 0])
        self.assertEqual(r['accepted'], 1)

    def test_unfavorable_large_gap_rejected(self):
        self.assertEqual(exchange_plan([480, 0], [.01, 2.], seed=19)['accepted'], 0)

    def test_pairs_disjoint_and_round_alternates(self):
        a = exchange_plan([450]*7, [1.]*7, seed=19)
        b = exchange_plan([450]*7, [1.]*7, seed=19, round_index=1)
        self.assertEqual(a['permutation'], [1,0,3,2,5,4,6])
        self.assertEqual(b['permutation'], [0,2,1,4,3,6,5])
        self.assertEqual(a, exchange_plan([450]*7,[1.]*7,seed=19))

    def test_invalid_inputs_rejected(self):
        for scores, temps in (([True],[1.]), ([481],[1.]), ([3],[0.]),
                              ([3],[float('nan')]), ([3],[]), ([],[])):
            with self.assertRaises(ValueError): exchange_plan(scores,temps,seed=1)
        with self.assertRaises(ValueError): exchange_plan([1],[1.],seed=True)
        with self.assertRaises(ValueError): exchange_plan([1],[1.],seed=1,round_index=-1)

    def fixture(self):
        bundle = load_bundle()
        good = np.asarray(bundle.record_board, np.int16)
        bad = good.copy()
        free = next(c for c in range(256) if c not in bundle.fixed_clues and 0 not in bundle.oriented_edges[int(good[c])])
        bad[free] = (bad[free]//4)*4 + (bad[free]+1)%4
        bs = validate_board(bad,bundle)['score']
        boards = np.stack([bad,good],axis=1)
        scores = np.asarray([bs,466],np.int32)
        engine = SimpleNamespace(n=2,bundle=bundle,xp=SimpleNamespace(asarray=Array),
            boards=Array(boards),scores=Array(scores),bestboards=Array(boards),bestscores=Array(scores),
            temps=Array([.1,1.]),positions=Array(np.argsort(boards.T//4,axis=1).astype(np.int16).T))
        return engine

    def test_engine_moves_state_together_and_protects_best(self):
        g=self.fixture(); before=g.boards.get(); best=g.bestscores.get()
        r=exchange_engine(g,seed=1)
        np.testing.assert_array_equal(g.boards.get(),before[:,[1,0]])
        np.testing.assert_array_equal(g.positions.get(),np.argsort(g.boards.get().T//4,axis=1).astype(np.int16).T)
        self.assertTrue(np.all(g.bestscores.get()>=best))
        self.assertEqual(r['new_move_proposals'],0)
        self.assertEqual(r['global_best'],466)
        for board,s in zip(g.boards.get().T,g.scores.get()):
            self.assertEqual(validate_board(board,g.bundle)['score'],s)

    def test_rejects_bad_state_before_mutation(self):
        g=self.fixture(); old=g.boards
        g.scores=Array(np.asarray([0,0],np.int32))
        with self.assertRaises(ValueError): exchange_engine(g,seed=1)
        self.assertIs(g.boards,old)

    def test_allocation_failure_is_atomic(self):
        g=self.fixture(); old={k:getattr(g,k) for k in ('boards','scores','positions','bestboards','bestscores')}
        count=0
        def fail(value):
            nonlocal count
            count+=1
            if count==3: raise MemoryError()
            return Array(value)
        g.xp=SimpleNamespace(asarray=fail)
        with self.assertRaises(MemoryError): exchange_engine(g,seed=1)
        for k,v in old.items(): self.assertIs(getattr(g,k),v)

if __name__=='__main__': unittest.main()
