import unittest
from scripts.benchmark_parent_learning import bank_for, gather
from repair_parent_pair_cli import choose_pair, WEIGHTS


class ParentPairTests(unittest.TestCase):
    def test_matches_frozen_pair_ranking(self):
        p,b = bank_for(370)
        rows,baseline,_,_ = gather(p,b,370)
        expected = max(rows,key=lambda pair:(sum(w*x for w,x in zip(WEIGHTS,rows[pair]['x'])),
                                            tuple(-v for v in pair)))
        selected=choose_pair(p,b)
        self.assertEqual(selected['pair'],expected)
        self.assertEqual(selected['pairs_scored'],36)
        self.assertEqual(selected['bank_best'],baseline)

    def test_invalid_and_repeated_inputs(self):
        p,b = bank_for(370)
        repeated=[b[0],b[0],b[1]]
        self.assertEqual(choose_pair(p,repeated)['pair'],(0,2))
        for bad in ([b[0]], [b[0],b[0]], [b[0]]*65, [b[0],[-1]*len(b[0])]):
            with self.assertRaises(ValueError):
                choose_pair(p,bad)


if __name__=='__main__':
    unittest.main()
