"""Exhaustive finite oracles for the optional binary objective fast path."""
import itertools
import random
import unittest
from binary_pairwise import maximize


def objective(unary, pairs, assignment):
    return sum(row[x] for row,x in zip(unary, assignment)) + sum(row[2*assignment[a]+assignment[b]] for (a,b),row in pairs.items())


class BinaryPairwiseTests(unittest.TestCase):
    def test_random_forests_and_cycles_match_exhaustive_oracle(self):
        rng=random.Random(17013)
        for n in range(0,9):
            for trial in range(16):
                unary=[tuple(rng.randrange(-5,6) for _ in range(2)) for _ in range(n)]
                pairs={(a,b):tuple(rng.randrange(-4,5) for _ in range(4)) for a in range(n) for b in range(a+1,n) if rng.random()<.32}
                expected=max(objective(unary,pairs,x) for x in itertools.product((0,1),repeat=n))
                found=maximize(unary,pairs,seconds=2)
                self.assertTrue(found['optimal'],(n,trial,found))
                self.assertEqual(found['value'],expected)
                self.assertEqual(objective(unary,pairs,found['assignment']),expected)

    def test_large_tree_avoids_exponential_enumeration(self):
        unary=[(0,0)]*256
        pairs={(v//2,v):(0,1,1,0) for v in range(1,256)}
        result=maximize(unary,pairs,seconds=2,max_component_bits=1)
        self.assertTrue(result['optimal'])
        self.assertEqual(result['value'],255)
        self.assertEqual(result['tree_components'],1)
        self.assertEqual(result['assignments_evaluated'],0)

    def test_separable_pairs_do_not_create_false_cycles(self):
        unary=[(0,0)]*20
        pairs={(a,b):(3,5,7,9) for a in range(20) for b in range(a+1,20)}
        result=maximize(unary,pairs,seconds=2,max_component_bits=1)
        self.assertTrue(result['optimal'])
        self.assertEqual(result['value'],9*190)
        self.assertEqual(result['tree_components'],20)
        self.assertEqual(result['removed_separable_pairs'],190)

    def test_unsupported_or_interrupted_never_claims_optimum(self):
        unary=[(0,1)]*5
        pairs={(a,b):(0,0,0,1) for a in range(5) for b in range(a+1,5)}
        for kwargs,expected in ((dict(max_component_bits=4),'unsupported_component'),(dict(seconds=0),'timeout'),(dict(should_stop=lambda:True),'stopped')):
            result=maximize(unary,pairs,**kwargs)
            self.assertFalse(result['optimal']); self.assertIsNone(result['assignment'])
            self.assertIsNone(result['value']); self.assertEqual(result['termination'],expected)
        ticks=[0]
        def stop():
            ticks[0]+=1
            return ticks[0]>13
        result=maximize(unary,pairs,seconds=2,should_stop=stop)
        self.assertFalse(result['optimal']);self.assertEqual(result['termination'],'stopped')

    def test_input_validation_and_callback_failure(self):
        for unary,pairs,kwargs in (([(True,1)],{},{}),([(0,1)],{(0,0):(0,0,0,1)},{}),([],{},dict(seconds=-1)),([],{},dict(max_component_bits=21)),([(0,1),(0,1)],{(1,0):(0,0,0,1)},{})):
            with self.assertRaises(ValueError): maximize(unary,pairs,**kwargs)
        def broken(): raise OSError('callback failed')
        with self.assertRaises(OSError):maximize([],{},should_stop=broken)


if __name__=='__main__':unittest.main()
