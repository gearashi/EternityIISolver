from dataclasses import replace
import itertools
import random
import unittest
from exact_model import generated_problem,constraint_edges
from test_refined_ordering import independent_oracle
from scripts.regional_elimination import bound


def score_loss(p,board):
    # Remove the scoring edges for the legality check. The same validator
    # directly enumerates frame/legal rotation, clue, and physical uniqueness.
    legal=replace(p,edges=(),domains=tuple((s,) for s in board))
    if independent_oracle(legal) is None:return None
    return sum(p.faces[board[a]][sa]!=p.faces[board[b]][sb] for a,b,sa,sb in constraint_edges(p))


class MinSumEliminationTests(unittest.TestCase):
    def test_bounds_include_unique_pieces_under_full_and_partial_edges(self):
        for seed in range(20):
            p,known=generated_problem(2,1+seed%3,seed,clue_count=seed%2)
            rng=random.Random(787+seed)
            q=replace(p,domains=tuple(tuple(s for s in d if rng.random()<.7) for d in p.domains),
                      edges=tuple(e for e in constraint_edges(p) if rng.random()<.65))
            values=[v for b in itertools.product(*q.domains) if (v:=score_loss(q,b)) is not None]
            for width in (2,3,4):
                r=bound(q,width=width)
                self.assertTrue(r['complete'],(seed,width,r))
                if values:
                    self.assertFalse(r['structurally_infeasible'])
                    self.assertLessEqual(r['lower_bound'],min(values))
                    if r['exact']:
                        self.assertEqual(r['lower_bound'],min(values))
                        self.assertEqual(score_loss(q,r['board']),min(values))
                if width==4:self.assertEqual(bool(r['structurally_infeasible']),not values)
    def test_cutoff_and_clue(self):
        p,_=generated_problem(3,2,7,clue_count=2)
        for r in (bound(p,max_work=0),bound(p,should_stop=lambda:True)):
            self.assertFalse(r['complete']);self.assertIsNone(r['board']);self.assertIsNone(r['lower_bound'])
        with self.assertRaises(ValueError):bound(p,width=1)
