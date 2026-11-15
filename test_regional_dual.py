import itertools
import random
import unittest
from dataclasses import replace
from exact_model import generated_problem
from test_regional_elimination import score_loss
from relational_regions import Work
from regional_dual import bound,certify


class DualCertificateControls(unittest.TestCase):
    def test_fractional_quantization_and_piece_prices_never_overcut(self):
        for seed in range(18):
            p,_=generated_problem(2,1+seed%3,seed,clue_count=seed%2)
            rng=random.Random(8009+seed)
            r=bound(p,scale=3)
            self.assertTrue(r['complete'])
            values={}
            for b in itertools.product(*p.domains):
                loss=score_loss(p,b)
                if loss is None:continue
                for c,s in enumerate(b):values[c,s]=min(values.get((c,s),loss),loss)
            for (c,s),v in values.items():self.assertLessEqual(r['by_state'][c][s],v)
            prices=[rng.randrange(-4,5) for _ in range(4)]
            fx=[{color:rng.randrange(4) for color in range(4)} for _ in range(4)]
            lower,rows=certify(p,p.domains,3,fx,prices,Work(100000))
            for (c,s),v in values.items():self.assertLessEqual(rows[c][s],v)
            self.assertLessEqual(lower,min(values.values()))
    def test_invalid_prices_and_cutoff(self):
        p,_=generated_problem(2,2,7)
        with self.assertRaises(ValueError):certify(p,p.domains,3,[{1:4}]*4,[0]*4,Work(10000))
        for options in (dict(max_work=0),dict(should_stop=lambda:True)):
            r=bound(p,**options)
            self.assertFalse(r['complete']);self.assertIsNone(r['by_state'])
