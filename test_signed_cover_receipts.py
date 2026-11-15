import unittest

from exact_model import generated_problem
from repair_signed_cover import decode


class SignedCoverReceipts(unittest.TestCase):
    def setUp(self):
        self.problem,self.board=generated_problem(2,3,83)
        self.cover=dict(status='complete',shape_rows=2,patterns=1,nominal=2)

    def read(self, result, cover=None):
        return decode(result,self.problem,self.board,2,5,self.cover if cover is None else cover)

    def test_partial_catalog_and_falsified_completion_cannot_prove_absence(self):
        header='family_complete 0 5 3 2 2 0 0 0'
        receipt=header+'\nsigned_rows 2 1 1 2'
        self.assertEqual(self.read(receipt)[0],'family_complete')
        for change in (dict(status='timeout'),dict(status='work_limit'),dict(patterns=True),
                       dict(nominal=-1),dict(shape_rows=0),dict(patterns=3)):
            with self.assertRaises(RuntimeError):
                self.read(receipt,dict(self.cover,**change))
        for suffix in ('1 1 1 2','2 1 0 2','2 1 1 1','2 1 1 3'):
            with self.assertRaises(RuntimeError):
                self.read(header+'\nsigned_rows '+suffix)

    def test_interruption_counts_at_most_one_unfinished_leaf(self):
        partial='work_limit 0 5 2 1 0 1 1 0\nsigned_rows 2 1 0 0'
        status,counts,_=self.read(partial)
        self.assertEqual(status,'work_limit')
        self.assertEqual(counts['nominal_covered'],0)
        with self.assertRaises(RuntimeError):
            self.read('work_limit 0 5 2 2 0 1 0 0\nsigned_rows 2 1 0 0')

    def test_partial_row_loading_cannot_claim_a_finished_pattern(self):
        text='work_limit 0 5 0 0 0 0 0 5\nsigned_rows 1 0 0 0'
        self.assertEqual(self.read(text)[1]['shape_rows'],1)
        with self.assertRaises(RuntimeError):
            self.read(text.replace('signed_rows 1 0 0 0','signed_rows 1 1 1 0'))
