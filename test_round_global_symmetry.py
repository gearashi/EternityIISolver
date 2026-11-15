import unittest
from dataclasses import replace

from exact_model import make_problem
from scripts.check_board_symmetry import isometries


class GlobalSymmetryTests(unittest.TestCase):
    def setUp(self):
        faces=[]
        for tile in ((0,0,1,1),(0,0,2,2),(0,0,1,2),(0,0,2,1)):
            faces.extend(tile[i:]+tile[:i] for i in range(4))
        self.p=make_problem(2,faces)

    def test_all_geometric_and_color_symmetries(self):
        report=isometries(self.p)
        self.assertEqual(report['distinct_orbits'],4)
        for row in report['cases']:
            self.assertTrue(row['complete'])
            self.assertEqual(len(row['preserving']),8)
            self.assertEqual(row['tested'],2)

    def test_oriented_anchor_breaks_color_exchange_and_three_rotations(self):
        row=isometries(replace(self.p,fixed={0:1}))['cases'][0]
        self.assertEqual(row['preserving'],[dict(quarter_turns=0,color_map={0:0,1:1,2:2})])

    def test_incomplete_permutation_cap_is_explicit(self):
        self.assertTrue(all(not r['complete'] for r in isometries(self.p,cap=1)['cases']))
