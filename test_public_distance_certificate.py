import json
from copy import deepcopy
from pathlib import Path
import unittest

from scripts.verify_distance_certificate import verify

ROOT=Path(__file__).resolve().parent


class PublicCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pieces=[tuple(map(int,row.split())) for row in (ROOT/'data/pieces.txt').read_text().splitlines() if row.strip()]
        cls.parent=json.loads((ROOT/'data/record466.json').read_text())['board']
        cls.cert=json.loads((ROOT/'docs/record466-distance.json').read_text())

    def test_raw_tiles_reproduce_checked_fractional_bound(self):
        r=verify(self.pieces,self.parent,self.cert)
        self.assertEqual((r['lower_bound'],r['numerator'],r['scale'],r['subsets']),(24,376,16,5961))

    def test_tampered_scope_total_and_region_cannot_pass(self):
        for key,value in (('target_score',467),('numerator',377),('parent_local_sha256','0'*64)):
            with self.subTest(field=key),self.assertRaises(ValueError):
                verify(self.pieces,self.parent,{**self.cert,key:value})
        bad=deepcopy(self.cert)
        bad['regional_certificates']=[[[0],1,16]]
        with self.assertRaisesRegex(ValueError,'Regional lower bound'):
            verify(self.pieces,self.parent,bad)
