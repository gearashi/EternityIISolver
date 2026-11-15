from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from exact_model import generated_problem
from repair_native_radius import search,_decode


class NativeDistanceTests(unittest.TestCase):
    def setUp(self):
        p,self.board=generated_problem(3,3,57)
        self.board[4]=self.board[4]//4*4+(self.board[4]+1)%4
        self.p=replace(p,domains=tuple((s,) for s in self.board))

    def test_missing_helper_preserves_exhausted_prefix(self):
        outcome=search(self.p,self.board,radius=5,helper=Path(__file__)/'missing')
        self.assertEqual(('native_helper_unavailable',3,self.board),
                         (outcome['termination'],outcome['max_exhausted_radius'],outcome['board']))

    def test_zero_time_or_work_cannot_prove_radius(self):
        for options,status in ((dict(seconds=0),'timeout'),(dict(max_work=0),'work_limit')):
            r=search(self.p,self.board,radius=5,**options)
            self.assertEqual((status,0),(r['termination'],r['max_exhausted_radius']))

    def test_helper_protocol_rejects_bad_counts_and_witness(self):
        for raw in ('','ok 0 0 0 0 0 0 0 0','complete 0 2 0 0 0 0 0 0',
                    'complete 1 1 1 0 0 0 0 0\n'+' '.join(map(str,self.board)),
                    'work_limit 0 2 2 0 0 0 0 0','complete 0 0 0 0 1 0 0 0'):
            with self.subTest(raw=raw),self.assertRaises(RuntimeError):
                _decode(raw,self.p,self.board,4,3)
        state,counts,b=_decode('work_limit 0 3 3 0 0 0 0 0',self.p,self.board,4,3)
        self.assertEqual(('work_limit',3,self.board),(state,counts['work'],b))

    def test_lower_radius_completed_if_next_helper_call_times_out(self):
        import subprocess
        def run(argv,**kwargs):
            if kwargs['input'].splitlines()[0].split()[1]=='4':
                return SimpleNamespace(stdout='complete 0 0 0 0 0 0 0 0\n')
            raise subprocess.TimeoutExpired(argv,1)
        with patch('repair_native_radius.subprocess.run',side_effect=run):
            r=search(self.p,self.board,radius=5,helper=__file__)
        self.assertEqual(('timeout',4),(r['termination'],r['max_exhausted_radius']))
        self.assertIsNone(r['native_phases'][-1]['work'])

    def test_real_compiled_helper_when_present(self):
        import os
        filename='eternity-radius.exe' if os.name=='nt' else 'eternity-radius'
        exe=Path(__file__).resolve().parent/'build'/'native-helper'/filename
        if not exe.exists():self.skipTest('Build the optional native helper to exercise this path')
        r=search(self.p,self.board,radius=5,helper=exe)
        self.assertEqual(('exhausted',5),(r['termination'],r['max_exhausted_radius']))
