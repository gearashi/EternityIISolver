from dataclasses import replace
import os
from pathlib import Path
from time import monotonic
import unittest

from exact_model import generated_problem
from repair_component_family import prepare
from repair_native_radius import search, _decode
from repair_transfer_bound import certificate


class PackagedComponentRadiusTests(unittest.TestCase):
    def setUp(self):
        p,self.parent=generated_problem(3,3,57)
        self.parent[4]=self.parent[4]//4*4+(self.parent[4]+1)%4
        self.problem=replace(p,domains=tuple((s,) for s in self.parent))
        self.prices=certificate(self.problem,self.parent)['potentials']

    def test_complete_empty_cover_and_explicit_cutoffs(self):
        for work,deadline,status in ((10**8,monotonic()+5,'complete'),(0,monotonic()+5,'work_limit'),
                                     (10**8,monotonic()-1,'timeout')):
            cover,rows=prepare(self.problem,self.parent,self.prices,6,work,deadline)
            self.assertEqual(cover['status'],status)
            self.assertLessEqual(cover['work'],work)
            self.assertEqual(rows,'0\n' if status=='complete' else '')

    def test_restricted_scope_is_never_accepted_by_unrestricted_parser(self):
        with self.assertRaises(RuntimeError):
            _decode('family_complete 0 5 5 0 0 0 0 0',self.problem,self.parent,6,10)
        self.assertEqual(_decode('family_complete 0 5 5 0 0 0 0 0',self.problem,self.parent,
                                 6,10,conditional=True)[0],'family_complete')

    def test_v1_only_binary_preserves_completed_lower_radius(self):
        import subprocess
        from types import SimpleNamespace
        from unittest.mock import patch
        def legacy(command,**kwargs):
            if len(command)==1:
                return SimpleNamespace(stdout='complete 0 0 0 0 0 0 0 0\n')
            raise subprocess.CalledProcessError(2,command)
        with patch('repair_native_radius.subprocess.run',side_effect=legacy):
            result=search(self.problem,self.parent,radius=6,helper=__file__)
        self.assertEqual(result['termination'],'native_helper_incompatible')
        self.assertEqual(result['max_exhausted_radius'],4)

    def test_ignored_family_selector_cannot_replace_a_scoped_proof(self):
        status,_,_=_decode('complete 0 5 5 0 0 0 0 0',self.problem,self.parent,6,10,
                            conditional=True,expected_rows=2)
        self.assertEqual(status,'native_helper_incompatible')

    def test_complete_cover_requires_every_supplied_row(self):
        header='family_complete 0 5 5 0 0 0 0 0'
        for trailing in ('', 'component_rows 1 1', 'component_rows 2 1', 'component_rows 3 3',
                         'component_rows 0 1', 'component_rows 2', 'component_rows no 2'):
            with self.subTest(trailing=trailing),self.assertRaises(RuntimeError):
                _decode(header+'\n'+trailing,self.problem,self.parent,6,10,conditional=True,expected_rows=2)
        _,counts,_=_decode(header+'\ncomponent_rows 2 2',self.problem,self.parent,6,10,
                           conditional=True,expected_rows=2)
        self.assertEqual(counts['component_rows_finished'],2)

    def test_valid_early_witness_can_leave_unexplored_rows(self):
        from test_native_radius_cycle import planted_cycle
        from exact_optimize import evaluate_board
        problem,parent,solution=planted_cycle(5,(6,8,16,18))
        gain=evaluate_board(problem,solution)['score']-evaluate_board(problem,parent)['score']
        head=f'family_complete {gain} 10 3 1 0 1 1 4'
        raw=head+'\n'+' '.join(map(str,solution))+'\ncomponent_rows 2 1'
        _,counts,board=_decode(raw,problem,parent,4,20,conditional=True,expected_rows=7)
        self.assertEqual(board,solution)
        self.assertEqual(counts['component_rows_finished'],1)
        with self.assertRaises(RuntimeError):
            _decode(raw.replace('component_rows 2 1','component_rows 2 2'),
                    problem,parent,4,20,conditional=True,expected_rows=7)

    def test_six_through_public_api(self):
        name='eternity-radius.exe' if os.name=='nt' else 'eternity-radius'
        exe=Path(__file__).resolve().parent/'build'/'native-helper'/name
        if not exe.is_file():
            self.skipTest('no native research binary')
        r=search(self.problem,self.parent,radius=6,helper=exe)
        self.assertEqual((r['termination'],r['max_exhausted_radius']),('exhausted',6))
        covers=[e for e in r['native_phases'] if e['phase']=='component_cover']
        self.assertEqual([e['radius'] for e in covers],[5,6])
        self.assertTrue(all(e['status']=='complete' for e in covers))


if __name__=='__main__':
    unittest.main()
