"""Conditional neighbor-capacity dispatch, opt-in routing, and owned frontier checks."""
import ast
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import dashboard_server
import exact_dfs
import exact_worker
import launcher
from dfs_profiles import dfs_profile_options
from exact_model import make_problem, check_solution
from neighbor_hall import hall_proves_choice


PROFILE = 'full-neighbor-hall'
IDENTITY = '0004bea82577b460560837a2cb02b17f3d29a56bfaa443911fece678d1b48624'
EDGES = ((8, 14, 2, 0), (14, 15, 1, 3), (14, 20, 2, 0))


def star(edges=EDGES, extra_rotation=False):
    # Red at cell 14 needs three different red pieces. Only tiles 8 and 15
    # can fill those neighboring cells, each with two orientations.
    slots = (8, 15, 20, 25, 10)
    faces, domains, fixed = [], [], {}
    for cell in range(36):
        color = 1 if cell in (8, 14, 15) else 2
        row, col = divmod(cell, 6)
        base = tuple(0 if outside else color for outside in (row == 0, col == 5, row == 5, col == 0))
        if extra_rotation and cell == 20:
            base = (1, 2, 2, 2)
        faces.extend(tuple(base[(i-r) % 4] for i in range(4)) for r in range(4))
        if cell in slots:
            domains.append([4*t+r for t in slots for r in (0, 2)])
        elif cell in (14, 28):
            domains.append([4*t+r for t in (14, 28) for r in (0, 2)])
        elif cell == 7:
            # This legal independent cell receives the ordinary shallow probe.
            domains.append([4*cell+r for r in (0, 1, 2)])
        else:
            fixed[cell] = 4*cell
            domains.append([4*cell])
    return make_problem(6, faces, fixed, domains, edges)


class NeighborHallTests(unittest.TestCase):
    def test_all_orientations_and_required_edges(self):
        for problem, expected in ((star(), True), (star(EDGES[1:]), False), (star(extra_rotation=True), False)):
            e = exact_dfs._Search(problem, 15, 17, None, None, None, None, 5, options=dfs_profile_options(PROFILE))
            masks = e.build()
            self.assertIsNotNone(e.propagate(masks, range(36), None))
            for kind, target, choice in (('cell', 14, 56), ('piece', 14, 14), ('edge', 0, 1)):
                detail = {}
                found = hall_proves_choice(e, dict(kind=kind, cell=target, domains=masks), choice, audit=detail)
                self.assertEqual(found, expected)
                if found:
                    refuted = next(row for row in detail['parts'] if row['refuted'])
                    self.assertEqual(refuted['states'], [r['state'] for r in refuted['roots']])
                    for r in refuted['roots']:
                        self.assertEqual(set(r['certificate']['neighbors']), {8, 15, 20})
                        self.assertEqual(int(r['certificate']['union'], 16), (1 << 8) | (1 << 15))
            self.assertEqual((e.probe_checks, e.branches), (0, 0))

    def test_interrupted_choice_and_compatible_frontier(self):
        p = star()
        old = dfs_profile_options('full-neighbor')
        opts = dfs_profile_options(PROFILE)
        self.assertEqual(opts, dict(old, neighbor_hall_v1=True))
        base = exact_dfs.solve(p, seconds=15, seed=17, options=old)
        ranked = exact_dfs.solve(p, seconds=15, seed=17, options=opts)
        self.assertEqual(ranked['board'], base['board'])
        self.assertTrue(check_solution(p, ranked['board']))
        self.assertEqual((ranked['branches']+1, ranked['probe_checks']), (base['branches'], base['probe_checks']))
        def interrupt(search, frame, choice):
            raise exact_dfs._Interrupted('stopped')
        with patch.object(exact_dfs, 'hall_proves_choice', interrupt):
            mid = exact_dfs.solve(p, seconds=15, seed=17, options=opts)
        cp = mid['checkpoint']
        self.assertEqual(cp['options_hash'], IDENTITY)
        self.assertEqual((cp['stack'][0]['candidates'], cp['stack'][0]['next']), ([14, 28], 0))
        continued = exact_dfs.solve(p, seconds=15, seed=17, options=opts, checkpoint=cp)
        for name in ('board', 'branches', 'probe_checks'):
            self.assertEqual(continued[name], ranked[name])
        with self.assertRaises(ValueError):
            exact_dfs.solve(p, seconds=15, seed=17, options=old, checkpoint=cp)

    def test_worker_roundtrip_and_distribution_membership(self):
        args = launcher.parser().parse_args(['start', '--method', 'exact', '--exact-engine', 'dfs', '--workers', '1', '--pruning', PROFILE])
        self.assertIn(PROFILE, launcher.run_arguments(args))
        cfg = dashboard_server.validate_settings(dict(method='exact', exact_engine='dfs', workers=1, pruning=PROFILE))
        root = Path(__file__).resolve().parent
        command = dashboard_server.worker_command(root, 8765, cfg)
        self.assertEqual(command[command.index('--pruning')+1], PROFILE)
        for engine, workers in [('dfs', 2), ('hybrid', 1), ('partitioned', 1), ('impact-dfs', 1)]:
            with self.assertRaises(ValueError):
                dashboard_server.validate_settings(dict(cfg, exact_engine=engine, workers=workers))
        with tempfile.TemporaryDirectory(prefix='eii-neighbor-hall-test-') as tmp:
            home = Path(tmp)
            for resumed in (False, True):
                self.assertEqual(exact_worker.run_task(home,engine='dfs',workers=1,seconds=0,pruning=PROFILE),0)
                status = json.loads((home/'runtime/status.json').read_text())
                cp = json.loads((home/'runtime/exact-dfs-checkpoint.json').read_text())
                self.assertEqual((status['pruning'], status['resumed'], cp['options_hash']), (PROFILE, resumed, IDENTITY))
        project = tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))
        self.assertIn('neighbor_hall', project['tool']['setuptools']['py-modules'])
        spec = ast.parse((root/'EternityIISolver.spec').read_text(encoding='utf-8'))
        imports = next(n.value for n in spec.body if isinstance(n,ast.Assign)
                       and any(isinstance(t,ast.Name) and t.id=='hiddenimports' for t in n.targets))
        self.assertIn('neighbor_hall', ast.literal_eval(imports))
