"""Real dispatch choices and compatibility of the opt-in full-dispatch profile."""
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import dashboard_server
import exact_dfs
import exact_parallel
import exact_worker
import launcher
from dfs_profiles import dfs_profile_options
from exact_model import check_solution, generated_problem, grid_edges, make_problem


def chain_problem():
    """Only one of the two color patterns on the six-cell row uses distinct tiles."""
    problem, board = generated_problem(8, 3, 314)
    path = range(9, 15)
    spare = 18
    inner = {cell for cell in range(64) if 0 < cell//8 < 7 and 0 < cell%8 < 7}
    dummy = inner - set(path) - {spare}
    faces = list(problem.faces)
    rows = [[state] for state in board]
    for cell in inner:
        state = board[cell]
        face = (3, 3, 3, 3)
        if cell in path[1:-1]:
            face = (1, 2, 1, 1)
        elif cell in (path[0], path[-1]):
            face = (1, 1, 1, 1)
        elif cell == spare:
            face = (2, 2, 2, 2)
        for rotation in range(4):
            faces[4*(state//4) + (state+rotation)%4] = tuple(face[(side-rotation)%4] for side in range(4))
        rows[cell] = [state, 4*(state//4) + (state+2)%4]
    first, last, shared = (board[c] for c in (path[0], path[-1], spare))
    rows[path[0]], rows[path[-1]], rows[spare] = [first, shared], [last, shared], [first, last, shared]
    edges = [e for e in grid_edges(8) if (e[0] in path and e[1] in path) or (e[0] in dummy and e[1] in dummy)]
    return make_problem(8, faces, domains=rows, edges=edges)


class DispatchProfileTests(unittest.TestCase):
    def test_selection_fingerprints_and_worker_forwarding(self):
        old = dfs_profile_options('full-pivot')
        enabled = dfs_profile_options('full-dispatch')
        self.assertEqual(enabled, dict(old, pivot_dispatch_v1=True))
        self.assertEqual(exact_dfs.normalize_options(dict(old, pivot_dispatch_v1=False)), exact_dfs.normalize_options(old))
        for options in ({'pivot_dispatch_v1': True}, dict(old, pivot_dispatch_v1=1)):
            with self.assertRaises(ValueError):
                exact_dfs.normalize_options(options)
        args = launcher.parser().parse_args(['start','--method','exact','--pruning','full-dispatch','--no-browser'])
        with patch.object(launcher,'ensure_dashboard',return_value='http://127.0.0.1:8765/'), \
             patch.object(launcher,'local_urlopen',return_value=BytesIO(b'{"accepted":true}')) as send, \
             redirect_stdout(StringIO()):
            self.assertEqual(launcher.start(args),0)
        request = json.loads(send.call_args.args[0].data)
        settings = dashboard_server.validate_settings(request)
        cmd = dashboard_server.worker_command(Path.cwd(),8765,settings)
        self.assertEqual(cmd[cmd.index('--pruning')+1],'full-dispatch')
        for engine in ('hybrid','partitioned','impact-dfs','cp-sat'):
            with self.assertRaisesRegex(ValueError,'standard DFS'):
                exact_worker._settings(engine,1,3,1,8765,pruning='full-dispatch')
            with self.assertRaisesRegex(ValueError,'standard DFS'):
                dashboard_server.validate_settings(dict(request,exact_engine=engine))
        with self.assertRaises(ValueError):
            dashboard_server.validate_settings(dict(request,workers=2))
        p,board = generated_problem(2,3,41)
        with self.assertRaisesRegex(ValueError,'without ordering hints'):
            exact_dfs.solve(p,seconds=0,hints=[board],options=enabled)
        with self.assertRaisesRegex(ValueError,'partitioned'):
            exact_parallel.solve(p,seconds=0,options=enabled)
        pending = exact_dfs.solve(p,seconds=0,options=old)['checkpoint']
        self.assertEqual(pending['options_hash'],'83eda5fb7b42fba41bffb528e65662493706d9aaa9d28e3298a25f5a02434767')
        fresh = exact_dfs.solve(p,seconds=0,options=enabled)['checkpoint']
        self.assertNotEqual(fresh['options_hash'],pending['options_hash'])
        for wrong,checkpoint in ((old,fresh),(enabled,pending)):
            with self.assertRaises(ValueError):
                exact_dfs.solve(p,seconds=0,options=wrong,checkpoint=checkpoint)

    def test_real_worker_saves_and_resumes_dispatch_profile(self):
        with TemporaryDirectory(prefix='eternity-dispatch-worker-') as workspace:
            home = Path(workspace)
            for expected_resume in (False,True):
                self.assertEqual(exact_worker.run_task(home,engine='dfs',workers=1,seconds=0,pruning='full-dispatch'),0)
                status = json.loads((home/'runtime/status.json').read_text(encoding='utf-8'))
                checkpoint = json.loads((home/'runtime/exact-dfs-checkpoint.json').read_text(encoding='utf-8'))
                self.assertIs(status['resumed'],expected_resume)
                self.assertEqual(status['pruning'],'full-dispatch')
                self.assertEqual(status['exact_outcome'],'timeout')
                if expected_resume:
                    self.assertEqual(checkpoint['options_hash'],old_hash)
                old_hash = checkpoint['options_hash']
        rejected = exact_dfs.solve(chain_problem(),seconds=0,options=dfs_profile_options('full-pivot'))['options_hash']
        self.assertNotEqual(old_hash,rejected)

    def test_skips_a_tried_orientation_and_preserves_an_unvisited_sibling(self):
        puzzle = chain_problem()
        for seed, savings in ((29,1),(126,0)):
            base = exact_dfs.solve(puzzle,seconds=10,seed=seed,options=dfs_profile_options('full-pivot'))
            calls = []
            original = exact_dfs._Search.skip_proved_choice
            def record(search,frame,state):
                calls.append((frame['next'],state))
                original(search,frame,state)
            with patch.object(exact_dfs._Search,'skip_proved_choice',record):
                got = exact_dfs.solve(puzzle,seconds=10,seed=seed,options=dfs_profile_options('full-dispatch'))
            self.assertEqual(base['branches']-got['branches'],savings)
            self.assertEqual(len(calls),savings)
            self.assertEqual(base['probe_checks'],got['probe_checks'])
            self.assertEqual(base['board'],got['board'])
            self.assertTrue(check_solution(puzzle,got['board']))

    def test_interruption_after_skip_preserves_original_list_and_next_cell(self):
        p = chain_problem()
        opts = dfs_profile_options('full-dispatch')
        stopping = False
        expected = {}
        original = exact_dfs._Search.skip_proved_choice
        def record(s,frame,choice):
            nonlocal stopping
            expected['candidates'] = frame['candidates'].copy()
            expected['next'] = frame['next'] + 1
            expected['branches'] = s.branches
            original(s,frame,choice)
            stopping = True
        with patch.object(exact_dfs._Search,'skip_proved_choice',record):
            stopped = exact_dfs.solve(p,seed=29,seconds=10,options=opts,should_stop=lambda: stopping)
        checkpoint = stopped['checkpoint']
        saved = checkpoint['stack'][-1]
        self.assertEqual((saved['candidates'],saved['next']),(expected['candidates'],expected['next']))
        self.assertNotIn('pivot_excluded',saved)
        self.assertEqual(checkpoint['counters']['branches'],expected['branches'])
        resumed = exact_dfs.solve(p,seed=29,seconds=10,options=opts,checkpoint=checkpoint)
        fresh = exact_dfs.solve(p,seed=29,seconds=10,options=opts)
        for key in ('board','branches','probe_checks'):
            self.assertEqual(resumed[key],fresh[key])


if __name__ == '__main__':
    unittest.main()
