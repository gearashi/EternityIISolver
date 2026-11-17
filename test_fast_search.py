"""Fast compiled search: adapter checks, real-helper runs on small puzzles, and worker wiring.

The real-helper tests build scripts/fast_dfs.cpp when no helper is present and
a C++17 compiler is available; otherwise they are skipped.
"""
import ast
from dataclasses import replace
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import tomllib
import unittest
import uuid
from unittest.mock import patch

import dashboard_server
import exact_worker
import fast_search
from exact_dfs import solve as reference_solve
from exact_model import check_solution, from_bundle, generated_problem, grid_edges, make_problem
from validator import load_bundle, validate_board

ROOT = Path(__file__).resolve().parent
HELPER = None
UNAVAILABLE = ''


def setUpModule():
    global HELPER, UNAVAILABLE
    name = 'eternity-fast.exe' if os.name == 'nt' else 'eternity-fast'
    built = ROOT / 'build' / 'native-helper' / name
    if not built.is_file():
        try:
            sys.path.insert(0, str(ROOT / 'scripts'))
            from build_fast_helper import build
            build(ROOT)
        except Exception as exc:  # no compiler on this machine
            UNAVAILABLE = f'{type(exc).__name__}: {exc}'[:200]
    if built.is_file():
        HELPER = built


class TemporaryState(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(tempfile.gettempdir()).resolve()
        self.home = self.temp_root / ('eternity-fast-search-' + uuid.uuid4().hex)
        self.home.mkdir()
        self.addCleanup(self.clean_state)

    def clean_state(self):
        if self.home.resolve().parent != self.temp_root or not self.home.name.startswith('eternity-fast-search-'):
            raise RuntimeError('Unexpected test cleanup path')
        shutil.rmtree(self.home)

    def need_helper(self):
        if HELPER is None:
            self.skipTest('Build the fast-search helper to exercise this path: ' + UNAVAILABLE)
        return HELPER


class AdapterTests(TemporaryState):
    def test_official_profile_matches_the_bundled_pieces(self):
        problem = from_bundle(load_bundle())
        self.assertEqual(fast_search.OFFICIAL_PIECES_SHA256, hashlib.sha256(fast_search._piece_text(problem).encode()).hexdigest())
        profile = fast_search.default_score_profile(problem)
        self.assertEqual(fast_search.OFFICIAL_SCORE_PROFILE['name'], profile['name'])
        sides = sum(color in profile['depletion_colors'] for piece in range(256) for color in problem.faces[4 * piece])
        self.assertLessEqual(profile['depletion_schedule'][-1][1], sides)
        self.assertLess(profile['depletion_schedule'][-1][0], profile['breaks'][0])
        for budget in range(len(profile['breaks']) + 1):
            arguments = fast_search._profile_arguments(profile, 256, budget)
            allowed = [int(x) for x in arguments[arguments.index('--breaks') + 1].split(',')] if budget else []
            self.assertEqual(budget, len(allowed))
            self.assertEqual(allowed, sorted(allowed))
            self.assertTrue(set(allowed) <= set(profile['breaks']))

    def test_other_pieces_get_the_plain_profile(self):
        problem, _ = generated_problem(6, 4, 3)
        profile = fast_search.default_score_profile(problem)
        self.assertEqual(('generic-late-breaks', ()), (profile['name'], profile['depletion_colors']))
        self.assertEqual(fast_search.default_score_profile(make_problem(16, generated_problem(16, 22, 1)[0].faces))['name'], 'generic-late-breaks')

    def test_invalid_requests_are_rejected_before_any_search(self):
        problem, answer = generated_problem(4, 3, 11)
        regional = replace(problem, edges=grid_edges(4)[:5])
        narrowed = replace(problem, domains=tuple((state,) for state in answer))
        for bad, options in ((regional, {}), (narrowed, {}), (problem, {'workers': 0}), (problem, {'workers': 65}),
                             (problem, {'mode': 'repair'}), (problem, {'seed': -1}), (problem, {'max_nodes': 0}),
                             (problem, {'mode': 'score', 'profile': {'breaks': (9, 3)}}),
                             (problem, {'mode': 'score', 'profile': {'breaks': (9,), 'depletion_colors': (1,)}}),
                             (problem, {'mode': 'score', 'reference': answer[:-1]})):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fast_search.solve(bad, 1, **dict({'state_dir': self.home, 'helper': __file__}, **options))
        self.assertEqual([], list(self.home.iterdir()))

    def test_zero_time_and_missing_helper(self):
        problem, _ = generated_problem(4, 3, 11)
        self.assertEqual('timeout', fast_search.solve(problem, 0, state_dir=self.home, helper=self.home / 'missing')['outcome'])
        self.assertEqual('stopped', fast_search.solve(problem, 5, should_stop=lambda: True, state_dir=self.home, helper=self.home / 'missing')['outcome'])
        with self.assertRaises(RuntimeError):
            fast_search.solve(problem, 5, state_dir=self.home, helper=self.home / 'missing')

    def test_score_board_is_an_independent_check(self):
        problem, answer = generated_problem(4, 3, 5, clue_count=2)
        self.assertEqual(24, fast_search.score_board(problem, answer))
        turned = list(answer)
        cell = next(c for c in (5, 6, 9, 10) if c not in problem.fixed)
        turned[cell] = turned[cell] // 4 * 4 + (turned[cell] + 1) % 4
        self.assertLess(fast_search.score_board(problem, turned), 24)
        duplicate = list(answer); duplicate[0] = duplicate[1]
        moved_clue = list(answer); a = next(iter(problem.fixed)); b = (a + 1) % 16; moved_clue[a], moved_clue[b] = moved_clue[b], moved_clue[a]
        for illegal in (answer[:-1], duplicate, moved_clue, [0.5] * 16):
            self.assertIsNone(fast_search.score_board(problem, illegal))

    def test_a_helper_board_with_a_false_score_is_refused(self):
        problem, answer = generated_problem(3, 3, 5)
        path = self.home / 'best.json'
        path.write_text(json.dumps({'claimed_score': 11, 'board': answer}))
        with self.assertRaises(RuntimeError):
            fast_search._helper_board(problem, path)
        path.write_text(json.dumps({'claimed_score': 12, 'board': answer}))
        self.assertEqual((12, answer), fast_search._helper_board(problem, path))

    def test_packaging_lists_the_module_and_the_helper(self):
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
        self.assertIn('fast_search', project['tool']['setuptools']['py-modules'])
        spec = (ROOT / 'EternityIISolver.spec').read_text(encoding='utf-8')
        tree = ast.parse(spec)
        hidden = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'hiddenimports' for t in n.targets))
        self.assertIn('fast_search', ast.literal_eval(hidden))
        self.assertIn('eternity-fast', spec)
        self.assertIn('scripts/fast_dfs.cpp', (ROOT / 'MANIFEST.in').read_text(encoding='utf-8'))


class RealHelperTests(TemporaryState):
    def test_strict_agrees_with_the_python_dfs(self):
        helper = self.need_helper()
        seen = set()
        for seed in range(12):
            problem, answer = generated_problem(4 + seed % 2, 3 + seed % 3, 100 + seed, clue_count=seed % 3)
            if seed % 2:  # usually impossible: one inner side of one free piece changes color
                piece = next(p for p in range(len(answer)) if p not in {s // 4 for s in problem.fixed.values()}
                             and 0 not in problem.faces[4 * p])
                base = list(problem.faces[4 * piece]); base[0] = base[0] % (3 + seed % 3) + 1
                faces = list(problem.faces)
                faces[4 * piece:4 * piece + 4] = [tuple(base[(side - r) % 4] for side in range(4)) for r in range(4)]
                problem = make_problem(problem.size, faces, problem.fixed)
            expected = reference_solve(problem, seconds=60, seed=1)['outcome']
            state = self.home / f'strict-{seed}'
            result = fast_search.solve(problem, 60, seed=seed, workers=1 + seed % 3, state_dir=state, helper=helper, prefix=2 + seed % 3)
            self.assertEqual(expected, result['outcome'], seed)
            seen.add(expected)
            if expected == 'solved':
                self.assertTrue(check_solution(problem, result['board']))
                self.assertEqual(len(grid_edges(problem.size)), result['score'])
            else:
                self.assertNotIn('proof_scope', result)
            self.assertGreater(result['branches'], 0)
        self.assertEqual({'solved', 'infeasible'}, seen)

    def test_score_ladder_reports_rising_validated_boards(self):
        helper = self.need_helper()
        problem, answer = generated_problem(6, 4, 21, clue_count=2)
        found, progress = [], []
        result = fast_search.solve(problem, 60, seed=3, workers=2, mode='score', state_dir=self.home, helper=helper,
                                   on_improvement=found.append, on_progress=progress.append)
        self.assertEqual('solved', result['outcome'])
        self.assertTrue(check_solution(problem, result['board']))
        scores = [item['score'] for item in found]
        self.assertEqual(scores, sorted(set(scores)))
        self.assertEqual(60, scores[-1])
        for item in found:
            self.assertEqual(item['score'], fast_search.score_board(problem, item['board']))
        targets = [item['target_score'] for item in progress]
        self.assertEqual(targets, sorted(targets))
        self.assertTrue(all(path.name.split('-')[0].isdigit() for path in (self.home / 'boards').iterdir()))

    def test_score_run_never_reports_boards_below_the_reference(self):
        helper = self.need_helper()
        problem, answer = generated_problem(5, 4, 8)
        found = []
        result = fast_search.solve(problem, 60, seed=1, mode='score', state_dir=self.home, helper=helper,
                                   reference=answer, on_improvement=found.append)
        self.assertEqual(('solved', []), (result['outcome'], found))

    def test_official_puzzle_stops_resumes_and_guards_its_checkpoints(self):
        helper = self.need_helper()
        problem = from_bundle(load_bundle())
        first = fast_search.solve(problem, 60, seed=5, workers=2, state_dir=self.home, helper=helper, max_nodes=2 ** 20)
        self.assertEqual(('timeout', False), (first['outcome'], first['resumed']))
        self.assertTrue(fast_search.has_saved_search(self.home))
        second = fast_search.solve(problem, 60, seed=5, workers=2, state_dir=self.home, helper=helper, max_nodes=2 ** 22)
        self.assertEqual(('timeout', True), (second['outcome'], second['resumed']))
        self.assertGreater(second['branches'], first['branches'])
        self.assertGreater(second['max_depth'], 100)
        for changed in ({'seed': 6, 'workers': 2}, {'seed': 5, 'workers': 3}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                fast_search.solve(problem, 60, state_dir=self.home, helper=helper, max_nodes=2 ** 22, **changed)
        stops = iter([False, False, True])
        stopped = fast_search.solve(problem, 60, seed=5, workers=2, state_dir=self.home, helper=helper,
                                    should_stop=lambda: next(stops, True), poll_seconds=0.05)
        self.assertEqual('stopped', stopped['outcome'])

    def test_official_score_boards_pass_the_release_validator(self):
        helper = self.need_helper()
        bundle = load_bundle()
        problem = from_bundle(bundle)
        found = []
        result = fast_search.solve(problem, 60, seed=2, workers=2, mode='score', state_dir=self.home, helper=helper,
                                   reference=list(bundle.record_board), on_improvement=found.append, max_nodes=3 * 10 ** 8)
        self.assertIn(result['outcome'], ('timeout', 'improved'))
        self.assertEqual([], [item for item in found if item['score'] <= 466])
        loosest = 480 - len(fast_search.OFFICIAL_SCORE_PROFILE['breaks'])
        for path in (self.home / 'boards').glob('*.json') if (self.home / 'boards').is_dir() else ():
            record = json.loads(path.read_text(encoding='utf-8'))
            report = validate_board(record['board'], bundle)
            self.assertTrue(report['valid'])
            self.assertEqual(record['score'], report['score'])
            self.assertGreaterEqual(report['score'], loosest)


class WorkerWiringTests(TemporaryState):
    def test_settings_accept_many_workers_only_for_fast_engines(self):
        for engine in exact_worker.FAST_ENGINES:
            self.assertIn(engine, exact_worker.ENGINES)
            self.assertEqual(16, exact_worker._settings(engine, 1.0, 1, 16, 8765)['workers'])
            self.assertEqual(8, dashboard_server.validate_settings({'method': 'exact', 'exact_engine': engine, 'workers': 8})['workers'])
            with self.assertRaises(ValueError):
                exact_worker._settings(engine, 1.0, 1, 65, 8765)
        with self.assertRaises(ValueError):
            exact_worker._settings('partitioned', 1.0, 1, 5, 8765)
        with self.assertRaises(ValueError):
            dashboard_server.validate_settings({'method': 'exact', 'exact_engine': 'cp-sat', 'workers': 8})
        page = (ROOT / 'Dashboard.html').read_text(encoding='utf-8')
        for engine in exact_worker.FAST_ENGINES:
            self.assertIn(f'<option value="{engine}">', page)

    def run_worker(self, engine, max_nodes, **options):
        # A placement cap ends each run, so the result does not depend on machine speed.
        capped = functools.partial(fast_search.solve, max_nodes=max_nodes)
        with patch.object(exact_worker, 'peak_memory_mb', return_value=64.0), \
                patch.object(fast_search, 'helper_path', return_value=self.need_helper()):
            code = exact_worker.run_task(self.home, engine=engine, seed=7, seconds=120.0,
                                         solver_factory=lambda _: capped, **options)
        return code, json.loads((self.home / 'runtime' / 'status.json').read_text(encoding='utf-8'))

    def test_worker_runs_resumes_and_restarts_the_strict_engine(self):
        code, status = self.run_worker('fast-dfs', 2 ** 20, workers=2)
        self.assertEqual((0, 'finished_bounded_run', 'fast-dfs', False), (code, status['state'], status['exact_engine'], status['resumed']))
        self.assertGreater(status['branches'], 0)
        self.assertEqual(466, status['best_score'])
        code, again = self.run_worker('fast-dfs', 2 ** 22, workers=2)
        self.assertEqual((0, True), (code, again['resumed']))
        self.assertGreater(again['branches'], status['branches'])
        code, mismatch = self.run_worker('fast-dfs', 2 ** 22, workers=1)
        self.assertEqual((1, 'error'), (code, mismatch['state']))
        self.assertIn('--fresh-exact', mismatch['error'])
        code, fresh = self.run_worker('fast-dfs', 2 ** 20, workers=1, fresh_exact=True)
        self.assertEqual((0, False, True), (code, fresh['resumed'], fresh['previous_checkpoint_archived']))

    def test_worker_keeps_the_saved_best_during_a_score_run(self):
        code, status = self.run_worker('fast-score', 10 ** 8, workers=2)
        self.assertEqual((0, 'finished_bounded_run', 466), (code, status['state'], status['best_score']))
        self.assertGreaterEqual(status['target_score'], 480 - len(fast_search.OFFICIAL_SCORE_PROFILE['breaks']))
        best = json.loads((self.home / 'runtime' / 'best.json').read_text(encoding='utf-8'))
        self.assertEqual(466, validate_board(best['board'], load_bundle())['score'])


if __name__ == '__main__':
    unittest.main()
