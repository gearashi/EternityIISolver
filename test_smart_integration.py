"""Experimental search wiring: fake backends and isolated state, no real search."""
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dashboard_server
import exact_worker
import launcher
import test_library_dashboard as library_tests
from validator import load_bundle, validate_board


class SmartIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='eternity-smart-integration-')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.runtime = self.home / 'runtime'
        self.runtime.mkdir()
        monitor = patch.object(exact_worker, 'peak_memory_mb', return_value=32.0)
        monitor.start()
        self.addCleanup(monitor.stop)

    def read(self, name='status.json'):
        return json.loads((self.runtime / name).read_text(encoding='utf-8'))

    def run_fake(self, engine, fake_solver, **settings):
        return exact_worker.run_task(self.home, engine=engine, seconds=2,
                                     solver_factory=lambda _: fake_solver, **settings)

    def test_adaptive_and_gpu_matching_options_reach_worker(self):
        seen = {}
        def fake(**kwargs):
            seen.update(kwargs)
            return {'outcome': 'timeout', 'board': None}
        self.assertEqual(self.run_fake('dfs', fake, pruning='adaptive'), 0)
        self.assertTrue(seen['options']['adaptive'])
        self.assertTrue(seen['options']['probe'])
        seen.clear()
        with patch('smart_repair.cached_parents', return_value=[]):
            self.assertEqual(self.run_fake('gpu-matching', fake, backend='cuda', gpu_repair_batch=7), 0)
        self.assertEqual(seen['gpu_backend'], 'cuda')
        self.assertEqual(seen['gpu_batch_size'], 7)
        self.assertFalse(seen['gpu_allow_cpu_fallback'])
        settings = dashboard_server.validate_settings(dict(method='exact', exact_engine='gpu-matching',
                                                           backend='opencl', gpu_repair_batch=7))
        command = dashboard_server.worker_command(self.home, 8765, settings)
        self.assertEqual(command[command.index('--gpu-repair-batch')+1], '7')
        self.assertEqual(command[command.index('--backend')+1], 'opencl')
        for bad in (0, 65, True, 2.5, 'abc'):
            with self.assertRaises(ValueError):
                dashboard_server.validate_settings(dict(gpu_repair_batch=bad))

    def test_gpu_matching_ui_has_separate_batch_and_labels(self):
        helper = library_tests.LibraryDashboardTests('test_boot_shows_cache_without_starting_archive_or_search')
        helper.server = SimpleNamespace(resources=Path(exact_worker.__file__).parent)
        rendered = helper.render_status(dict(state='stopped', method='exact', can_start=True,
                      exact_engine='gpu-matching', settings={'method':'exact', 'exact_engine':'gpu-matching',
                      'gpu_repair_batch':7}, cost_device='fixture GPU', cost_backend='cuda'))
        self.assertFalse(rendered['gpuSettings']['hidden'])
        self.assertFalse(rendered['gpuRepairSettings']['hidden'])
        self.assertTrue(rendered['gpuReplicaSettings']['hidden'])
        self.assertIn('CPU exact assignment', rendered['gpu']['text'])
        self.assertIn('fixture GPU', rendered['gpu']['text'])
        self.assertEqual(rendered['proposedLabel']['text'], 'Completed neighborhoods')

    def test_full_profile_reaches_all_optional_rules(self):
        seen={}
        def fake(**kwargs):
            seen.update(kwargs)
            return {'outcome':'timeout','board':None,'branches':0,'conflicts':0}
        self.assertEqual(self.run_fake('dfs',fake,pruning='full'),0)
        self.assertTrue(seen['options']['probe'])
        self.assertEqual(seen['options']['extra_rules'],{})

    def test_new_engine_settings_and_cli_arguments(self):
        for engine in ('optimize', 'sat-optimize', 'core-repair', 'matching', 'gpu-matching', 'partitioned'):
            workers = 4 if engine in ('optimize', 'partitioned') else 1
            settings = dashboard_server.validate_settings(dict(method='exact', exact_engine=engine,
                                          workers=workers, pruning='strong', fresh_exact=True))
            command = dashboard_server.worker_command(self.home, 8765, settings)
            self.assertIn('--fresh-exact', command)
            self.assertEqual(command[command.index('--engine') + 1], engine)
            self.assertEqual(command[command.index('--pruning') + 1], 'strong')
            args = launcher.parser().parse_args(['run', '--method', 'exact', '--exact-engine', engine,
                                      '--workers', str(workers), '--pruning', 'strong', '--fresh-exact', '--state-dir', str(self.home)])
            forwarded = launcher.run_arguments(args)
            self.assertIn('--fresh-exact', forwarded)
            self.assertEqual(forwarded[forwarded.index('--pruning') + 1], 'strong')
            exact_worker._settings(engine, 2, 17, workers, 8765, pruning='strong')
        for engine in ('core-repair', 'matching', 'gpu-matching', 'sat-optimize'):
            with self.assertRaises(ValueError):
                dashboard_server.validate_settings(dict(method='exact', exact_engine=engine, workers=2))
            with self.assertRaises(ValueError):
                exact_worker._settings(engine, 2, 17, 2, 8765)

    def test_profiles_are_forwarded_to_dfs_and_checkpoint_mismatch_is_preserved(self):
        seen = {}
        def fake(**kwargs):
            seen.update(kwargs)
            return {'outcome': 'timeout', 'board': None}
        self.assertEqual(self.run_fake('dfs', fake, pruning='strong'), 0)
        self.assertEqual(seen['options'], dict(gac=True, independent_completion=True,
                                            dual_branch=True, edge_branch=True, extra_rules=None))
        self.assertEqual(self.read('exact-session.json')['settings']['pruning'], 'strong')
        self.assertEqual(self.run_fake('dfs', fake, pruning='baseline'), 0)
        self.assertEqual(seen['options'], {})

    def test_repair_callbacks_validate_and_persist_only_strict_improvements(self):
        # Test-only score witness exercises persistence, not a claimed 467 find.
        candidate = list(self.bundle.record_board)
        candidate[17], candidate[18] = candidate[18], candidate[17]
        def validation(board, bundle):
            if board == candidate:
                return {'valid': True, 'complete': False, 'score': 467}
            return validate_board(board, bundle)
        def fake(**kwargs):
            self.assertEqual(kwargs['reference'], list(self.bundle.record_board))
            kwargs['on_improvement']({'board': candidate, 'score': 467})
            kwargs['on_improvement']({'board': candidate, 'score': 467})
            kwargs['on_improvement']({'board': list(self.bundle.record_board), 'score': 466})
            return {'outcome': 'improved', 'board': candidate, 'score': 467}
        with patch.object(exact_worker, 'validate_board', side_effect=validation):
            self.assertEqual(self.run_fake('optimize', fake, workers=4), 0)
        record = self.read('best.json')
        self.assertEqual(record['board'], candidate)
        self.assertEqual(record['score'], 467)
        self.assertFalse(record['complete'])
        self.assertFalse(record['automatic_upload'])
        self.assertIsNone(record['library_known'])
        self.assertEqual(len(list((self.home / 'results').glob('*.json'))), 1)
        self.assertEqual(self.read()['state'], 'finished_bounded_run')
        self.assertNotEqual(self.read()['state'], 'solved')

    def test_invalid_callback_score_cannot_replace_reference(self):
        before = dict(board=list(self.bundle.record_board), score=466, sentinel='preserve')
        (self.runtime / 'best.json').write_text(json.dumps(before))
        def fake(**kwargs):
            kwargs['on_improvement']({'board': list(self.bundle.record_board), 'score': 480})
            raise AssertionError('Invalid callback was not rejected')
        self.assertEqual(self.run_fake('core-repair', fake), 1)
        self.assertEqual(self.read('best.json'), before)
        self.assertEqual(self.read()['state'], 'error')

    def test_cached_parent_improvement_is_marked_known_and_database_is_read_only(self):
        candidate = list(self.bundle.record_board)
        candidate[17], candidate[18] = candidate[18], candidate[17]
        database = self.home / 'library' / 'library.sqlite3'
        database.parent.mkdir()
        blob = struct.pack('<256H', *candidate)
        digest = hashlib.sha256(blob).hexdigest()
        with sqlite3.connect(database) as connection:
            connection.execute('CREATE TABLE boards(public_sha TEXT,score INTEGER,local_sha TEXT,placement BLOB)')
            connection.execute('INSERT INTO boards VALUES(?,?,?,?)', ('fixture', 467, digest, blob))
        connection.close()
        before = database.read_bytes()
        def validation(board, bundle):
            return {'valid': True, 'complete': False, 'score': 467} if board == candidate else validate_board(board, bundle)
        def fake(**kwargs):
            self.assertEqual(kwargs['parents'], [candidate])
            kwargs['on_improvement']({'board': candidate, 'score': 467})
            kwargs['on_progress']({'phase': 'connected_region_repair', 'neighborhoods': 12, 'recombinations': 3,
                                   'diversification_attempts': 7, 'diversifications': 6, 'tabu_rejections': 2,
                                   'region_calls': 4, 'region_improvements': 1, 'region_seconds': 0.75, 'parent_count': 2})
            return {'outcome': 'timeout', 'board': candidate, 'score': 467, 'neighborhoods': 12, 'recombinations': 3}
        with patch.object(exact_worker, 'validate_board', side_effect=validation), patch('smart_repair.cached_parents', return_value=[candidate]) as parents:
            self.assertEqual(self.run_fake('matching', fake), 0)
        self.assertEqual(parents.call_args.kwargs['limit'], 32)
        self.assertGreaterEqual(parents.call_args.kwargs['max_seconds'], 0)
        self.assertLessEqual(parents.call_args.kwargs['max_seconds'], 2)
        self.assertEqual(parents.call_args.kwargs['seed'], exact_worker.DEFAULT_SEED)
        self.assertTrue(callable(parents.call_args.kwargs['should_stop']))
        self.assertTrue(self.read('best.json')['library_known'])
        self.assertEqual(self.read('best.json')['source'], 'existing local library arrangement')
        self.assertEqual(database.read_bytes(), before)
        self.assertIsNone(self.read()['branches'])
        self.assertEqual(self.read()['neighborhoods'], 12)
        self.assertEqual(self.read()['recombinations'], 3)
        for key, value in dict(diversification_attempts=7, diversifications=6, tabu_rejections=2,
                               region_calls=4, region_improvements=1, region_seconds=0.75, parent_count=2).items():
            self.assertEqual(self.read()[key], value)

    def test_target_infeasible_is_never_global_impossible(self):
        for engine in ('optimize', 'sat-optimize', 'core-repair'):
            def fake(**kwargs):
                return {'outcome': 'infeasible_for_target', 'board': list(self.bundle.record_board), 'score': 466,
                        'proof_scope': 'fixture-selected-target'}
            with self.subTest(engine=engine):
                self.assertEqual(self.run_fake(engine, fake), 0)
                status = self.read()
                self.assertEqual(status['state'], 'finished_bounded_run')
                self.assertFalse(status['coverage_verified'])
                self.assertEqual(status['proof_scope'], 'fixture-selected-target')
                self.assertIn('not a proof', status['exact_conclusion'])

    def test_partition_queue_resume_and_explicit_archive(self):
        queue = self.runtime / 'exact-partitioned-queue.json'
        queue.write_text('original frontier')
        def incompatible(**kwargs):
            self.assertEqual(kwargs['queue_path'], queue.resolve())
            raise ValueError('Fixture profile mismatch')
        self.assertEqual(self.run_fake('partitioned', incompatible, workers=4), 1)
        self.assertEqual(queue.read_text(), 'original frontier')
        self.assertIn('--fresh-exact', self.read()['error'])
        def fresh(**kwargs):
            self.assertFalse(queue.exists())
            self.assertTrue(kwargs['options']['gac'])
            return {'outcome': 'infeasible', 'board': None, 'coverage_verified': False,
                    'proof_scope': 'remaining_checkpoint_frontier'}
        self.assertEqual(self.run_fake('partitioned', fresh, workers=2, pruning='strong', fresh_exact=True), 0)
        archives = list(self.runtime.glob('exact-partitioned-queue.archived-*.json'))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_text(), 'original frontier')
        self.assertEqual(self.read()['state'], 'checkpoint_frontier_exhausted')

    def test_restart_flag_is_one_shot_in_dashboard_preferences(self):
        calls = []
        class Worker:
            pid = 12345
            def poll(self): return None
        def spawn(command, **kwargs):
            calls.append(command)
            return Worker()
        with dashboard_server.create_server(self.home, 0, worker_factory=spawn, library_factory=library_tests.FakeArchiveManager) as server:
            (self.runtime / 'status.json').write_text(json.dumps(dict(state='stopped', cost_backend='stale-backend',
                        cost_device='stale-device', cost_pairs=999, tickets_total=999, target_score=480,
                        diversification_attempts=999, diversifications=999, tabu_rejections=999, region_calls=999,
                        region_improvements=999, region_seconds=999, parent_count=999)))
            status, accepted = server.start_worker(dict(method='exact', exact_engine='partitioned', workers=2,
                                                       pruning='strong', fresh_exact=True))
            self.assertTrue(accepted)
            self.assertIn('--fresh-exact', calls[0])
            self.assertFalse(server.settings['fresh_exact'])
            self.assertFalse(self.read('dashboard-settings.json')['fresh_exact'])
            self.assertEqual(status['pruning'], 'strong')
            for stale_key in ('cost_backend', 'cost_device', 'cost_pairs', 'tickets_total', 'target_score'):
                self.assertNotIn(stale_key, status)
            for counter in ('diversification_attempts', 'diversifications', 'tabu_rejections', 'region_calls', 'region_improvements', 'region_seconds', 'parent_count'):
                self.assertEqual(status[counter], 0)

    def test_mock_ui_distinguishes_repair_metrics_and_native_counters(self):
        helper = library_tests.LibraryDashboardTests('test_boot_shows_cache_without_starting_archive_or_search')
        helper.server = SimpleNamespace(resources=Path(exact_worker.__file__).parent)
        base = dict(state='running', method='exact', can_start=False, workers=1, best_score=467,
                    exact_engine='matching', neighborhoods=14, recombinations=5, branches=None,
                    library_known=True, exact_phase='connected_region_repair', parent_count=17,
                    diversification_attempts=8, diversifications=6, tabu_rejections=2,
                    region_calls=4, region_improvements=1, region_seconds=0.75)
        rendered = helper.render_status(base)
        self.assertEqual(rendered['proposed']['text'], '14')
        self.assertEqual(rendered['accepted']['text'], '5')
        self.assertEqual(rendered['repairParents']['text'], '17')
        self.assertEqual(rendered['diversifications']['text'], '6')
        self.assertEqual(rendered['regionSearches']['text'], '4 / 1')
        self.assertEqual(rendered['exactPhase']['text'], 'Repairing a connected region')
        self.assertIn('saved best remains protected', rendered['activityNote']['text'])
        self.assertEqual(rendered['proposedLabel']['text'], 'Completed neighborhoods')
        self.assertEqual(rendered['recordLabel']['text'], 'Existing cached board')
        self.assertIn('not exhaustive search branches', rendered['activityNote']['text'])
        for engine in ('optimize', 'sat-optimize', 'core-repair'):
            rendered = helper.render_status(dict(base, exact_engine=engine, exact_counters_available=False))
            self.assertEqual(rendered['proposed']['text'], 'Not yet reported')
        rendered = helper.render_status(dict(base, state='stopped', can_start=True, exact_engine='partitioned', workers=4))
        self.assertFalse(rendered['workerCount']['disabled'])
        self.assertFalse(rendered['pruningSettings']['hidden'])
        self.assertFalse(rendered['freshSettings']['hidden'])


    def repair_checkpoint_solver(self, capture):
        from repair_checkpoint import CheckpointCodec, neighborhood_key
        from smart_repair import checkpoint_settings
        from test_repair_checkpoint import initial_state
        def fake(problem, reference, seed, checkpoint=None, on_checkpoint=None, **kwargs):
            options = checkpoint_settings(seed=seed, gpu_backend=kwargs.get('gpu_backend'),
                                          gpu_batch_size=kwargs.get('gpu_batch_size', 16))
            codec = CheckpointCodec(problem, options)
            state = codec.decode(checkpoint) if checkpoint is not None else initial_state(problem, reference, seed)
            capture.append((checkpoint is not None, list(reference), state['completed']))
            state['completed'] += 1; state['rounds'] += 1; state['batches'] += 1
            state['elapsed_total'] += 0.1
            key = neighborhood_key(reference, [0], state['completed'])
            state['seen'].append(key)
            payload = codec.encode(state)
            on_checkpoint(payload)
            return {'outcome':'timeout', 'board':state['best'], 'score':state['best_score'],
                    'checkpoint':payload, 'parent_count':len(state['bank'])}
        return fake

    def test_matching_resume_retains_population_and_skips_library_scan(self):
        for engine in ('matching', 'gpu-matching'):
            captured = []
            fake = self.repair_checkpoint_solver(captured)
            with patch('smart_repair.cached_parents', return_value=[]) as parents:
                self.assertEqual(self.run_fake(engine, fake), 0)
                self.assertEqual(parents.call_count, 1)
            path = self.runtime / f'exact-{engine}-checkpoint.json'
            saved = json.loads(path.read_text())
            self.assertEqual(saved['state']['completed'], 1)
            with patch('smart_repair.cached_parents', side_effect=AssertionError('Resume reread library')):
                self.assertEqual(self.run_fake(engine, fake), 0)
            self.assertEqual(captured[-1][0], True)
            self.assertEqual(captured[-1][2], 1)
            self.assertEqual(json.loads(path.read_text())['state']['completed'], 2)
            self.assertTrue(self.read()['resumed'])
            self.assertTrue(self.read('exact-session.json')['repair_search_state_resumed'])
            self.assertFalse(self.read('exact-session.json')['native_search_state_resumed'])
            self.assertEqual(self.read()['best_score'], 466)

    def test_matching_incompatible_checkpoint_preserved_until_explicit_restart(self):
        captured = []; fake = self.repair_checkpoint_solver(captured)
        with patch('smart_repair.cached_parents', return_value=[]):
            self.assertEqual(self.run_fake('matching', fake), 0)
        path = self.runtime / 'exact-matching-checkpoint.json'
        original = path.read_bytes()
        self.assertEqual(self.run_fake('matching', fake, seed=19), 1)
        self.assertEqual(path.read_bytes(), original)
        self.assertIn('--fresh-exact', self.read()['error'])
        with patch('smart_repair.cached_parents', return_value=[]):
            self.assertEqual(self.run_fake('matching', fake, seed=19, fresh_exact=True), 0)
        backups = list(self.runtime.glob('exact-matching-checkpoint.archived-*.json'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        self.assertFalse(captured[-1][0])
        self.assertEqual(json.loads(path.read_text())['settings']['seed'], 19)

    def test_matching_corruption_and_invalid_callback_cannot_replace_saved_state(self):
        captured = []; fake = self.repair_checkpoint_solver(captured)
        with patch('smart_repair.cached_parents', return_value=[]):
            self.assertEqual(self.run_fake('matching', fake), 0)
        path = self.runtime / 'exact-matching-checkpoint.json'
        original = path.read_bytes(); best = (self.runtime / 'best.json').read_bytes()
        def bad(checkpoint=None, on_checkpoint=None, **kwargs):
            checkpoint['state']['best_score'] = 480
            on_checkpoint(checkpoint)
            self.fail('Invalid checkpoint callback was accepted')
        self.assertEqual(self.run_fake('matching', bad), 1)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual((self.runtime / 'best.json').read_bytes(), best)
        path.write_text('{"broken":')
        self.assertEqual(self.run_fake('matching', fake), 1)
        self.assertEqual(path.read_text(), '{"broken":')
        self.assertEqual((self.runtime / 'best.json').read_bytes(), best)
        self.assertIn('--fresh-exact', self.read()['error'])

    def test_matching_checkpoint_cannot_lower_existing_saved_best(self):
        from exact_model import from_bundle
        from repair_checkpoint import CheckpointCodec
        from repair_diversify import Diversifier
        from smart_repair import checkpoint_settings
        from test_repair_checkpoint import initial_state
        problem = from_bundle(self.bundle)
        lower = Diversifier(problem).perturb(self.bundle.record_board, seed=17, moves=2)['board']
        checked = validate_board(lower, self.bundle)
        self.assertTrue(checked['valid']); self.assertLess(checked['score'], 466)
        codec = CheckpointCodec(problem, checkpoint_settings(seed=exact_worker.DEFAULT_SEED))
        payload = codec.encode(initial_state(problem, lower, exact_worker.DEFAULT_SEED))
        path = self.runtime / 'exact-matching-checkpoint.json'
        path.write_text(json.dumps(payload))
        def fake(reference, checkpoint=None, **kwargs):
            self.assertEqual(validate_board(reference, self.bundle)['score'], 466)
            state = codec.decode(checkpoint)
            return {'outcome':'timeout', 'board':state['best'], 'score':state['best_score']}
        self.assertEqual(self.run_fake('matching', fake), 0)
        self.assertEqual(self.read()['best_score'], 466)
        self.assertEqual(self.read('best.json')['score'], 466)

    def test_matching_ui_exposes_resume_and_explicit_restart(self):
        helper = library_tests.LibraryDashboardTests('test_boot_shows_cache_without_starting_archive_or_search')
        helper.server = SimpleNamespace(resources=Path(exact_worker.__file__).parent)
        for engine in ('matching', 'gpu-matching'):
            rendered = helper.render_status(dict(state='stopped', method='exact', can_start=True,
                          exact_engine=engine, settings={'method':'exact', 'exact_engine':engine}))
            self.assertFalse(rendered['freshSettings']['hidden'])
            self.assertEqual(rendered['freshLabel']['text'], 'Restart repair population')
            self.assertIn('resumes compatible working boards', rendered['resumeHint']['text'])
        rendered = helper.render_status(dict(state='stopped', method='exact', can_start=True,
                      exact_engine='cp-sat', settings={'method':'exact', 'exact_engine':'cp-sat'}))
        self.assertTrue(rendered['freshSettings']['hidden'])



if __name__ == '__main__':
    unittest.main()
