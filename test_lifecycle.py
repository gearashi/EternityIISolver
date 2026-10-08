"""Portable lifecycle checks using real OS locks and an injected, CPU-only engine.

All writable state lives in temporary folders. These tests never run GPU search
or contact the public board library.
"""
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import uuid

import numpy as np

import app_paths
from process_control import RunLock, process_alive, read_status


ROOT = Path(__file__).resolve().parent


class TemporaryStateTest(unittest.TestCase):
    def setUp(self):
        self.temp_base = Path(tempfile.gettempdir()).resolve()
        self.home = self.temp_base / ('eternity-lifecycle-' + uuid.uuid4().hex)
        # Inherit the existing temporary-folder ACL on managed Windows hosts.
        self.home.mkdir()
        self.addCleanup(self.clean_state)

    def clean_state(self):
        if self.home.resolve().parent != self.temp_base or not self.home.name.startswith('eternity-lifecycle-'):
            raise RuntimeError('Refusing to clean an unexpected fixture directory')
        shutil.rmtree(self.home)

    def child(self, code, *arguments):
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        return subprocess.run(
            [sys.executable, '-B', '-c', code, *map(str, arguments)],
            cwd=self.home, env=env, capture_output=True, text=True, timeout=15,
        )


class ProcessLifecycleTests(TemporaryStateTest):
    def test_lock_contention_and_release_across_processes(self):
        lock_path = self.home / 'runtime' / 'run.lock'
        probe = '''
import sys
from process_control import RunLock
try:
    lock = RunLock(sys.argv[1]).acquire()
except RuntimeError:
    raise SystemExit(23)
lock.close()
'''
        with RunLock(lock_path):
            blocked = self.child(probe, lock_path)
            self.assertEqual(blocked.returncode, 23, blocked.stderr)
        self.assertTrue(lock_path.exists(), 'Lock inode must persist after release')
        released = self.child(probe, lock_path)
        self.assertEqual(released.returncode, 0, released.stderr)

    def test_process_exit_releases_lock_without_explicit_close(self):
        lock_path = self.home / 'run.lock'
        crashed = self.child('''
import os, sys
from process_control import RunLock
lock = RunLock(sys.argv[1]).acquire()
os._exit(7)
''', lock_path)
        self.assertEqual(crashed.returncode, 7, crashed.stderr)
        with RunLock(lock_path):
            self.assertTrue(lock_path.exists())

    def test_liveness_rejects_invalid_pids_and_finds_current_process(self):
        for pid in (None, True, False, '1', 1.5, 0, -1, 2**100):
            with self.subTest(pid=pid):
                self.assertFalse(process_alive(pid))
        self.assertTrue(process_alive(os.getpid()))

    def test_missing_status_and_exited_process_status(self):
        self.assertEqual(read_status(self.home), {'state': 'not_started', 'process_alive': False})
        child = self.child('import os; print(os.getpid())')
        self.assertEqual(child.returncode, 0, child.stderr)
        runtime = self.home / 'runtime'
        runtime.mkdir()
        (runtime / 'status.json').write_text(json.dumps({
            'state': 'finished_bounded_run', 'pid': int(child.stdout), 'best_score': 466,
        }), encoding='utf-8')
        saved = read_status(self.home)
        self.assertFalse(saved['process_alive'])
        self.assertEqual(saved['best_score'], 466)


class StatePathTests(TemporaryStateTest):
    def test_explicit_path_precedes_environment_override(self):
        with patch.dict(os.environ, {'ETERNITY_SOLVER_HOME': str(self.home / 'environment')}):
            self.assertEqual(app_paths.state_root(), (self.home / 'environment').resolve())
            self.assertEqual(app_paths.state_root(self.home / 'explicit'), (self.home / 'explicit').resolve())

    def test_platform_defaults_use_user_state_directories(self):
        user_home = self.home / 'user'
        with patch.dict(os.environ, {}, clear=True), patch.object(app_paths.Path, 'home', return_value=user_home):
            for platform, expected in (
                ('win32', user_home / 'AppData' / 'Local' / 'EternityIISolver'),
                ('darwin', user_home / 'Library' / 'Application Support' / 'EternityIISolver'),
                ('linux', user_home / '.local' / 'share' / 'eternity-ii-solver'),
            ):
                with self.subTest(platform=platform), patch.object(app_paths.sys, 'platform', platform):
                    self.assertEqual(app_paths.state_root(), expected)
            with patch.object(app_paths.sys, 'platform', 'linux'), patch.dict(os.environ, {'XDG_DATA_HOME': str(self.home / 'xdg')}):
                self.assertEqual(app_paths.state_root(), self.home / 'xdg' / 'eternity-ii-solver')
            with patch.object(app_paths.sys, 'platform', 'win32'), patch.dict(os.environ, {'LOCALAPPDATA': str(self.home / 'local')}):
                self.assertEqual(app_paths.state_root(), self.home / 'local' / 'EternityIISolver')
        self.assertFalse(user_home.exists(), 'Resolving paths must not create state directories')


class SourceControlTests(TemporaryStateTest):
    CONTROL = '''
import importlib.abc, sys
forbidden = {'cupy', 'pyopencl', 'gpu_engine', 'gpu_backends'}
class NoGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise AssertionError('Control command imported GPU package: ' + fullname)
sys.meta_path.insert(0, NoGPUImports())
import launcher
result = launcher.main(sys.argv[1:])
assert not forbidden.intersection(sys.modules)
raise SystemExit(result)
'''

    def test_validate_runs_without_gpu_packages(self):
        result = self.child(self.CONTROL, 'validate')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report['valid'])
        self.assertEqual(report['score'], 466)

    def test_status_runs_without_gpu_packages_or_creating_state(self):
        state = self.home / 'not-created'
        result = self.child(self.CONTROL, 'status', '--json', '--state-dir', state)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'state': 'not_started', 'process_alive': False})
        self.assertFalse(state.exists())


class DeviceArray:
    """Only the transfer/indexing surface used by the host lifecycle."""
    def __init__(self, value):
        self.value = np.asarray(value)

    def get(self):
        return self.value.copy()

    def __getitem__(self, key):
        return DeviceArray(self.value[key])


class FakeEngine:
    backend = 'fake-test-engine'
    device = 'CPU fixture; no GPU work'

    def __init__(self, bundle, n, seed, *, on_step=None, **unused):
        self.bundle, self.n = bundle, n
        self.host_rng = np.random.default_rng(seed)
        self.counters = DeviceArray(np.zeros(n * 2, dtype=np.uint64))
        self.bestscores = DeviceArray(np.full(n, 466, dtype=np.int32))
        self.bestboards = DeviceArray(np.tile(np.asarray(bundle.record_board, dtype=np.int16)[:, None], (1, n)))
        self.on_step = on_step
        self.step_calls = 0
        self.checkpoint_paths = []
        self.resume_paths = []
        self.verified = []

    def initialize(self, seeds):
        self.seeds = np.asarray(seeds, dtype=np.int16)

    def resume(self, path):
        self.resume_paths.append(Path(path))
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['boards'], self.bestboards.get())
        return True

    def verify_device_scores(self, indices):
        self.verified.extend(int(i) for i in indices)
        return len(list(indices))

    def cool(self, elapsed):
        pass

    def step(self, steps):
        self.step_calls += 1
        self.counters.value[:self.n] += steps
        if self.on_step:
            self.on_step(self)
        return 40.0

    def strategy_status(self):
        return []

    def checkpoint(self, path):
        path = Path(path)
        self.checkpoint_paths.append(path)
        np.savez(path, boards=self.bestboards.get())


class SolverLifecycleTests(TemporaryStateTest):
    def setUp(self):
        super().setUp()
        import solver
        self.solver = solver
        self.instances = []
        self.on_step = None

    def factory(self, bundle, n, seed, **options):
        engine = FakeEngine(bundle, n, seed, on_step=self.on_step, **options)
        self.instances.append(engine)
        return engine

    def unused_port(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            return sock.getsockname()[1]

    def run_solver(self, *extra, port=None):
        args = ['--state-dir', str(self.home), '--no-library', '--replicas', '32',
                '--port', str(port or self.unused_port()), *extra]
        # A library or GPU import here would violate the test's isolation.
        with patch.dict(sys.modules, {'library_cache': None, 'gpu_engine': None}), redirect_stdout(io.StringIO()):
            return self.solver.main(args, engine_factory=self.factory)

    def saved_status(self):
        return json.loads((self.home / 'runtime' / 'status.json').read_text(encoding='utf-8'))

    def assert_lock_released(self):
        with RunLock(self.home / 'runtime' / 'run.lock'):
            pass

    def test_zero_duration_checkpoints_and_resumes_without_moves(self):
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        self.assertEqual(self.run_solver('--seconds', '0'), 0, self.saved_status())
        first = self.instances[-1]
        self.assertEqual(first.step_calls, 0)
        self.assertEqual(self.saved_status()['state'], 'finished_bounded_run')
        self.assertEqual(len(first.checkpoint_paths), 1)
        self.assertTrue(first.checkpoint_paths[0].is_file())
        self.assertGreaterEqual(len(first.verified), 32)
        self.assert_lock_released()
        self.assertEqual(self.run_solver('--seconds', '0'), 0, self.saved_status())
        self.assertEqual(self.instances[-1].resume_paths, first.checkpoint_paths)
        self.assertTrue(self.saved_status()['resumed'])
        self.assertEqual({sig: signal.getsignal(sig) for sig in previous_handlers}, previous_handlers)

    def test_dashboard_stop_saves_checkpoint_and_closes_server(self):
        port = self.unused_port()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        base = f'http://127.0.0.1:{port}'

        def request_stop(engine):
            with opener.open(base + '/status', timeout=3) as response:
                live = json.load(response)
            self.assertEqual(live['state'], 'running')
            self.assertEqual(live['pid'], os.getpid())
            with opener.open(base + '/board', timeout=3) as response:
                self.assertEqual(json.load(response)['board'], list(engine.bundle.record_board))
            foreign = urllib.request.Request(base + '/stop', data=b'', headers={'Origin': 'https://foreign.invalid'})
            with self.assertRaises(urllib.error.HTTPError) as rejected:
                opener.open(foreign, timeout=3)
            self.assertEqual(rejected.exception.code, 403)
            self.assertFalse((self.home / 'runtime' / 'STOP').exists())
            local = urllib.request.Request(base + '/stop', data=b'', headers={'Origin': base})
            with opener.open(local, timeout=3) as response:
                self.assertEqual(response.status, 202)

        self.on_step = request_stop
        self.assertEqual(self.run_solver('--seconds', '10', port=port), 0)
        engine = self.instances[-1]
        self.assertEqual(engine.step_calls, 1)
        self.assertEqual(self.saved_status()['state'], 'stopped')
        self.assertEqual(self.saved_status()['proposed_moves'], 32 * 16)
        self.assertEqual(len(engine.checkpoint_paths), 1)
        self.assert_lock_released()
        with socket.socket() as probe:
            probe.settimeout(1)
            self.assertNotEqual(probe.connect_ex(('127.0.0.1', port)), 0)

    def test_occupied_dashboard_port_fails_before_engine_creation(self):
        with socket.socket() as blocker:
            blocker.bind(('127.0.0.1', 0))
            blocker.listen()
            self.assertEqual(self.run_solver('--seconds', '0', port=blocker.getsockname()[1]), 1)
        self.assertEqual(self.instances, [])
        status = self.saved_status()
        self.assertEqual(status['state'], 'error')
        self.assertIn('Cannot open local dashboard port', status['error'])
        self.assertFalse((self.home / 'runtime' / 'checkpoint.npz').exists())
        self.assert_lock_released()

    def test_engine_failure_records_error_and_releases_lock(self):
        def fail_step(_):
            raise RuntimeError('deliberate fixture engine failure')

        self.on_step = fail_step
        self.assertEqual(self.run_solver('--seconds', '10'), 1)
        status = self.saved_status()
        self.assertEqual(status['state'], 'error')
        self.assertIn('deliberate fixture engine failure', status['error'])
        self.assert_lock_released()


class PersistentDashboardTests(TemporaryStateTest):
    def setUp(self):
        super().setUp()
        from dashboard_server import create_server
        self.spawned = []

        def launch(command, **options):
            class Worker:
                pid = 0x7FFFFFFE
                returncode = None

                def poll(self):
                    return self.returncode

            worker = Worker()
            self.spawned.append((command, options, worker))
            return worker

        self.server = create_server(self.home, port=0, worker_factory=launch)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.base = f'http://127.0.0.1:{self.server.port}'

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, endpoint, value=None, *, headers=None, raw=None):
        data = json.dumps(value).encode() if value is not None else raw
        request_headers = {'Content-Type': 'application/json', 'Origin': self.base} if data is not None else {}
        request_headers.update(headers or {})
        request = urllib.request.Request(self.base + endpoint, data=data, headers=request_headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=3) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_idle_dashboard_and_validated_board_fallback(self):
        code, status = self.request('/status')
        self.assertEqual(code, 200)
        self.assertEqual(status['state'], 'stopped')
        self.assertTrue(status['dashboard_available'])
        self.assertTrue(status['can_start'])
        self.assertFalse(status['can_stop'])
        self.assertEqual(status['replicas'], 4096)
        self.assertEqual(status['dashboard_pid'], os.getpid())
        self.assertEqual(self.spawned, [])
        (self.home / 'runtime' / 'best.json').write_text('{"board": [1, 1]}', encoding='utf-8')
        _, board = self.request('/board')
        self.assertEqual(board['board'], list(self.server.bundle.record_board))
        metadata = json.loads((self.home / 'runtime' / 'dashboard.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['state'], 'running')
        self.assertEqual(metadata['port'], self.server.port)

    def test_concurrent_start_is_single_worker_and_stop_keeps_dashboard_alive(self):
        settings = {'replicas': 512, 'backend': 'opencl', 'seed': 123, 'no_library': True, 'seconds': 0}
        with ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(lambda _: self.request('/start', settings), range(4)))
        self.assertEqual(sorted(code for code, _ in replies), [200, 200, 200, 202])
        self.assertEqual(len(self.spawned), 1)
        command, options, worker = self.spawned[0]
        self.assertIn('--no-dashboard', command)
        self.assertIn('--preserve-stop', command)
        self.assertIn('--no-library', command)
        self.assertEqual(command[command.index('--replicas') + 1], '512')
        self.assertEqual(command[command.index('--seconds') + 1], '0')
        self.assertEqual(options['env']['PYINSTALLER_RESET_ENVIRONMENT'], '1')
        if os.name == 'nt':
            self.assertTrue(options['creationflags'] & subprocess.CREATE_NO_WINDOW)
        else:
            self.assertTrue(options['start_new_session'])
        code, status = self.request('/stop', {})
        self.assertEqual(code, 202)
        self.assertEqual(status['state'], 'stopping')
        self.assertTrue((self.home / 'runtime' / 'STOP').is_file())
        worker.returncode = 0
        code, status = self.request('/status')
        self.assertEqual(code, 200)
        self.assertEqual(status['state'], 'stopped')
        self.assertTrue(status['can_start'])
        self.assertEqual(status['replicas'], 512)
        self.assertEqual(self.request('/start', settings)[0], 202)
        self.assertEqual(len(self.spawned), 2)

    def test_stop_during_start_survives_and_previous_error_is_cleared(self):
        runtime = self.home / 'runtime'
        (runtime / 'STOP').write_text('Old stopped run', encoding='utf-8')
        (runtime / 'status.json').write_text(json.dumps({
            'state': 'error', 'pid': 0, 'error': 'old error', 'traceback': 'old trace',
            'replicas': 32, 'backend': 'cuda',
        }), encoding='utf-8')
        self.assertEqual(self.request('/start', {'replicas': 256, 'backend': 'opencl'})[0], 202)
        self.assertFalse((runtime / 'STOP').exists())
        _, pending = self.request('/status')
        self.assertEqual(pending['state'], 'starting')
        self.assertEqual(pending['replicas'], 256)
        self.assertNotIn('error', pending)
        self.assertNotIn('traceback', pending)
        self.assertEqual(self.request('/stop', {})[0], 202)
        self.assertTrue((runtime / 'STOP').exists())
        self.assertEqual(self.request('/status')[1]['state'], 'stopping')
        self.assertTrue((runtime / 'STOP').exists())

    def test_fresh_worker_status_can_have_different_pid_from_launcher(self):
        self.assertEqual(self.request('/start', {'replicas': 128})[0], 202)
        (self.home / 'runtime' / 'status.json').write_text(json.dumps({
            'pid': os.getpid(), 'state': 'running', 'replicas': 128, 'best_score': 466,
        }), encoding='utf-8')
        _, status = self.request('/status')
        self.assertEqual(status['pid'], os.getpid())
        self.assertEqual(status['state'], 'running')
        self.assertFalse(status['can_start'])

    def test_rejects_foreign_requests_invalid_settings_and_oversized_bodies(self):
        self.assertEqual(self.request('/start', {}, headers={'Origin': 'https://foreign.invalid'})[0], 403)
        self.assertEqual(self.request('/start', {}, headers={'Host': 'foreign.invalid'})[0], 403)
        self.assertEqual(self.request('/start', {}, headers={'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.request('/start', raw=b' ' * 4097)[0], 413)
        self.assertEqual(self.request('/start', raw=b'{broken')[0], 400)
        for value in ({'replicas': True}, {'replicas': 31}, {'replicas': 32769},
                      {'backend': 'other'}, {'seed': -1}, {'seed': 2**32},
                      {'no_library': 1}, {'seconds': -1}, {'seconds': float('inf')},
                      {'seconds': True}, {'arbitrary': 'argument'}):
            with self.subTest(value=value):
                self.assertEqual(self.request('/start', value)[0], 400)
        self.assertEqual(self.spawned, [])

    def test_stale_worker_status_is_stopped_and_dashboard_has_own_lock(self):
        (self.home / 'runtime' / 'status.json').write_text(json.dumps({
            'pid': 0, 'state': 'running', 'replicas': 1024,
        }), encoding='utf-8')
        _, status = self.request('/status')
        self.assertEqual(status['state'], 'stopped')
        self.assertTrue(status['can_start'])
        from dashboard_server import create_server
        with self.assertRaises(RuntimeError):
            create_server(self.home, 0)
        # The worker uses an independent lock and can start under the supervisor.
        with RunLock(self.home / 'runtime' / 'run.lock'):
            pass


if __name__ == '__main__':
    unittest.main()
