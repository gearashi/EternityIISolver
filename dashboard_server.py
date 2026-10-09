"""Persistent local dashboard that starts and stops separate solver workers."""
import argparse
import json
import math
import re
import urllib.parse
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer

from app_paths import resource_root, state_root
from process_control import RunLock, read_status
from validator import load_bundle, validate_board


DEFAULT_SETTINGS = {'replicas': 4096, 'backend': 'auto', 'seed': 20261007,
                    'no_library': False, 'seconds': None, 'method': 'gpu', 'exact_engine': 'dfs', 'workers': 1}
MAX_BODY = 4096


def validate_settings(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT_SETTINGS):
        raise ValueError('Expected an object containing supported search settings')
    settings = dict(DEFAULT_SETTINGS, **value)
    if settings['method'] not in ('gpu', 'exact'):
        raise ValueError('Search method must be gpu or exact')
    if settings['exact_engine'] not in ('dfs', 'sat', 'cp-sat', 'hybrid'):
        raise ValueError('Exact engine must be dfs, sat, cp-sat or hybrid')
    if type(settings['workers']) is not int or not 1 <= settings['workers'] <= 4:
        raise ValueError('CPU workers must be an integer from 1 to 4')
    if settings['method'] == 'exact' and settings['exact_engine'] != 'cp-sat' and settings['workers'] != 1:
        raise ValueError('SAT and backtracking use one CPU worker')
    if type(settings['replicas']) is not int or not 32 <= settings['replicas'] <= 32768:
        raise ValueError('Parallel searches must be an integer from 32 to 32768')
    if settings['backend'] not in ('auto', 'cuda', 'opencl'):
        raise ValueError('Backend must be auto, cuda or opencl')
    if type(settings['seed']) is not int or not 0 <= settings['seed'] <= 0xFFFFFFFF:
        raise ValueError('Seed must be an integer from 0 to 4294967295')
    if type(settings['no_library']) is not bool:
        raise ValueError('no_library must be a boolean')
    seconds = settings['seconds']
    if seconds is not None and (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
        raise ValueError('Seconds must be finite and nonnegative, or null for an unbounded run')
    return settings


def worker_command(home, port, settings):
    command = [sys.executable]
    if not getattr(sys, 'frozen', False):
        command.append(str(Path(__file__).resolve().with_name('launcher.py')))
    if settings['method'] == 'exact':
        command.extend(['exact', '--preserve-stop', '--state-dir', str(home), '--port', str(port),
                        '--engine', settings['exact_engine'], '--workers', str(settings['workers']), '--seed', str(settings['seed'])])
        if settings['exact_engine'] == 'hybrid':
            command.extend(['--backend', settings['backend'], '--replicas', str(settings['replicas'])])
        if settings['seconds'] is not None:
            command.extend(['--seconds', str(settings['seconds'])])
        return command
    command.extend(['run', '--no-dashboard', '--preserve-stop', '--state-dir', str(home),
                    '--port', str(port), '--replicas', str(settings['replicas']),
                    '--backend', settings['backend'], '--seed', str(settings['seed'])])
    if settings['seconds'] is not None:
        command.extend(['--seconds', str(settings['seconds'])])
    if settings['no_library']:
        command.append('--no-library')
    return command


def atomic_json(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(pending, path)


def export_best_response(home):
    from manual_export import export_best
    home = Path(home)
    output = export_best(home)
    for key in ('json', 'layout', 'validation'):
        relative = Path(output[key + '_path']).relative_to(home / 'exports').as_posix()
        output[key + '_url'] = '/exports/' + urllib.parse.quote(relative, safe='/')
    return output


def resolve_exported_file(home, request_path):
    relative = urllib.parse.unquote(request_path[len('/exports/'):])
    if not re.fullmatch(r'[A-Za-z0-9_-]+/(?:board\.json|layout\.txt|validation\.json)', relative):
        raise ValueError('Invalid export filename')
    folder = (Path(home) / 'exports').resolve()
    path = (folder / relative).resolve(strict=True)
    if not path.is_relative_to(folder) or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Export file is not available')
    return path


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, home, port=8765, *, worker_factory=None):
        self.home = state_root(home)
        self.runtime = self.home / 'runtime'
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.resources = resource_root()
        self.bundle = load_bundle()
        self.worker_factory = worker_factory or subprocess.Popen
        self.worker = None
        self.worker_status_before = None
        self.mutex = threading.RLock()
        self.settings = dict(DEFAULT_SETTINGS)
        self.closed = False
        self.identity_published = False
        self.lock = RunLock(self.runtime / 'dashboard.lock').acquire()
        try:
            settings_file = self.runtime / 'dashboard-settings.json'
            if settings_file.exists():
                try:
                    self.settings = validate_settings(json.loads(settings_file.read_text(encoding='utf-8')))
                except (OSError, ValueError, TypeError, OverflowError):
                    pass
            super().__init__(('127.0.0.1', port), DashboardHandler)
            self.port = self.server_address[1]
            self.save_identity('running')
            self.identity_published = True
        except BaseException:
            if hasattr(self, 'socket'):
                self.socket.close()
            self.lock.close()
            raise

    def server_bind(self):
        # This server binds only the numeric loopback address. HTTPServer's
        # default getfqdn() lookup is unnecessary and can delay local startup.
        TCPServer.server_bind(self)
        self.server_name = 'localhost'
        self.server_port = self.server_address[1]

    def save_identity(self, state):
        atomic_json(self.runtime / 'dashboard.json', {'state': state, 'pid': os.getpid(), 'port': self.port})

    def status(self):
        with self.mutex:
            try:
                saved = read_status(self.home)
            except (OSError, ValueError, TypeError, AttributeError):
                saved = {'state': 'stopped', 'process_alive': False}
            active = saved.get('process_alive', False) and saved.get('state') in ('starting', 'running', 'stopping')
            if self.worker is not None:
                exit_code = self.worker.poll()
                fresh_status = self.status_revision() != self.worker_status_before
                if exit_code is None:
                    active = True
                    # Python launchers can have a different PID from the worker.
                    # Trust a newly written worker status, never stale PID equality.
                    if not fresh_status:
                        saved.update(state='starting', pid=self.worker.pid, process_alive=True)
                        saved.update({key: self.settings[key] for key in ('replicas', 'backend', 'seed', 'method', 'exact_engine', 'workers')})
                        saved.pop('error', None)
                        saved.pop('traceback', None)
                else:
                    if not fresh_status:
                        active = False
                        saved['process_alive'] = False
                    if exit_code and saved.get('state') != 'error' and not active:
                        saved.update(state='error', error=f'Solver exited with code {exit_code}; inspect the local logs.')
                    self.worker = None
            if not active and saved.get('state') in ('not_started', 'starting', 'running', 'stopping'):
                saved.update(state='stopped', process_alive=False)
            if active and (self.runtime / 'STOP').exists():
                saved['state'] = 'stopping'
            for key in ('replicas', 'backend', 'seed'):
                saved.setdefault(key, self.settings[key])
            saved.setdefault('best_score', 466)
            saved.setdefault('source_best_score', 466)
            saved.update(dashboard_available=True, dashboard_pid=os.getpid(), port=self.port,
                         can_start=not active, can_stop=bool(active), state_dir=str(self.home), settings=dict(self.settings))
            legacy_active = active and self.worker is None and saved.get('external_network_enabled') is not False
            saved.setdefault('method', 'gpu')
            saved.setdefault('search_method', 'gpu-board-repair')
            saved.setdefault('counter_semantics', 'local move attempts; not DFS nodes or BOINC credit')
            library = dict(saved.get('library') or {})
            if legacy_active:
                saved['external_network_enabled'] = None
                saved['legacy_worker_warning'] = 'An older worker is still running. Its network behavior is unverified; stop it before using the offline update.'
                library.update(mode='legacy-unverified', network_enabled=None, online_freshness_verified=False)
            else:
                saved['external_network_enabled'] = False
                saved.pop('legacy_worker_warning', None)
                if library.get('network_requests_this_session'):
                    library['previous_run_network_requests'] = library['network_requests_this_session']
                library.update(mode='offline', network_enabled=False, running=False,
                               online_freshness_verified=False, network_requests_this_session=0,
                               detail_downloads_this_session=0)
            saved['library'] = library
            return saved

    def status_revision(self):
        try:
            stat = (self.runtime / 'status.json').stat()
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    def start_worker(self, value):
        settings = validate_settings(value)
        with self.mutex:
            status = self.status()
            if not status['can_start']:
                return status, False
            command = worker_command(self.home, self.port, settings)
            self.worker_status_before = self.status_revision()
            # Worker --preserve-stop leaves a newer Stop request intact while
            # GPU initialization is in progress.
            (self.runtime / 'STOP').unlink(missing_ok=True)
            options = {'stdin': subprocess.DEVNULL, 'close_fds': True,
                       'env': dict(os.environ, PYINSTALLER_RESET_ENVIRONMENT='1')}
            if os.name == 'nt':
                options['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                options['start_new_session'] = True
            with (self.runtime / 'solver.stdout.log').open('ab') as stdout, (self.runtime / 'solver.stderr.log').open('ab') as stderr:
                self.worker = self.worker_factory(command, stdout=stdout, stderr=stderr, **options)
            self.settings = settings
            atomic_json(self.runtime / 'dashboard-settings.json', settings)
            status = self.status()
            status.update(replicas=settings['replicas'], backend=settings['backend'], seed=settings['seed'],
                          method=settings['method'], exact_engine=settings['exact_engine'], workers=settings['workers'])
            return status, True

    def stop_worker(self):
        with self.mutex:
            (self.runtime / 'STOP').write_text('Stop requested from persistent dashboard\n', encoding='utf-8')
            status = self.status()
            if status['can_stop']:
                status['state'] = 'stopping'
            return status

    def export_best(self):
        with self.mutex:
            return export_best_response(self.home)

    def exported_file(self, request_path):
        return resolve_exported_file(self.home, request_path)

    def board(self):
        board = self.bundle.record_board
        try:
            candidate = json.loads((self.runtime / 'best.json').read_text(encoding='utf-8'))['board']
            if validate_board(candidate, self.bundle)['valid']:
                board = candidate
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return {'board': list(board), 'faces': self.bundle.oriented_edges}

    def server_close(self):
        if self.closed:
            return
        self.closed = True
        try:
            super().server_close()
        finally:
            try:
                # TCPServer also calls this method if bind/listen fails during
                # construction, before a port or running identity exists.
                if self.identity_published:
                    self.save_identity('stopped')
            finally:
                self.lock.close()


class DashboardHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *_):
        pass

    def send_payload(self, value, status=200, content_type='application/json; charset=utf-8', *, filename=None):
        payload = value if isinstance(value, bytes) else json.dumps(value).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        if filename is not None:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; connect-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(payload)

    def local_request(self):
        hosts = {f'127.0.0.1:{self.server.port}', f'localhost:{self.server.port}'}
        if self.headers.get('Host') not in hosts:
            self.send_payload({'error': 'Local dashboard Host required'}, 403)
            return False
        origin = self.headers.get('Origin')
        if origin is not None and origin != 'http://' + self.headers.get('Host'):
            self.send_payload({'error': 'Same-origin dashboard request required'}, 403)
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        try:
            if self.path in ('/', '/Dashboard.html'):
                self.send_payload((self.server.resources / 'Dashboard.html').read_bytes(), content_type='text/html; charset=utf-8')
            elif self.path == '/status':
                self.send_payload(self.server.status())
            elif self.path == '/board':
                self.send_payload(self.server.board())
            elif self.path.startswith('/exports/'):
                path = self.server.exported_file(self.path)
                mime = 'application/json; charset=utf-8' if path.suffix == '.json' else 'text/plain; charset=utf-8'
                self.send_payload(path.read_bytes(), content_type=mime, filename=path.name)
            elif self.path == '/health':
                self.send_payload({'dashboard_pid': os.getpid(), 'port': self.server.port, 'state': 'running'})
            else:
                self.send_payload({'error': 'Unknown endpoint'}, 404)
        except (OSError, ValueError) as exc:
            self.send_payload({'error': 'Requested file is unavailable'}, 404)

    def do_POST(self):
        if not self.local_request():
            return
        if self.path not in ('/start', '/stop', '/export-best'):
            self.send_payload({'error': 'Unknown endpoint'}, 404)
            return
        if self.headers.get_content_type() != 'application/json':
            self.send_payload({'error': 'Content-Type application/json required'}, 415)
            return
        try:
            raw_length = self.headers.get('Content-Length', '')
            if not raw_length.isdecimal():
                raise ValueError('A valid Content-Length is required')
            length = int(raw_length)
            if not 1 <= length <= MAX_BODY:
                self.send_payload({'error': 'JSON body must contain 1 to 4096 bytes'}, 413)
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Incomplete request body')
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError('Expected a JSON object')
            if self.path == '/start':
                status, started = self.server.start_worker(value)
                self.send_payload(status, 202 if started else 200)
            elif self.path == '/export-best':
                if value:
                    raise ValueError('Export expects an empty JSON object')
                self.send_payload(self.server.export_best(), 201)
            else:
                if value:
                    raise ValueError('Stop expects an empty JSON object')
                self.send_payload(self.server.stop_worker(), 202)
        except (ValueError, TypeError, OverflowError) as exc:
            self.send_payload({'error': str(exc)}, 400)
        except OSError as exc:
            self.send_payload({'error': str(exc)}, 500)


def create_server(state_dir=None, port=8765, *, worker_factory=None):
    return DashboardServer(state_dir, port, worker_factory=worker_factory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port must be 1..65535')
    with create_server(args.state_dir, args.port) as server:
        try:
            server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
