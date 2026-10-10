"""Portable controls for local Eternity II GPU repair and exact search."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.parse
import webbrowser

from app_paths import resource_root, state_root
from process_control import read_status

VERSION = '0.2.2'


class _NoLocalRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise OSError('Local dashboard redirects are disabled')


def local_urlopen(request, *, timeout):
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1'
            or parsed.username is not None or parsed.password is not None):
        raise ValueError('Control requests must target the local dashboard')
    # Local controls bypass system proxies and cannot redirect to remote services.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoLocalRedirect()).open(request, timeout=timeout)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--version', action='version', version=VERSION)
    commands = result.add_subparsers(dest='command', required=True)
    for name in ('start', 'run', 'stop', 'status', 'open', 'serve'):
        command = commands.add_parser(name)
        command.add_argument('--state-dir', type=Path, help='Folder for checkpoints, cache, logs and results')
        if name in ('start', 'run'):
            command.add_argument('--method', choices=('gpu', 'exact'), default='gpu')
            command.add_argument('--exact-engine', choices=('dfs', 'sat', 'cp-sat', 'hybrid'), default='dfs')
            command.add_argument('--workers', type=int, default=1, help='Exact CPU workers (1; CP-SAT supports up to 4)')
            command.add_argument('--backend', choices=('auto', 'cuda', 'opencl'), default='auto')
            command.add_argument('--replicas', type=int, default=4096)
            command.add_argument('--seed', type=int, default=20261007)
            command.add_argument('--seconds', type=float, help='Bounded run; otherwise continues until stopped')
            command.add_argument('--no-library', action='store_true', help='Skip solver cache checks; archive downloads require the dashboard button')
            command.add_argument('--port', type=int, default=8765)
        if name in ('open', 'serve'):
            command.add_argument('--port', type=int, default=8765)
        if name == 'run':
            command.add_argument('--no-dashboard', action='store_true', help=argparse.SUPPRESS)
            command.add_argument('--preserve-stop', action='store_true', help=argparse.SUPPRESS)
        if name == 'start':
            command.add_argument('--no-browser', action='store_true')
        if name == 'status':
            command.add_argument('--json', action='store_true')
    validate = commands.add_parser('validate', help='Independently check a saved board (default: bundled source)')
    validate.add_argument('board_json', nargs='?', type=Path)
    commands.add_parser('diagnose', help='Run bounded GPU correctness diagnostics; diagnose --help lists options')
    commands.add_parser('boinc', help='Run an experimental bounded BOINC workunit; boinc --help lists options')
    commands.add_parser('inspect-cpu', help='Read CPU workunit inputs and ticket IDs without running a search')
    commands.add_parser('diagnose-hybrid', help='Run bounded GPU random-board sampling diagnostics')
    commands.add_parser('diagnose-exact', help='Run small offline CPU search-engine correctness checks')
    commands.add_parser('exact', help='Run an offline exact CPU search; exact --help lists options')
    commands.add_parser('export-best', help='Save validated candidate files locally for manual review; never uploads')
    return result


def run_arguments(args):
    if getattr(args, 'method', 'gpu') == 'exact':
        result = ['--state-dir', str(state_root(args.state_dir)), '--engine', args.exact_engine,
                  '--workers', str(args.workers), '--seed', str(args.seed), '--port', str(args.port)]
        if args.exact_engine == 'hybrid':
            result.extend(['--backend', args.backend, '--replicas', str(args.replicas)])
        if args.seconds is not None:
            result.extend(['--seconds', str(args.seconds)])
        if getattr(args, 'preserve_stop', False):
            result.append('--preserve-stop')
        return result
    result = ['--state-dir', str(state_root(args.state_dir)), '--backend', args.backend,
              '--replicas', str(args.replicas), '--seed', str(args.seed), '--port', str(args.port)]
    if args.seconds is None:
        result.append('--forever')
    else:
        result.extend(['--seconds', str(args.seconds)])
    if args.no_library:
        result.append('--no-library')
    if getattr(args, 'no_dashboard', False):
        result.append('--no-dashboard')
    if getattr(args, 'preserve_stop', False):
        result.append('--preserve-stop')
    return result


def dashboard_url(status):
    port = status.get('port', 8765)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Saved dashboard port is invalid')
    return f'http://127.0.0.1:{port}/'


def ensure_dashboard(home, port=8765):
    from process_control import process_alive
    runtime = home / 'runtime'
    control_path = runtime / 'dashboard.json'
    if control_path.is_file():
        try:
            control = json.loads(control_path.read_text(encoding='utf-8'))
            if process_alive(control.get('pid')):
                url = dashboard_url(control)
                with local_urlopen(url + 'status', timeout=2) as response:
                    status = json.load(response)
                if status.get('dashboard_pid') == control['pid']:
                    return url
        except (OSError, ValueError, KeyError):
            pass
    runtime = home / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    command = [sys.executable]
    if not getattr(sys, 'frozen', False):
        command.append(str(Path(__file__).resolve()))
    command.extend(['serve', '--state-dir', str(home), '--port', str(port)])
    options = {'stdin': subprocess.DEVNULL, 'close_fds': True}
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options['start_new_session'] = True
    # PyInstaller child processes must unpack their own independent resources.
    options['env'] = dict(os.environ, PYINSTALLER_RESET_ENVIRONMENT='1')
    with (runtime / 'dashboard.stdout.log').open('ab') as stdout, (runtime / 'dashboard.stderr.log').open('ab') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, **options)
    deadline = time.monotonic() + 20
    url = f'http://127.0.0.1:{port}/'
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Dashboard exited during startup. Check {runtime / "dashboard.stderr.log"}. Another app may own port {port}.')
        try:
            with local_urlopen(url + 'status', timeout=1) as response:
                live = json.load(response)
            # Never mistake another process on the port for our new search.
            if live.get('dashboard_pid') and control_path.is_file():
                control = json.loads(control_path.read_text(encoding='utf-8'))
                if live['dashboard_pid'] == control.get('pid') and live.get('state_dir') == str(home):
                    return url
        except (OSError, ValueError):
            pass
        time.sleep(.2)
    raise RuntimeError(f'Dashboard startup timed out. Check {runtime / "dashboard.stderr.log"}.')


def start(args):
    home = state_root(args.state_dir)
    url = ensure_dashboard(home, args.port)
    payload = json.dumps({'method': args.method, 'exact_engine': args.exact_engine, 'workers': args.workers, 'replicas': args.replicas, 'backend': args.backend,
                          'seed': args.seed, 'seconds': args.seconds, 'no_library': args.no_library}).encode()
    request = urllib.request.Request(url + 'start', data=payload,
                                     headers={'Content-Type': 'application/json'}, method='POST')
    with local_urlopen(request, timeout=10) as response:
        result = json.load(response)
    if not args.no_browser:
        webbrowser.open(url)
    print(f'Start requested. Dashboard: {url}\nSaved state: {home}')
    return 0


def stop(home):
    runtime = home / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / 'STOP').write_text('Manual stop requested\n', encoding='utf-8')
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        saved = read_status(home)
        if not saved.get('process_alive'):
            print(f'Search stopped. Saved state: {home}')
            return 0
        time.sleep(.2)
    print('Stop requested. The search will finish its current operation and save progress.')
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'diagnose':
        from test_gpu import main as diagnose
        return diagnose(argv[1:]) or 0
    if argv and argv[0] == 'boinc':
        from boinc_worker import main as batch
        return batch(argv[1:])
    if argv and argv[0] == 'inspect-cpu':
        from boinc_cpu_workunit import main as inspect_cpu
        return inspect_cpu(argv[1:])
    if argv and argv[0] == 'diagnose-hybrid':
        from hybrid_sampling import main as diagnose_hybrid
        return diagnose_hybrid(argv[1:])
    if argv and argv[0] == 'diagnose-exact':
        from exact_diagnostics import main as diagnose_exact
        return diagnose_exact(argv[1:])
    if argv and argv[0] == 'exact':
        from exact_worker import main as exact_search
        return exact_search(argv[1:])
    if argv and argv[0] == 'export-best':
        from manual_export import main as export_best
        return export_best(argv[1:])
    args = parser().parse_args(argv or ['open'])
    try:
        if args.command == 'validate':
            from validator import board_from_json, load_bundle, validate_board
            bundle = load_bundle()
            board = board_from_json(json.loads(args.board_json.read_text(encoding='utf-8'))) if args.board_json else bundle.record_board
            report = validate_board(board, bundle)
            print(json.dumps(report, indent=2))
            return 0 if report['valid'] else 2
        home = state_root(args.state_dir)
        if args.command == 'serve':
            from dashboard_server import main as serve
            return serve(['--state-dir', str(home), '--port', str(args.port)])
        if args.command == 'run':
            if args.method == 'exact':
                from exact_worker import main as run_solver
            else:
                from solver import main as run_solver
            return run_solver(run_arguments(args))
        if args.command == 'start':
            return start(args)
        if args.command == 'stop':
            return stop(home)
        saved = read_status(home)
        if args.command == 'open':
            url = ensure_dashboard(home, args.port)
            webbrowser.open(url)
            print(f'Dashboard: {url}')
        elif args.json:
            print(json.dumps(saved, indent=2))
        else:
            print(f"State: {saved['state']}\nProcess running: {saved['process_alive']}\nBest: {saved.get('best_score', 'not started')}/480\nSaved state: {home}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
