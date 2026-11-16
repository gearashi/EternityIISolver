"""Bounded frozen-dashboard DFS lifecycle check, isolated from user state."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dfs_profiles import dfs_profile_options
from exact_dfs import normalize_options
from exact_rules import RULES_VERSION
from validator import validate_board


def profile_fingerprint(profile):
    options = normalize_options(dfs_profile_options(profile))
    if profile in ('full-pivot','full-dispatch','full-neighbor') and options.get('pivot_cut_v1') is not True:
        raise AssertionError('The expected opt-in profile has no pivot proof setting')
    if bool(options.get('pivot_dispatch_v1')) != (profile in ('full-dispatch','full-neighbor')):
        raise AssertionError('Unexpected one-sided dispatch setting')
    if bool(options.get('neighbor_dispatch_v1')) != (profile == 'full-neighbor'):
        raise AssertionError('Unexpected forced-neighbor proof setting')
    payload = {'options': options, 'rules_version': RULES_VERSION if options['extra_rules'] is not None else None}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise RuntimeError('Unexpected local redirect')

def log_tail(path, limit=65536):
    """Keep startup diagnostics bounded, including when the temporary home is removed."""
    try:
        with path.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell()-limit))
            return stream.read(limit).decode('utf-8', errors='replace')
    except OSError as exc:
        return f'Could not read server log: {exc}'

def save_failure(output, report, server_log):
    # Print diagnostics even when CI does not upload artifacts after a failed step.
    print(json.dumps(report), file=sys.stderr)
    print('--- isolated server log ---\n'+server_log, file=sys.stderr)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        output.with_suffix('.server.log').write_text(server_log, encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--startup-timeout',type=float,default=60,
                        help='Seconds allowed for a cold frozen process to become ready (default: 60).')
    args=parser.parse_args()
    if not 0 < args.startup_timeout <= 120:
        parser.error('--startup-timeout must be positive and at most 120 seconds')
    exe=args.executable.resolve()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    with tempfile.TemporaryDirectory(prefix='eternity-exact-app-') as directory:
        home=Path(directory).resolve()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        url=f'http://127.0.0.1:{port}'
        def request(path,body=None,timeout=3):
            data=None if body is None else json.dumps(body).encode()
            req=urllib.request.Request(url+path,data=data,headers={'Content-Type':'application/json'})
            with opener.open(req,timeout=timeout) as response:return json.load(response)
        last_status=None
        last_http_error=None
        def wait_for(predicate,limit=30):
            nonlocal last_status,last_http_error
            until=time.monotonic()+limit
            while time.monotonic()<until:
                returncode=process.poll()
                if returncode is not None:
                    raise AssertionError(f'Isolated app exited before readiness (exit code {returncode})')
                try:
                    last_status=request('/status',timeout=max(.01,min(1,until-time.monotonic())))
                    last_http_error=None
                    if predicate(last_status):return last_status
                except OSError as exc:
                    last_http_error=f'{type(exc).__name__}: {exc}'
                time.sleep(.1)
            raise AssertionError(f'Timed out waiting for isolated app after {limit}s: '
                                 f'status={last_status}; last HTTP error={last_http_error}')
        options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        with (home/'server.log').open('wb') as log:
            process=subprocess.Popen([str(exe),'serve','--state-dir',str(home),'--port',str(port)],stdout=log,stderr=log,
                env=dict(os.environ,PYINSTALLER_RESET_ENVIRONMENT='1'),**options)
            owned=False
            failure=None
            failure_traceback=None
            cleanup_errors=[]
            def belongs(s):return s.get('dashboard_pid')==process.pid and s.get('state_dir')==str(home)
            try:
                initial=wait_for(lambda s:belongs(s) and s.get('can_start'),limit=args.startup_timeout)
                owned=True
                runs=[]
                for index in range(2):
                    request('/start',{'method':'exact','exact_engine':'dfs','workers':1,'seed':20261007,'seconds':1.5})
                    ended=wait_for(lambda s:s.get('method')=='exact' and s.get('state')=='finished_bounded_run' and s.get('can_start'))
                    assert ended.get('external_network_enabled') is False
                    assert ended.get('resumed') is bool(index),ended
                    assert (home/'runtime/exact-dfs-checkpoint.json').is_file()
                    best=json.loads((home/'runtime/best.json').read_text())
                    checked=validate_board(best['board']);assert checked['valid'] and checked['score']==466
                    runs.append({k:ended.get(k) for k in ('state','exact_outcome','resumed','branches','conflicts','best_score','external_network_enabled')})
                checkpoint=home/'runtime/exact-dfs-checkpoint.json'
                request('/start',{'method':'exact','exact_engine':'dfs','workers':1,'seconds':3})
                request('/stop',{})
                stopped=wait_for(lambda s:s.get('state')=='stopped' and s.get('can_start'))
                assert (home/'runtime/STOP').exists() and checkpoint.is_file()
                assert not (home/'runtime/checkpoint.npz').exists()
                report={'passed':True,'scope':'isolated frozen HTTP Start, bounded timeout, DFS resume and Stop; no full solution claimed',
                    'runs':runs,'stop_state':stopped['state'],'external_network_enabled':False,
                    'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
                # Keep the existing default-start, resume and Stop receipt before
                # starting a distinct profile in this temporary home.
                default_checkpoint = json.loads(checkpoint.read_text(encoding='utf-8'))
                assert default_checkpoint['options_hash'] == profile_fingerprint('baseline')
                known_hashes = {default_checkpoint['options_hash'], profile_fingerprint('full')}
                profiles = (('full-pivot', 'pivot_profile', 'default_checkpoint_archived'),
                            ('full-dispatch', 'dispatch_profile', 'pivot_checkpoint_archived'),
                            ('full-neighbor', 'neighbor_profile', 'dispatch_checkpoint_archived'))
                for profile, receipt_key, archive_field in profiles:
                    prior_bytes = checkpoint.read_bytes()
                    previous_archives = set(checkpoint.parent.glob('exact-dfs-checkpoint.archived-*.json'))
                    request('/start', {'method': 'exact', 'exact_engine': 'dfs', 'workers': 1,
                                       'seed': 20261007, 'seconds': 1.5, 'pruning': profile, 'fresh_exact': True})
                    end = wait_for(lambda s: belongs(s) and s.get('can_start')
                                   and s.get('state') == 'finished_bounded_run' and s.get('pruning') == profile)
                    assert end.get('exact_outcome') == 'timeout' and end.get('resumed') is False, end
                    assert end.get('method') == 'exact' and end.get('exact_engine') == 'dfs', end
                    assert end.get('workers') == 1 and end.get('external_network_enabled') is False, end
                    enabled = json.loads(checkpoint.read_text(encoding='utf-8'))
                    enabled_hash = profile_fingerprint(profile)
                    assert enabled_hash not in known_hashes and enabled['options_hash'] == enabled_hash
                    known_hashes.add(enabled_hash)
                    assert enabled['problem_hash'] == default_checkpoint['problem_hash']
                    assert enabled['seed'] == default_checkpoint['seed'] == 20261007
                    archived = set(checkpoint.parent.glob('exact-dfs-checkpoint.archived-*.json')) - previous_archives
                    assert len(archived) == 1 and archived.pop().read_bytes() == prior_bytes
                    report[receipt_key] = {'pruning': end['pruning'], 'exact_outcome': end['exact_outcome'],
                                           'resumed': end['resumed'], 'workers': end['workers'],
                                           'seconds_limit': 1.5, 'options_hash': enabled_hash, archive_field: True}
            except BaseException as exc:
                failure=exc
                failure_traceback=traceback.format_exc()
                raise
            finally:
                returncode_before_cleanup=process.poll()
                try:
                    # A failed startup has no confirmed server ownership. Do not
                    # make another HTTP request that can obscure the first error.
                    if owned:
                        current=request('/status')
                        if belongs(current) and current.get('can_stop'):
                            request('/stop',{});wait_for(lambda s:belongs(s) and s.get('can_start'))
                except Exception as exc:
                    cleanup_errors.append(f'Worker cleanup: {type(exc).__name__}: {exc}')
                finally:
                    try:
                        if process.poll() is None:
                            process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill();process.wait(timeout=10)
                    except Exception as exc:
                        cleanup_errors.append(f'Server cleanup: {type(exc).__name__}: {exc}')
                if failure is not None or cleanup_errors:
                    diagnostic={'passed':False,'error':str(failure) if failure is not None else 'Cleanup failed',
                                'traceback':failure_traceback,'server_pid':process.pid,
                                'server_returncode':process.poll(),
                                'server_returncode_before_cleanup':returncode_before_cleanup,
                                'ownership_confirmed':owned,
                                'last_status':last_status,'last_http_error':last_http_error,
                                'startup_timeout_seconds':args.startup_timeout,'cleanup_errors':cleanup_errors}
                    try:
                        save_failure(args.output,diagnostic,log_tail(home/'server.log'))
                    except Exception as exc:
                        print(f'Could not preserve smoke diagnostics: {exc}',file=sys.stderr)
                    if failure is None:
                        raise AssertionError('; '.join(cleanup_errors))
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
if __name__=='__main__':main()
