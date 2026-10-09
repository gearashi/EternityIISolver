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
from validator import validate_board

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
