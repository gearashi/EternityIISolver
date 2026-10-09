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
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from validator import validate_board

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise RuntimeError('Unexpected local redirect')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    exe=args.executable.resolve()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    with tempfile.TemporaryDirectory(prefix='eternity-exact-app-') as directory:
        home=Path(directory).resolve()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        url=f'http://127.0.0.1:{port}'
        def request(path,body=None):
            data=None if body is None else json.dumps(body).encode()
            req=urllib.request.Request(url+path,data=data,headers={'Content-Type':'application/json'})
            with opener.open(req,timeout=3) as response:return json.load(response)
        def wait_for(predicate,limit=25):
            until=time.monotonic()+limit
            last=None
            while time.monotonic()<until:
                try:
                    last=request('/status')
                    if predicate(last):return last
                except OSError:pass
                time.sleep(.1)
            raise AssertionError(f'Timed out waiting for isolated app: {last}')
        options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        with (home/'server.log').open('wb') as log:
            process=subprocess.Popen([str(exe),'serve','--state-dir',str(home),'--port',str(port)],stdout=log,stderr=log,
                env=dict(os.environ,PYINSTALLER_RESET_ENVIRONMENT='1'),**options)
            owned=False
            def belongs(s):return s.get('dashboard_pid')==process.pid and s.get('state_dir')==str(home)
            try:
                initial=wait_for(lambda s:belongs(s) and s.get('can_start'))
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
            finally:
                try:
                    current=request('/status')
                    if owned and belongs(current) and current.get('can_stop'):
                        request('/stop',{});wait_for(lambda s:s.get('can_start'))
                finally:
                    process.terminate();process.wait(timeout=10)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
if __name__=='__main__':main()
