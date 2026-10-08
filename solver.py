"""Persistent GPU Eternity II search. Stops only on STOP, Ctrl+C, error or 480."""
import argparse,datetime,hashlib,json,os,signal,sqlite3,threading,time,traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from app_paths import resource_root,state_root
from process_control import RunLock,process_alive
ROOT=resource_root()
from validator import load_bundle,validate_board
import numpy as np

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for attempt in range(6):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==5:raise
            time.sleep(.05)
def digest(board):return hashlib.sha256(np.asarray(board,dtype='<u2').tobytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main(argv=None,engine_factory=None):
    p=argparse.ArgumentParser();p.add_argument('--forever',action='store_true');p.add_argument('--seconds',type=float,default=60);p.add_argument('--replicas',type=int,default=4096);p.add_argument('--seed',type=int,default=20261007);p.add_argument('--no-library',action='store_true');p.add_argument('--no-dashboard',action='store_true');p.add_argument('--preserve-stop',action='store_true');p.add_argument('--port',type=int,default=8765);p.add_argument('--backend',choices=['auto','cuda','opencl'],default='auto');p.add_argument('--state-dir',type=Path);a=p.parse_args(argv)
    if not 32<=a.replicas<=32768:raise ValueError('replicas must be32..32768')
    if not 1<=a.port<=65535:raise ValueError('port must be 1..65535')
    if not np.isfinite(a.seconds) or a.seconds<0:raise ValueError('seconds must be finite and nonnegative')
    home=state_root(a.state_dir);runtime=home/'runtime';runtime.mkdir(parents=True,exist_ok=True);(home/'results').mkdir(exist_ok=True)
    bundle=load_bundle();globalbest=list(bundle.record_board);bestscore=466
    if (runtime/'best.json').exists():
        prior=json.loads((runtime/'best.json').read_text());v=validate_board(prior['board'],bundle)
        if v['valid'] and v['score']>=bestscore:globalbest=prior['board'];bestscore=v['score']
    lock=RunLock(runtime/'run.lock').acquire()
    if not a.preserve_stop:(runtime/'STOP').unlink(missing_ok=True)
    status={'state':'starting','pid':os.getpid(),'last_update':now(),'best_score':bestscore,'source_best_score':466,'replicas':a.replicas,'port':a.port,'state_dir':str(home)}
    monitor=None;server=None;gpu=None;db=None;state_lock=threading.Lock();old_signals={}
    def request_stop(*_):
        (runtime/'STOP').write_text('Stop requested by interrupt or termination signal\n',encoding='utf-8')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*unused):pass
        def do_GET(self):
            if self.path in ('/','/Dashboard.html'):
                payload=(ROOT/'Dashboard.html').read_bytes();mime='text/html; charset=utf-8'
            elif self.path=='/status':
                with state_lock:payload=json.dumps(status).encode()
                mime='application/json'
            elif self.path=='/board':payload=json.dumps({'board':globalbest,'faces':bundle.oriented_edges}).encode();mime='application/json'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
        def do_POST(self):
            if self.path!='/stop':self.send_error(404);return
            origin=self.headers.get('Origin','')
            if origin and origin not in (f'http://127.0.0.1:{a.port}',f'http://localhost:{a.port}'):
                self.send_error(403);return
            (runtime/'STOP').write_text('User stopped from local dashboard\n');self.send_response(202);self.end_headers();self.wfile.write(b'{"stopping":true}')
    try:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT,signal.SIGTERM):
                old_signals[sig]=signal.signal(sig,request_stop)
        atomic_json(runtime/'status.json',status)
        if not a.no_dashboard:
            try:server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
            except OSError as e:raise RuntimeError(f'Cannot open local dashboard port {a.port}; search was not started: {e}') from e
        seeds=[globalbest]
        for file in sorted((ROOT/'data/seeds').glob('*.json')):
            doc=json.loads(file.read_text())
            if 'board' not in doc:continue
            v=validate_board(doc['board'],bundle)
            if not v['valid']:raise ValueError(f'Invalid seed: {file.name}')
            if doc['board'] not in seeds:seeds.append(doc['board'])
        if not a.no_library:
            from library_cache import LibraryMonitor
            monitor=LibraryMonitor(home/'library',interval_seconds=900,pieces_path=ROOT/'data/pieces.txt')
            for file in [ROOT/'data/record466.json',*sorted((ROOT/'data/seeds').glob('*.json'))]:
                try:
                    doc=json.loads(file.read_text())
                    if 'board' in doc:monitor.register_known_document(doc,'2c037e70f7e93518a48733c7aacd096226b7f23728efabd6289088b91c0945cd' if file.name=='record466.json' else file.stem)
                except Exception as e:status['library_seed_warning']=str(e)
            monitor.start()
        if engine_factory is None:
            from gpu_engine import create_engine
            engine_factory=create_engine
        gpu=engine_factory(bundle,a.replicas,a.seed,backend=a.backend,cache_dir=runtime/'gpu-cache');gpu.initialize(seeds)
        resumed=gpu.resume(runtime/'checkpoint.npz') if (runtime/'checkpoint.npz').exists() else False
        gpu.verify_device_scores(range(min(32,a.replicas)))
        db=sqlite3.connect(runtime/'discoveries.sqlite');db.execute('CREATE TABLE IF NOT EXISTS boards (hash TEXT PRIMARY KEY, score INTEGER, first_seen TEXT, library_known TEXT, board TEXT)')
        previous=np.minimum(gpu.bestscores.get(),bestscore);basecounts=gpu.counters.get();baseproposals=int(basecounts[:a.replicas].sum());duplicates=0;novel=0;pending=0
        started=time.monotonic();last_report=started;last_save=started;last_reset=started;last_proposals=baseproposals;steps=16;kernel_ms=0;batch_count=0
        status.update(state='running',gpu=gpu.device,backend=getattr(gpu,'backend',a.backend),best_score=bestscore,seed_boards=len(seeds),resumed=resumed,start_time=now(),cpu_role='Validation, library cache and checkpoints; move search runs on the selected GPU')
        atomic_json(runtime/'status.json',status)
        atomic_json(runtime/'best.json',{'score':bestscore,'board':globalbest,'validated':True,'source':'known seed or prior verified best'})
        while True:
            elapsed=time.monotonic()-started
            stop_reason='stopped' if (runtime/'STOP').exists() else 'solved' if bestscore==480 else 'finished_bounded_run' if not a.forever and elapsed>=a.seconds else None
            if stop_reason is None:
                gpu.cool(elapsed);kernel_ms=gpu.step(steps);batch_count+=1
                if kernel_ms>180:steps=max(1,steps//2)
                elif kernel_ms<30:steps=min(512,steps*2)
            stamp=time.monotonic()
            if stop_reason or stamp-last_report>=3:
                personal=gpu.bestscores.get();improved=np.flatnonzero(personal>previous)
                for idx in improved:
                    b=gpu.bestboards[:,int(idx)].get().tolist();v=validate_board(b,bundle)
                    if not v['valid'] or v['score']!=int(personal[idx]):raise RuntimeError('Invalid GPU candidate; search stopped')
                    h=digest(b);known=monitor.is_known(b) if monitor else None
                    cur=db.execute('INSERT OR IGNORE INTO boards VALUES (?,?,?,?,?)',(h,v['score'],now(),json.dumps(known),json.dumps(b)))
                    if cur.rowcount==0 or known is True:duplicates+=1
                    elif known is False:
                        novel+=1;gpu.arm_rewards[gpu.arms[int(idx)]]+=(v['score']-464)**2
                    else:pending+=1
                    if v['score']>bestscore:
                        globalbest=b;bestscore=v['score'];record={'score':bestscore,'board':b,'validated':True,'library_known':known,'sha256_uint16le':h,'found_at':now()}
                        atomic_json(runtime/'best.json',record);atomic_json(home/'results'/f'{bestscore}-{h}.json',record)
                        print(f'NEW VERIFIED BEST {bestscore}/480 {h}',flush=True)
                        gpu.seeds=np.vstack([gpu.seeds,np.asarray(b,np.int16)])[-32:]
                        status['last_improvement']=now()
                if monitor:
                    for h,raw in db.execute("SELECT hash,board FROM boards WHERE library_known='null' LIMIT 64").fetchall():
                        answer=monitor.is_known(json.loads(raw))
                        if answer is not None:
                            db.execute('UPDATE boards SET library_known=? WHERE hash=?',(json.dumps(answer),h))
                            if answer:duplicates+=1
                            else:novel+=1
                pending=db.execute("SELECT count(*) FROM boards WHERE library_known='null'").fetchone()[0]
                db.commit();previous=personal
                counters=gpu.counters.get();proposals=int(counters[:a.replicas].sum());accepted=int(counters[a.replicas:].sum())
                gpu.verify_device_scores(gpu.host_rng.choice(a.replicas,min(8,a.replicas),replace=False))
                with state_lock:status.update(last_update=now(),elapsed_seconds=round(elapsed,1),best_score=bestscore,proposed_moves=proposals,accepted_moves=accepted,moves_per_second=round((proposals-last_proposals)/max(stamp-last_report,1e-9)),kernel_ms=round(kernel_ms,2),steps_per_batch=steps,batches=batch_count,duplicate_results=duplicates,new_results=novel,pending_novelty_checks=pending,strategies=gpu.strategy_status(),library=monitor.status() if monitor else {'enabled':False})
                atomic_json(runtime/'status.json',status);last_report=stamp;last_proposals=proposals
            if stop_reason:
                status['state']='solved' if bestscore==480 else stop_reason
                break
            if stamp-last_save>=60:gpu.checkpoint(runtime/'checkpoint.npz');last_save=stamp
            if stamp-last_reset>=120:
                ids=gpu.reseed();previous[ids]=gpu.bestscores[ids].get();gpu.adapt();last_reset=stamp
            # Yield a small interval to keep the display and stop controls responsive.
            time.sleep(.002)
        gpu.checkpoint(runtime/'checkpoint.npz')
    except BaseException as e:
        status.update(state='stopped' if isinstance(e,KeyboardInterrupt) else 'error',error=str(e),traceback=traceback.format_exc());print(status['traceback'],flush=True)
    finally:
        try:
            if db:db.close()
            if monitor:
                monitor.stop();status['library']=monitor.status()
            status.update(last_update=now(),best_score=bestscore);atomic_json(runtime/'status.json',status)
        finally:
            if server:server.shutdown();server.server_close()
            for sig,previous_handler in old_signals.items():signal.signal(sig,previous_handler)
            lock.close()
    return 1 if status['state']=='error' else 0
if __name__=='__main__':raise SystemExit(main())
