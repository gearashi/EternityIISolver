"""Bounded offline comparison of actual-board GPU costs and cycle decomposition.

Sequential workers; read-only local cache; no canonical runtime changes/network.
"""
from pathlib import Path
import sys, time, json, hashlib, platform, subprocess, os
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FILES=('gpu_assignment.py','smart_repair.py','binary_pairwise.py','exact_model.py','validator.py','scripts/benchmark_round2_repair.py')
def hashes():return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES}
def digest(board):return hashlib.sha256(json.dumps(board,separators=(',',':')).encode()).hexdigest()
def child(mode):
    from validator import load_bundle,validate_board
    from exact_model import from_bundle,problem_hash
    from search_resources import peak_memory_mb
    import smart_repair
    from gpu_assignment import AssignmentEvaluator
    import numpy as np
    b=load_bundle();p=from_bundle(b);ref=list(b.record_board)
    started=time.monotonic();deadline=started+40
    def stop():return time.monotonic()>=deadline or peak_memory_mb()>2048
    parents=smart_repair.cached_parents(p,ROOT.parent/'library/library.sqlite3',limit=12)
    parents=[ref]+[v for v in parents if v!=ref]
    if any(not validate_board(v,b)['valid'] for v in parents):raise AssertionError('Invalid cached parent')
    if mode.startswith('cost-'):
        backend=mode[5:];jobs=[]
        for i in range(16):
            jobs.append((parents[i%len(parents)],smart_repair.independent_cells(p,seed=17+i,parity=i if i<2 else None)))
        cpu=AssignmentEvaluator(p,backend='cpu',batch_size=16,should_stop=stop)
        gpu=AssignmentEvaluator(p,backend=backend,batch_size=16,should_stop=stop,cache_dir=ROOT.parent/'research/round2-gpu-cache')
        t=time.monotonic();expected=cpu.evaluate(jobs);cpu_matrix=time.monotonic()-t
        t=time.monotonic();actual=gpu.evaluate(jobs);cold=time.monotonic()-t
        equal=all(np.array_equal(x['weights'],y['weights']) and np.array_equal(x['states'],y['states']) for x,y in zip(expected,actual))
        if not equal:raise AssertionError('GPU differs from scalar CPU oracle')
        smart_repair.assignment_repair_batch(p,jobs[:1],evaluator=cpu,should_stop=stop)
        trials=[]
        for repeat in range(2):
            for name,evaluator in ([('cpu',cpu),('gpu',gpu)] if repeat==0 else [('gpu',gpu),('cpu',cpu)]):
                if stop():raise RuntimeError('Diagnostic resource/time limit')
                t=time.monotonic();out=smart_repair.assignment_repair_batch(p,jobs,evaluator=evaluator,should_stop=stop);elapsed=time.monotonic()-t
                checked=[validate_board(r['board'],b) for r in out]
                if len(out)!=len(jobs) or not all(r['optimal'] and c['valid'] and r['score']==c['score'] for r,c in zip(out,checked)):raise AssertionError('Incomplete/invalid assignment')
                trials.append(dict(repeat=repeat,method=name,seconds=elapsed,scores=[r['score'] for r in out]))
        if any(t['scores']!=trials[0]['scores'] for t in trials):raise AssertionError('CPU/GPU score mismatch')
        result=dict(trials=trials,cpu_matrix_seconds=cpu_matrix,gpu_cold_matrix_seconds=cold,all_matrices_equal=True,
                    jobs=[{'board_sha256':digest(board),'cells':cells} for board,cells in jobs],**gpu.stats())
    else:
        from unittest.mock import patch
        import contextlib
        from ortools.sat.python import cp_model
        # Import native code before both measured modes.
        warm=cp_model.CpModel();warm.new_bool_var('warm');warm_solver=cp_model.CpSolver();warm_solver.parameters.num_search_workers=1;warm_solver.solve(warm)
        result={'trials':[]}
        selected=parents[1:9]
        if not selected:raise ValueError('Cycle comparison requires at least one cached alternate parent')
        for repeat in range(2):
            for fast in ([False,True] if repeat==0 else [True,False]):
                context=contextlib.nullcontext() if fast else patch('binary_pairwise.maximize',return_value={'optimal':False})
                with context:
                    for i,parent in enumerate(selected):
                        if stop():raise RuntimeError('Diagnostic resource/time limit')
                        t=time.monotonic();out=smart_repair.cycle_recombine(p,ref,parent,seconds=min(1.5,deadline-time.monotonic()),seed=17+i,should_stop=stop);elapsed=time.monotonic()-t
                        checked=validate_board(out['board'],b)
                        if not checked['valid'] or checked['score']!=out['score']:raise AssertionError('Cycle validation failed')
                        result['trials'].append(dict(repeat=repeat,fast_path=fast,parent_sha256=digest(parent),seconds=elapsed,**{k:v for k,v in out.items() if k!='board'}))
        result['ablation']='fast_path=false monkeypatches binary_pairwise.maximize to return nonoptimal and uses unchanged CP-SAT fallback'
    result.update(mode=mode,passed=True,problem_sha256=problem_hash(p),reference_sha256=digest(ref),
                  source_sha256=hashes(),parent_sha256=[digest(v) for v in parents],elapsed_seconds=time.monotonic()-started,
                  peak_memory_mb=peak_memory_mb(),scope='Original Eternity II; all five oriented clues; each frozen-complement repair neighborhood only')
    print(json.dumps(result,allow_nan=False))
def main():
    import argparse
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--child',choices=('cost-cuda','cost-opencl','cycles'));ap.add_argument('--output',type=Path);args=ap.parse_args()
    if args.child:child(args.child);return
    if args.output is None or args.output.exists():ap.error('Specify a new --output path')
    report=dict(schema_version=1,utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),source_sha256_before=hashes(),
                limits={'per_case_seconds':40,'process_timeout_seconds':50,'soft_memory_mb':2048,'cpu_workers':1,'gpu_count':1},results=[],
                non_claims=['Neighborhood optima are not full puzzle optima.','Two order-reversed repeats are exploratory desktop timings.','Hardware results on NVIDIA do not certify AMD or Apple hardware.','No network, uploads or canonical runtime writes.'])
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0')
    for mode in ('cost-cuda','cost-opencl','cycles'):
        run=subprocess.run([sys.executable,'-B','-X','utf8',__file__,'--child',mode],cwd=ROOT,env=env,capture_output=True,text=True,timeout=50)
        item=json.loads(run.stdout) if run.returncode==0 else {'mode':mode,'passed':False,'stdout':run.stdout,'stderr':run.stderr}
        report['results'].append(item);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:item[k] for k in ('mode','passed','elapsed_seconds','peak_memory_mb') if k in item}),flush=True)
    report['source_sha256_after']=hashes();report['sources_unchanged']=report['source_sha256_before']==hashes()
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if not all(v['passed'] for v in report['results']):raise SystemExit(1)
if __name__=='__main__':main()
