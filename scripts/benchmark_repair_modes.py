"""Offline, isolated, sequential comparison of bounded exact/repair modes.

Uses the bundled validated 466 board and optionally reads the existing local
SQLite cache. No runtime mutation, downloads, GPU work or full-puzzle proof.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from exact_model import from_bundle,problem_hash
from search_resources import peak_memory_mb
from validator import load_bundle,validate_board
import exact_optimize
import exact_sat
import smart_repair

MODES={'sat-baseline':12.0,'sat-reverse-support':12.0,'score-optimization':18.0,
       'core-repair':18.0,'independent-matching':5.0,'cycle-recombination':12.0}
FILES=('exact_sat.py','exact_optimize.py','smart_repair.py','exact_model.py','validator.py',
       'scripts/benchmark_repair_modes.py')


def hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}


def board_hash(board):
    return hashlib.sha256(json.dumps(board,separators=(',',':')).encode()).hexdigest()


def child(args):
    bundle=load_bundle();problem=from_bundle(bundle);reference=list(bundle.record_board)
    base=validate_board(reference,bundle)
    if not base['valid'] or base['score']!=466: raise AssertionError('Bundled reference must validate at466')
    memory_stop=[];last_memory=[0.0];improvements=[]
    beginning=time.monotonic();deadline=beginning+args.seconds
    def stopped():
        now=time.monotonic()
        if now-last_memory[0]>=.1:
            last_memory[0]=now
            if peak_memory_mb()>1536: memory_stop.append(True)
        return bool(memory_stop)
    def bounded_stop():
        return stopped() or time.monotonic()>=deadline
    def independent(board):
        checked=validate_board(board,bundle)
        if not checked['valid']: raise AssertionError('Official board validator rejected a candidate')
        return checked['score']
    def improvement(update):
        actual=independent(update['board'])
        if actual!=update['score']: raise AssertionError('Callback score disagrees with official validator')
        improvements.append({**update,'board_sha256':board_hash(update['board'])})
    if args.mode.startswith('sat-'):
        result=exact_sat.solve(problem,seconds=args.seconds,seed=args.seed,workers=1,
                               should_stop=stopped,reverse_support=args.mode=='sat-reverse-support')
    elif args.mode=='score-optimization':
        result=exact_optimize.optimize(problem,reference,seconds=args.seconds,seed=args.seed,workers=1,
                                        should_stop=stopped,on_improvement=improvement)
    elif args.mode=='core-repair':
        result=exact_optimize.core_guided_repair(problem,reference,seconds=args.seconds,seed=args.seed,workers=1,
                                                should_stop=stopped,on_improvement=improvement)
    elif args.mode=='independent-matching':
        results=[];best=reference;best_score=466
        for index in range(32):
            if bounded_stop(): break
            cells=smart_repair.independent_cells(problem,seed=args.seed+index,parity=index if index<2 else None)
            run=smart_repair.assignment_repair(problem,reference,cells,should_stop=bounded_stop)
            actual=independent(run['board'])
            if actual!=run['score']: raise AssertionError('Assignment score mismatch')
            if actual>best_score: best=run['board'];best_score=actual
            results.append({key:value for key,value in run.items() if key!='board'})
        result={'outcome':'solved' if best_score==480 else 'bounded_repair_complete','score':best_score,
                'board':best,'neighborhoods':results,'completed_optimal':sum(r['optimal'] for r in results),
                'proof_scope':'each completed nonadjacent-cell neighborhood only'}
    else:
        parents=smart_repair.cached_parents(problem,args.cache,limit=16)
        parents=[board for board in parents if board!=reference]
        results=[];best=reference;best_score=466
        for index,parent in enumerate(parents[:8]):
            if bounded_stop(): break
            parent_score=independent(parent)
            run=smart_repair.cycle_recombine(problem,reference,parent,seconds=min(2,max(0,deadline-time.monotonic())),
                                             seed=args.seed+index,should_stop=bounded_stop)
            actual=independent(run['board'])
            if actual!=run['score']: raise AssertionError('Cycle score mismatch')
            if actual>best_score: best=run['board'];best_score=actual
            results.append({**{key:value for key,value in run.items() if key!='board'},
                            'parent_score':parent_score,'parent_sha256':board_hash(parent)})
        result={'outcome':'solved' if best_score==480 else 'bounded_repair_complete' if results else 'no_cached_parents',
                'score':best_score,'board':best,'neighborhoods':results,'cached_parents_loaded':len(parents),
                'completed_optimal':sum(r['optimal'] for r in results),
                'proof_scope':'each completed two-parent cycle neighborhood only'}
    if result.get('board') is not None:
        result['independently_verified_score']=independent(result['board'])
        result['independently_validated']=True
    if result.get('outcome')=='infeasible' and memory_stop:
        raise AssertionError('Memory interruption was reported as infeasibility')
    result.update(mode=args.mode,reference_score=466,reference_sha256=board_hash(reference),
                  problem_sha256=problem_hash(problem),source_sha256=hashes(),requested_seconds=args.seconds,
                  process_search_seconds=time.monotonic()-beginning,peak_memory_mb=peak_memory_mb(),
                  memory_limit_reached=bool(memory_stop),improvements=improvements,workers=1,
                  scope='Original full16x16 puzzle; all five exact oriented clues; all480 edges')
    print(json.dumps(result,allow_nan=False))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',action='store_true')
    parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    parser.add_argument('--seconds',type=float,default=2)
    parser.add_argument('--wall-seconds',type=float,default=120)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--cache',type=Path,default=ROOT.parent/'library'/'library.sqlite3')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if not 0<args.seconds<=18 or not 0<args.wall_seconds<=120:
        parser.error('This harness permits at most18seconds per task and120seconds total')
    if args.child:
        if args.mode is None: parser.error('Child requires --mode')
        return child(args)
    if args.output is None: parser.error('--output is required')
    if args.output.exists(): parser.error('Use a new output path to retain prior evidence')
    before=hashes();started=time.monotonic()
    report={'schema_version':1,'started_utc':datetime.now(timezone.utc).isoformat(),
            'python':sys.version,'platform':platform.platform(),'source_sha256_before':before,
            'limits':{'worker_count':1,'sequential':True,'wall_seconds':args.wall_seconds,
                      'per_mode_seconds':MODES,'soft_peak_memory_mb':1536,'child_timeout_extra_seconds':8},
            'results':[],'non_claims':[
                'Every timeout or resource stop is inconclusive.',
                'Neighborhood optimality does not establish full-board optimality or infeasibility.',
                'SAT and optimization target different objectives; elapsed time is not a solved-puzzle speed comparison.',
                'Single short desktop trials are exploratory, not a statistically reliable algorithm ranking.',
                'All boards are local; no runtime state, GPU, project telemetry or network is used.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    def save(): args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    for mode in args.modes:
        budget=MODES[mode];remaining=args.wall_seconds-(time.monotonic()-started)
        if remaining<budget+8:
            report['wall_cap_stopped']=True;break
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','--mode',mode,
                 '--seconds',str(budget),'--seed',str(args.seed),'--cache',str(args.cache)]
        beginning=time.monotonic()
        try:
            run=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True,timeout=min(budget+8,remaining))
            if run.returncode:
                result={'mode':mode,'outcome':'error','returncode':run.returncode,
                        'stdout':run.stdout[-4000:],'stderr':run.stderr[-6000:]}
            else:
                result=json.loads(run.stdout)
        except subprocess.TimeoutExpired as exc:
            result={'mode':mode,'outcome':'external_timeout','stderr':str(exc.stderr)[-4000:]}
        result['total_process_seconds']=time.monotonic()-beginning
        report['results'].append(result);save()
        keys=('mode','outcome','score','branches','conflicts','build_seconds','rounds',
              'completed_optimal','process_search_seconds','peak_memory_mb')
        print(json.dumps({key:result[key] for key in keys if key in result}),flush=True)
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
                  source_sha256_after=hashes())
    report['sources_unchanged']=before==report['source_sha256_after'] and all(
        item.get('source_sha256',before)==before for item in report['results'])
    save()
    return int(any(item['outcome']=='error' for item in report['results']))


if __name__=='__main__': raise SystemExit(main())
