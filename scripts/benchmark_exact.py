"""Bounded offline comparison. One isolated process/one worker per engine/case.

Both encoding and native solve count against each budget. Resource interruption
and unknown are inconclusive. Regional infeasibility applies only to frozen input.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from exact_model import generated_problem,from_bundle,check_solution,problem_hash
from search_resources import peak_memory_mb
from validator import load_bundle,validate_board
ENGINES={'cp-sat':'exact_cp','sat':'exact_sat','dfs':'exact_dfs'}
CASES=['generated-4','generated-6','generated-8','region-4','region-6','region-8','region-12','whole-five-clue']

def case(name):
    if name.startswith('generated-'):
        size=int(name.split('-')[1]);p,answer=generated_problem(size,size+3,901+size,1)
        assert check_solution(p,answer)
        return p,'synthetic known-solvable correctness/comparison case'
    bundle=load_bundle()
    if name=='whole-five-clue':return from_bundle(bundle),'full original puzzle; all five clues and all 480 edges'
    size=int(name.split('-')[1]);top=3;left=16-size
    cells=[(top+r)*16+left+c for r in range(size) for c in range(size)]
    return from_bundle(bundle,board=list(bundle.record_board),free_cells=cells),'actual 466 board region; outside pieces fixed; only edges touching region constrained'

def child(args):
    problem,scope=case(args.case)
    interrupted=[]
    sampled_at=[0.0]
    def should_stop():
        now=time.monotonic()
        if now-sampled_at[0]<0.1:return bool(interrupted)
        sampled_at[0]=now
        if peak_memory_mb()>args.memory_mb:
            interrupted.append('memory_limit');return True
        return False
    begin=time.monotonic()
    solver=importlib.import_module(ENGINES[args.engine])
    remaining=max(0,args.seconds-(time.monotonic()-begin))
    result=solver.solve(problem,seconds=remaining,workers=1,seed=17,should_stop=should_stop)
    result.update(case=args.case,engine=args.engine,scope=scope,problem_sha256=problem_hash(problem),budget_seconds=args.seconds,process_task_seconds=time.monotonic()-begin,peak_memory_mb=peak_memory_mb(),workers_requested=1)
    board=result.get('board')
    if result['outcome']=='solved':
        if board is None or not check_solution(problem,board):raise AssertionError('Independent exact task validation failed')
        result['independently_validated']=True
        if problem.size==16:
            full=validate_board(board);assert full['valid'];result['full_board_score']=full['score'];result['full_board_complete']=full['complete']
    if interrupted:
        result['resource_interruption']='memory_limit';assert result['outcome']!='infeasible'
    print(json.dumps(result))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',action='store_true');parser.add_argument('--engine',choices=ENGINES);parser.add_argument('--case',choices=CASES)
    parser.add_argument('--seconds',type=float,default=5);parser.add_argument('--full-seconds',type=float,default=30)
    parser.add_argument('--memory-mb',type=int,default=2048);parser.add_argument('--output',type=Path)
    parser.add_argument('--engines',nargs='+',choices=ENGINES,default=list(ENGINES));parser.add_argument('--cases',nargs='+',choices=CASES,default=CASES)
    args=parser.parse_args()
    if args.child:return child(args)
    if args.output is None:parser.error('--output required')
    if not 0<args.seconds<=300 or not 0<args.full_seconds<=300 or not 128<=args.memory_mb<=4096:parser.error('Invalid resource caps')
    report={'schema_version':1,'started_utc':datetime.now(timezone.utc).isoformat(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)), 'python':sys.version,'platform':platform.platform(),'versions':{p:importlib.metadata.version(p) for p in ('ortools','python-sat')},'limits':{'workers_per_case':1,'soft_peak_memory_mb':args.memory_mb,'ordinary_seconds':args.seconds,'whole_seconds':args.full_seconds,'hard_process_timeout_extra_seconds':15},'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['exact_model.py','exact_cp.py','exact_sat.py','exact_dfs.py','search_resources.py','scripts/benchmark_exact.py']},'results':[],'non_claims':['No bounded timeout proves impossibility.','Neighborhood infeasibility only covers its fixed outside pieces.','Synthetic cases do not establish full-puzzle performance.','DFS is Python, SAT/CP native; comparison includes implementation and encoding effects.','Concurrent desktop/BOINC load is uncontrolled; timings are exploratory, not hardware-normalized.']}
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    for index,name in enumerate(args.cases):
        engines=args.engines[index%len(args.engines):]+args.engines[:index%len(args.engines)]
        for engine in engines:
            budget=args.full_seconds if name=='whole-five-clue' else args.seconds
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','--case',name,'--engine',engine,'--seconds',str(budget),'--memory-mb',str(args.memory_mb)]
            start=time.monotonic()
            try:
                run=subprocess.run(cmd,cwd=ROOT,env=environment,capture_output=True,text=True,timeout=budget+15)
                if run.returncode:result={'case':name,'engine':engine,'outcome':'error','returncode':run.returncode,'stderr':run.stderr[-4000:]}
                else:result=json.loads(run.stdout)
            except subprocess.TimeoutExpired:result={'case':name,'engine':engine,'outcome':'external_timeout'}
            result['total_process_seconds']=time.monotonic()-start
            report['results'].append(result)
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            print(json.dumps({k:v for k,v in result.items() if k in ('case','engine','outcome','elapsed_seconds','process_task_seconds','total_process_seconds','peak_memory_mb','full_board_score')}),flush=True)
    report['finished_utc']=datetime.now(timezone.utc).isoformat();args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if any(r['outcome']=='error' for r in report['results']):return 1
if __name__=='__main__':raise SystemExit(main())
