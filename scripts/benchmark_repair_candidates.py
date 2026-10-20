"""Offline bounded comparisons of the five proposed structured-repair changes.

This script tests candidates 1--4 directly and optional full-portfolio ablations.
Checkpoint split-run equivalence (candidate 5) is tested by the dedicated suite.
A single validated read-only cache sample is frozen for every child comparison.
Default direct cases use one seed, one CPU worker, 15-second region budgets,
75-second external child timeouts, and cooperative 2048 MiB host-memory limits.
No live app or BOINC state is written; socket attempts are rejected and recorded.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import benchmark_repair_portfolio as common

DIRECT=('validation','ties','pairing','regions-fixed','regions-adaptive')
PORTFOLIO=('baseline','reuse-only','ties','pairing','adaptive','combined')
CASES=DIRECT+tuple('portfolio-'+name for name in PORTFOLIO)
EXTRA_SOURCES=('repair_policy.py','repair_checkpoint.py','exact_worker.py',
               'scripts/benchmark_repair_candidates.py')


def sources():
    return {name:common.sha_file(ROOT/name) for name in (*common.SOURCE_FILES,*EXTRA_SOURCES)}


def protected_homes(home):
    homes=[home]
    if sys.platform=='win32':
        live=Path(os.environ.get('LOCALAPPDATA',Path.home()/'AppData/Local'))/'EternityIISolver'
        if live.resolve()!=home.resolve():homes.append(live.resolve())
    return homes


def protected(home):
    homes=protected_homes(home)
    result={}
    for base in homes:
        for name,sha in common.protected_hashes(base).items():
            result[str(base/Path(name))]=sha
        runtime=base/'runtime'
        if runtime.is_dir():
            for path in runtime.iterdir():
                if path.is_file() and (path.suffix in ('.json','.npz','.sqlite','.sqlite3') or path.name=='STOP'):
                    result[str(path)]=common.sha_file(path)
    return result,homes


def checked_board(board,bundle):
    from validator import validate_board
    checked=validate_board(board,bundle)
    if not checked['valid']:raise AssertionError('Independent complete-board validation failed')
    return checked


def warm_cpu(stopped):
    started=time.monotonic()
    from ortools.sat.python import cp_model
    if stopped():raise TimeoutError('CPU warm-up exceeded resource budget')
    model=cp_model.CpModel();bit=model.new_bool_var('warm');model.add(bit==1)
    solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=2
    status=solver.solve(model)
    if status!=cp_model.OPTIMAL or solver.value(bit)!=1:raise RuntimeError('CPU warm-up failed')
    return {'seconds':time.monotonic()-started,'cpu_workers':1,'passed':True}


def compare_validation(problem,bank,stopped):
    from gpu_assignment import AssignmentEvaluator
    from smart_repair import structurally_valid
    evaluator=AssignmentEvaluator(problem,backend='cpu',should_stop=stopped)
    valid=bank[:16];invalid=[]
    for board in valid[:4]:
        duplicate=list(board);duplicate[1]=duplicate[0];invalid.append(duplicate)
        outside=list(board);outside[0]=len(problem.faces);invalid.append(outside)
        rotated=list(board);rotated[0]=4*(board[0]//4)+(board[0]+1)%4;invalid.append(rotated)
        clue=list(board);cell,state=next(iter(problem.fixed.items()));clue[cell]=4*(state//4)+(state+1)%4;invalid.append(clue)
    boards=valid+invalid;expected=[True]*len(valid)+[False]*len(invalid)
    records=[]
    for repeat in range(3):
        modes=('rebuild','reuse') if repeat%2==0 else ('reuse','rebuild')
        for mode in modes:
            if stopped():raise TimeoutError('Validation comparison reached resource cap')
            started=time.monotonic()
            result=[structurally_valid(problem,b) if mode=='rebuild' else evaluator.structurally_valid(b) for b in boards]
            elapsed=time.monotonic()-started
            if result!=expected:raise AssertionError('Validation implementations disagree with invalid fixtures')
            records.append({'repeat':repeat,'mode':mode,'seconds':elapsed,'boards':len(boards)})
    return {'passed':True,'valid_cases':len(valid),'invalid_cases':len(invalid),'records':records,
            'scope':'Same full legality checks; only immutable puzzle preparation is reused'}


def compare_ties(problem,bank,bundle,args,stopped):
    import numpy as np
    from gpu_assignment import AssignmentEvaluator
    from smart_repair import independent_cells,_assignment_from_matrix
    cpu=AssignmentEvaluator(problem,backend='cpu',should_stop=stopped)
    device=AssignmentEvaluator(problem,backend=args.backend,cache_dir=args.case_dir/'gpu-cache',
                               batch_size=16,should_stop=stopped,allow_cpu_fallback=False)
    jobs=[(board,independent_cells(problem,seed=args.seed+i,parity=i%2)) for i,board in enumerate(bank[:2])]
    baseline=None;unique=[set() for _ in jobs];records=[]
    for tie_seed in (None,args.seed,(args.seed+113)&0xffffffff,(args.seed+907)&0xffffffff):
        if stopped():raise TimeoutError('Tie comparison reached resource cap')
        before=time.monotonic();expected=cpu.evaluate(jobs,tie_seed=tie_seed);cpu_seconds=time.monotonic()-before
        before=time.monotonic();actual=device.evaluate(jobs,tie_seed=tie_seed);device_seconds=time.monotonic()-before
        scores=[];hashes=[]
        for i,(left,right) in enumerate(zip(expected,actual)):
            if not np.array_equal(left['weights'],right['weights']) or not np.array_equal(left['states'],right['states']):
                raise AssertionError('Tie matrices disagree between CPU and selected device')
            one=_assignment_from_matrix(problem,left,should_stop=stopped,structural_problem=cpu.structural_problem)
            two=_assignment_from_matrix(problem,right,should_stop=stopped,structural_problem=device.structural_problem)
            if not one['optimal'] or not two['optimal'] or one['board']!=two['board']:
                raise AssertionError('CPU/device tie-aware matching results disagree')
            checked=checked_board(two['board'],bundle)
            if checked['score']!=two['score']:raise AssertionError('Tie score differs from independent validator')
            scores.append(checked['score']);digest=common.board_hash(two['board']);hashes.append(digest);unique[i].add(digest)
        if baseline is None:baseline=scores
        elif scores!=baseline:raise AssertionError('Secondary tie objective changed primary optimum')
        records.append({'tie_seed':tie_seed,'cpu_matrix_seconds':cpu_seconds,'device_matrix_seconds':device_seconds,
                        'scores':scores,'result_sha256':hashes})
    return {'passed':True,'all_matrices_equal':True,'all_primary_optima_equal':True,'all_repairs_valid':True,
            'distinct_equal_score_results_per_neighborhood':[len(x) for x in unique],
            'records':records,'device_stats':device.stats(),
            'scope':'Two fixed actual-board neighborhoods and four tie settings; diversity is not a time-to-solution result'}


def compare_pairing(problem,bank,bundle,args,stopped):
    from repair_policy import PairSelector
    from smart_repair import cycle_recombine
    selector=PairSelector(problem);records=[]
    if len(bank)<3:raise RuntimeError('Pair comparison needs at least three validated distinct parents')
    for repeat in range(6):
        seed=args.seed+repeat
        modes=('random','structured') if repeat%2==0 else ('structured','random')
        for mode in modes:
            if stopped():raise TimeoutError('Pair comparison reached resource cap')
            before=time.monotonic()
            if mode=='random':
                i,j=0,random.Random(seed).randrange(1,len(bank))
                selection_seconds=time.monotonic()-before
                feature_started=time.monotonic()
                features=selector.choose([bank[i],bank[j]],seed=seed,samples=1,should_stop=stopped)
                if features is None:raise TimeoutError('Pair feature accounting cancelled')
                features=dict(features,left=i,right=j)
                feature_accounting_seconds=time.monotonic()-feature_started
            else:
                features=selector.choose(bank,seed=seed,samples=8,should_stop=stopped)
                selection_seconds=time.monotonic()-before
                feature_accounting_seconds=0.
            if features is None:raise TimeoutError('Pair selection cancelled')
            before=time.monotonic()
            result=cycle_recombine(problem,bank[features['left']],bank[features['right']],
                                   seconds=.25,seed=seed,should_stop=stopped)
            cycle_seconds=time.monotonic()-before;checked=checked_board(result['board'],bundle)
            if checked['score']!=result['score'] or result['score']<result['reference_score']:
                raise AssertionError('Pair recombination failed score validation')
            records.append({'repeat':repeat,'mode':mode,'seed':seed,'selection_seconds':selection_seconds,
                            'cycle_seconds':cycle_seconds,'feature_accounting_seconds':feature_accounting_seconds,'features':features,'score':result['score'],
                            'reference_score':result['reference_score'],'optimal':result['optimal'],
                            'cycle_backend':result.get('cycle_backend'),'termination':result['termination'],
                            'board_sha256':common.board_hash(result['board'])})
    return {'passed':True,'records':records,
            'scope':'Different selected parent pairs; flexibility counts alone do not establish search quality'}


def compare_regions(problem,reference,bundle,args,stopped):
    from repair_policy import AdaptiveRegions
    from repair_regions import select_region,region_repair
    adaptive=args.child=='regions-adaptive';policy=AdaptiveRegions(max_seconds=1.,share=.2)
    rng=random.Random(args.seed);best=list(reference);best_score=checked_board(best,bundle)['score']
    started=time.monotonic();deadline=started+args.seconds;records=[];spent=0.
    # Direct operator comparison lends the region planner enough virtual prior
    # matching time to permit the same wall budget. No claim about portfolio
    # scheduling efficiency is made by this direct operator-only experiment.
    allowance_elapsed=args.seconds/.2
    while not stopped() and time.monotonic()<deadline and len(records)<128:
        remaining=deadline-time.monotonic()
        if adaptive:
            plan=policy.plan(elapsed_seconds=allowance_elapsed,remaining_seconds=max(0.,remaining),should_stop=stopped)
        else:
            index=len(records)
            plan={'arm':index%4,'region_size':8+4*(index%4),'donor_count':2+2*((index//4)%3),
                  'seconds':min(.25,remaining)}
        if not plan or plan['seconds']<.02:break
        attempt_started=time.monotonic();attempt_deadline=attempt_started+plan['seconds']
        def stop_region():return stopped() or time.monotonic()>=attempt_deadline or time.monotonic()>=deadline
        seed=rng.randrange(2**31)
        cells=select_region(problem,best,seed=seed,region_size=plan['region_size'],
                            donor_count=plan['donor_count'],should_stop=stop_region)
        before_score=best_score
        if cells and not stop_region():
            result=region_repair(problem,best,cells,seconds=max(0.,attempt_deadline-time.monotonic()),
                                 seed=seed,should_stop=stop_region)
            checked=checked_board(result['board'],bundle)
            if checked['score']!=result['score'] or checked['score']<before_score:
                raise AssertionError('Regional result failed score validation')
            if checked['score']>best_score:best=list(result['board']);best_score=checked['score']
            outcome=result['termination']
        else:outcome='preparation_timeout'
        elapsed=time.monotonic()-attempt_started;spent+=elapsed
        if adaptive:policy.observe(plan['arm'],elapsed_seconds=elapsed,score_gain=best_score-before_score)
        records.append({'plan':plan,'seconds':elapsed,'cells':len(cells),'score':best_score,
                        'gain':best_score-before_score,'termination':outcome})
    return {'passed':True,'adaptive':adaptive,'score':best_score,'board_sha256':common.board_hash(best),
            'board':best,'records':records,'calls':len(records),'region_seconds':spent,
            'elapsed_seconds':time.monotonic()-started,'policy_state':policy.export_state() if adaptive else None,
            'interpretation':'No score improvement is inconclusive, not evidence to disable a profile',
            'scope':'Operator-only equal wall budgets; scheduling share is tested in full portfolio and finite tests'}


def compare_portfolio(problem,bank,bundle,args,stopped):
    from smart_repair import solve
    mode=args.child.removeprefix('portfolio-')
    flags={name:False for name in ('reuse_structure','neutral_ties','pair_selection','adaptive_regions')}
    if mode=='combined':flags={name:True for name in flags}
    else:
        name={'reuse-only':'reuse_structure','ties':'neutral_ties','pairing':'pair_selection','adaptive':'adaptive_regions'}.get(mode)
        if name:flags[name]=True
    improvements=[];samples=[];last_score=466
    def improved(item):
        nonlocal last_score
        checked=checked_board(item['board'],bundle)
        if checked['score']!=item['score'] or checked['score']<=last_score:raise AssertionError('Invalid improvement callback')
        last_score=checked['score'];improvements.append(dict(item,board=list(item['board'])))
    def progress(item):
        if len(samples)<128:samples.append(dict(item))
    result=solve(problem,list(bundle.record_board),seconds=args.seconds,seed=args.seed,workers=1,
                 parents=bank,should_stop=stopped,on_improvement=improved,on_progress=progress,
                 gpu_backend=None if args.backend=='cpu' else args.backend,gpu_batch_size=16,
                 gpu_cache_dir=args.case_dir/'gpu-cache',gpu_allow_cpu_fallback=False,**flags)
    checked=checked_board(result['board'],bundle)
    if checked['score']!=result['score'] or result['score']!=last_score:raise AssertionError('Invalid final portfolio score')
    return {'passed':True,'mode':mode,'flags':flags,'result':result,'progress':samples,'improvements':improvements,
            'final_independent_validation':checked}


def child(args):
    network=common.network_guard()
    from exact_model import from_bundle,problem_hash
    from search_resources import peak_memory_mb
    from validator import load_bundle
    started=time.monotonic();before=sources();fixture_before=common.sha_file(args.fixture)
    reasons=set();peak=0.;last_check=float('-inf')
    def stopped():
        nonlocal peak,last_check
        now=time.monotonic()
        if now-last_check>=.05:
            last_check=now;peak=max(peak,peak_memory_mb())
            if peak>2048:reasons.add('host_memory_limit')
        if now>=started+70:reasons.add('cooperative_child_wall_limit')
        return bool(reasons)
    bundle=load_bundle();problem=from_bundle(bundle)
    fixture=json.loads(args.fixture.read_text(encoding='utf-8'))
    if fixture['problem_sha256']!=problem_hash(problem):raise RuntimeError('Parent fixture problem mismatch')
    bank=fixture['parents']
    if not 1<=len(bank)<=32:raise RuntimeError('Invalid parent fixture bound')
    for board in bank:checked_board(board,bundle)
    if args.child=='validation':warmup={'seconds':0.};result=compare_validation(problem,bank,stopped)
    else:
        if args.child=='ties' or args.child.startswith('portfolio-'):
            warmup=common.warm_up(problem,list(bundle.record_board),bundle,args,stopped)
        else:warmup=warm_cpu(stopped)
        if args.child=='ties':result=compare_ties(problem,bank,bundle,args,stopped)
        elif args.child=='pairing':result=compare_pairing(problem,bank,bundle,args,stopped)
        elif args.child.startswith('regions-'):result=compare_regions(problem,list(bundle.record_board),bundle,args,stopped)
        else:result=compare_portfolio(problem,bank,bundle,args,stopped)
    after=sources();fixture_after=common.sha_file(args.fixture)
    report={'case':args.child,'seed':args.seed,'passed':result['passed'] and before==after and fixture_before==fixture_after and not network and not reasons,
            'source_sha256_before':before,'source_sha256_after':after,'fixture_sha256_before':fixture_before,
            'fixture_sha256_after':fixture_after,'warmup':warmup,'result':result,'network_attempts':network,
            'resource_interruptions':sorted(reasons),'peak_host_memory_mb':max(peak,peak_memory_mb()),
            'total_child_seconds':time.monotonic()-started}
    print(json.dumps(report,allow_nan=False),flush=True)
    return 0 if report['passed'] else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--home',type=Path,default=ROOT.parent)
    parser.add_argument('--backend',choices=('cpu','cuda','opencl'),default='cuda')
    parser.add_argument('--seconds',type=float,default=15)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(DIRECT))
    parser.add_argument('--seeds',nargs='+',type=int,default=[17])
    parser.add_argument('--child',choices=CASES)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--case-dir',type=Path)
    parser.add_argument('--fixture',type=Path)
    args=parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0<args.seconds<=45:parser.error('--seconds must be positive and <=45')
    if any(type(s) is not int or not 0<=s<2**32 for s in [args.seed,*args.seeds]) or len(args.seeds)>2:parser.error('Use at most two uint32 seeds')
    if len(set(args.cases))!=len(args.cases) or len(set(args.seeds))!=len(args.seeds):parser.error('Cases/seeds must be distinct')
    args.home=args.home.resolve()
    homes=protected_homes(args.home)
    forbidden=[(home/name).resolve() for home in homes for name in ('runtime','library','data')]
    if args.child:
        if args.case_dir is None or args.fixture is None:parser.error('Child needs case directory and fixture')
        args.case_dir=args.case_dir.resolve();args.fixture=args.fixture.resolve()
        if any(common.inside(args.case_dir,path) for path in forbidden):parser.error('Child may not write protected state')
        return child(args)
    if args.output is None:parser.error('--output is required')
    args.output=args.output.resolve();artifacts=args.output.with_name(args.output.name+'.artifacts')
    if args.output.exists() or artifacts.exists():parser.error('Output and artifacts must be fresh paths')
    if any(common.inside(args.output,path) for path in forbidden):parser.error('Output must be outside protected state')
    if not (args.home/'runtime/STOP').is_file():parser.error('Canonical search must remain stopped')
    network=common.network_guard();before=sources();protected_before,_=protected(args.home)
    from exact_model import from_bundle,problem_hash
    from repair_parents import cached_parents
    from validator import load_bundle
    bundle=load_bundle();problem=from_bundle(bundle)
    started=time.monotonic();selection_started=time.monotonic()
    sampled=cached_parents(problem,args.home/'library/library.sqlite3',limit=32,seed=args.seeds[0],max_seconds=5.)
    parents=[list(bundle.record_board)]
    for board in sampled:
        checked_board(board,bundle)
        if board not in parents:parents.append(list(board))
        if len(parents)>=32:break
    preparation=time.monotonic()-selection_started
    if len(parents)<3:raise RuntimeError('Need at least three validated cached parents for the comparisons')
    args.output.parent.mkdir(parents=True,exist_ok=True);artifacts.mkdir()
    fixture=artifacts/'parents.json'
    fixture.write_text(json.dumps({'problem_sha256':problem_hash(problem),'parents':parents},separators=(',',':'))+'\n',encoding='utf-8')
    report={'schema':'eternity-repair-candidates-benchmark/v1','started_utc':datetime.now(timezone.utc).isoformat(),
            'environment':common.environment(),'source_sha256_before':before,'protected_files_sha256_before':protected_before,
            'fixture_sha256':common.sha_file(fixture),'parents':[{'sha256':common.board_hash(b),'score':checked_board(b,bundle)['score']} for b in parents],
            'parent_preparation_seconds':preparation,'cases':args.cases,'seeds':args.seeds,'backend':args.backend,
            'limits':{'per_case_solver_seconds':args.seconds,'external_child_seconds':75,'cooperative_child_seconds':70,
                      'soft_host_memory_mib':2048,'cupy_memory_bytes':2048*1024**2,'cpu_workers':1,'gpu_batch':16},
            'results':[],'non_claims':['No full-puzzle solution or infeasibility proof','Tiny or sampled successes do not establish faster time to 480',
                                     'Parent cycle counts are opportunity proxies, not solved coverage','Host/time limits are cooperative between native calls']}
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    try:
        for number,seed in enumerate(args.seeds):
            order=args.cases if number==0 else list(reversed(args.cases))
            for case in order:
                directory=artifacts/(str(seed)+'-'+case);directory.mkdir();temporary=directory/'tmp';temporary.mkdir()
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                         PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',CUPY_GPU_MEMORY_LIMIT=str(2048*1024**2),
                         CUPY_CACHE_DIR=str(directory/'gpu-cache'),CUDA_CACHE_PATH=str(directory/'cuda-driver-cache'),
                         TEMP=str(temporary),TMP=str(temporary),TMPDIR=str(temporary))
                command=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--child',case,'--seed',str(seed),
                         '--backend',args.backend,'--seconds',str(args.seconds),'--home',str(args.home),
                         '--case-dir',str(directory),'--fixture',str(fixture)]
                print(json.dumps({'event':'case_started','case':case,'seed':seed}),flush=True)
                child_started=time.monotonic()
                try:
                    completed=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=75)
                    (directory/'stdout.txt').write_text(completed.stdout,encoding='utf-8')
                    (directory/'stderr.txt').write_text(completed.stderr,encoding='utf-8')
                    try:item=json.loads(completed.stdout)
                    except ValueError:item={'case':case,'seed':seed,'passed':False,'error':'Child did not return JSON'}
                    item['exit_code']=completed.returncode
                    if completed.returncode:item['passed']=False;item['stderr_tail']=completed.stderr[-4000:]
                except subprocess.TimeoutExpired:
                    item={'case':case,'seed':seed,'passed':False,'error':'external_process_timeout'}
                item['command']=command;item['total_process_seconds']=time.monotonic()-child_started
                report['results'].append(item);save()
                print(json.dumps({'event':'case_finished','case':case,'seed':seed,'passed':item['passed']}),flush=True)
    finally:
        report['source_sha256_after']=sources();report['protected_files_sha256_after'],_=protected(args.home)
        report['sources_unchanged']=before==report['source_sha256_after'] and all(item.get('source_sha256_before')==before and item.get('source_sha256_after')==before for item in report['results'])
        report['protected_files_unchanged']=protected_before==report['protected_files_sha256_after']
        report['fixture_unchanged']=report['fixture_sha256']==common.sha_file(fixture)
        report['network_attempts']=network;report['elapsed_seconds']=time.monotonic()-started
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        report['passed']=len(report['results'])==len(args.seeds)*len(args.cases) and all(item['passed'] for item in report['results']) and report['sources_unchanged'] and report['protected_files_unchanged'] and report['fixture_unchanged'] and not network
        save()
    print(json.dumps({'event':'complete','passed':report['passed'],'output':str(args.output)}),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
