"""Bounded offline CPU-engine correctness checks on tiny constructed puzzles."""
import argparse
import importlib
import json
from pathlib import Path
from exact_model import generated_problem,check_solution,make_problem
ENGINES={'dfs':'exact_dfs','sat':'exact_sat','cp-sat':'exact_cp'}

def diagnose(engine='all', seconds=5):
    results=[]
    for name in ENGINES if engine=='all' else [engine]:
        solve=importlib.import_module(ENGINES[name]).solve
        problem,known=generated_problem(3,5,101,1)
        result=solve(problem,seconds=seconds,workers=1,seed=11)
        if result['outcome']!='solved' or not check_solution(problem,result['board']):
            raise RuntimeError(f'{name} failed the tiny satisfiable case: {result["outcome"]}')
        impossible=make_problem(problem.size,problem.faces,problem.fixed,domains=[(),*problem.domains[1:]])
        no=solve(impossible,seconds=seconds,workers=1,seed=11)
        if no['outcome']!='infeasible':raise RuntimeError(f'{name} failed the tiny infeasible case: {no["outcome"]}')
        results.append({'engine':name,'solvable_case_validated':True,'contradiction_detected':True,'elapsed_seconds':result['elapsed_seconds']+no['elapsed_seconds']})
    if engine=='all': results.extend(diagnose_smart(seconds))
    return {'passed':True,'scope':'Tiny 3x3 CPU correctness cases; not a performance result or an Eternity II solution','results':results,'external_network_enabled':False}


def diagnose_smart(seconds=5):
    """Exercise every new dependency in a frozen process, including spawn."""
    import exact_dfs,exact_optimize,exact_parallel,smart_repair,exact_sat_optimize
    problem,known=generated_problem(3,5,101,1)
    reference=list(known)
    corners=[c for c in (0,2,6,8) if c not in problem.fixed]
    a,b=corners[:2];pa,pb=reference[a]//4,reference[b]//4
    reference[a]=next(s for s in problem.domains[a] if s//4==pb)
    reference[b]=next(s for s in problem.domains[b] if s//4==pa)
    results=[]
    for name,run in (
        ('strong-dfs',lambda:exact_dfs.solve(problem,seconds=seconds,options=dict(gac=True,independent_completion=True,dual_branch=True,edge_branch=True,extra_rules={}))),
        ('adaptive-dfs',lambda:exact_dfs.solve(problem,seconds=seconds,options=dict(gac=True,independent_completion=True,dual_branch=True,edge_branch=True,probe=True,extra_rules={},adaptive=True))),
        ('sat-optimize',lambda:exact_sat_optimize.optimize(problem,reference,seconds=seconds)),
        ('partitioned',lambda:exact_parallel.solve(problem,seconds=seconds,workers=2)),
        ('optimize',lambda:exact_optimize.optimize(problem,reference,seconds=seconds)),
        ('core-repair',lambda:exact_optimize.core_guided_repair(problem,reference,seconds=seconds)),
        ('matching',lambda:smart_repair.assignment_repair(problem,reference,corners)),
    ):
        result=run()
        if not check_solution(problem,result.get('board')):
            raise RuntimeError(name+' failed the tiny smart-search witness check')
        results.append({'engine':name,'solvable_case_validated':True})
    return results

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=['all',*ENGINES],default='all')
    parser.add_argument('--seconds',type=float,default=5)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if not 0<args.seconds<=30:parser.error('Seconds must be positive and at most30')
    report=diagnose(args.engine,args.seconds)
    text=json.dumps(report,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf-8')
    print(text,end='');return 0
if __name__=='__main__':raise SystemExit(main())
