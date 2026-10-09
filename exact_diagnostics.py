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
    return {'passed':True,'scope':'Tiny 3x3 CPU correctness cases; not a performance result or an Eternity II solution','results':results,'external_network_enabled':False}

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
