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
    import repair_diversify,repair_parents,repair_regions
    import sqlite3,struct,tempfile
    from contextlib import closing
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
    # A complete mutable core exercises adjacent and frozen-boundary edges in
    # the compact regional model, using the same known-solvable tiny puzzle.
    cells=repair_regions.select_region(problem,reference,seed=11,region_size=8,donor_count=0)
    regional=repair_regions.region_repair(problem,reference,cells,seconds=seconds,seed=11)
    if not check_solution(problem,regional.get('board')) or regional['score']<=regional['reference_score']:
        raise RuntimeError('connected-region failed the tiny score-improvement witness check')
    if any(regional['board'][c]!=reference[c] for c in range(9) if c not in cells):
        raise RuntimeError('connected-region changed the frozen complement')
    results.append({'engine':'connected-region','solvable_case_validated':True,
                    'score_improvement_validated':True,'frozen_complement_preserved':True})

    original=list(known)
    perturbed=repair_diversify.Diversifier(problem).perturb(known,seed=11,moves=2)
    valid=exact_optimize.evaluate_board(problem,perturbed['board'])
    if not valid['valid'] or not perturbed['changed'] or known!=original:
        raise RuntimeError('diversification failed the tiny legal-working-board check')
    results.append({'engine':'diversification','legal_working_board_validated':True,
                    'incumbent_preserved':True})

    # This temporary database is never a user's cache. Exercise actual read-only
    # sampling, decoding, exact deduplication and greedy diversity selection.
    candidates={tuple(board) for board in (known,reference,perturbed['board'])}
    with tempfile.TemporaryDirectory(prefix='eternity-parent-diagnostic-') as temporary:
        database=Path(temporary)/'parents.sqlite3'
        with closing(sqlite3.connect(database)) as db:
            db.execute('CREATE TABLE boards (score INTEGER, placement BLOB)')
            for board in sorted(candidates):
                checked=exact_optimize.evaluate_board(problem,board)
                db.execute('INSERT INTO boards VALUES (?,?)',
                           (checked['score'],struct.pack('<9H',*board)))
            db.execute('INSERT INTO boards VALUES (?,?)',(12,struct.pack('<9H',*known)))
            db.execute('INSERT INTO boards VALUES (?,?)',(12,b'invalid'))
            db.commit()
        before=database.read_bytes()
        parents=repair_parents.cached_parents(problem,database,limit=len(candidates),seed=11,
                                              candidate_limit=16,max_seconds=seconds)
        if database.read_bytes()!=before:
            raise RuntimeError('parent sampling modified its cache')
    if (len(parents)!=len(candidates) or {tuple(board) for board in parents}!=candidates
            or any(not exact_optimize.evaluate_board(problem,board)['valid'] for board in parents)):
        raise RuntimeError('parent sampling failed its tiny read-only validated-cache check')
    if exact_optimize.evaluate_board(problem,parents[0])['score']!=12:
        raise RuntimeError('parent sampling failed to preserve its verified elite')
    results.append({'engine':'diverse-parents','legal_parents_validated':True,
                    'duplicate_filtered':True,'cache_unchanged':True})
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
