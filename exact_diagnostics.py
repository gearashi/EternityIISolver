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
    return {'passed':True,'scope':'Tiny 3x3 and 5x5 CPU correctness cases; not a performance result or an Eternity II solution','results':results,'external_network_enabled':False}


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
    results.extend(diagnose_repair_state(seconds))
    return results


def diagnose_repair_state(seconds=5):
    """Execute the packaged neutral, policy and JSON-continuation paths offline.

    Tiny fixed-batch CPU cases establish correctness and frozen dependency
    availability. They do not measure puzzle-solving speed or use native
    time-limited regional searches in the continuation comparison.
    """
    from copy import deepcopy
    import random
    import tempfile
    import numpy as np
    from gpu_assignment import AssignmentEvaluator, assignment_arc_costs
    from repair_checkpoint import CheckpointCodec, load_checkpoint
    from repair_policy import AdaptiveRegions, PairSelector
    from smart_repair import assignment_repair, checkpoint_settings, independent_cells, solve

    problem,known=generated_problem(5,7,273,clue_count=2)
    board=list(known);rng=random.Random(192);groups={}
    for cell,state in enumerate(board):
        if cell not in problem.fixed:
            groups.setdefault(sum(color==0 for color in problem.faces[state]),[]).append(cell)
    for cells in groups.values():
        pieces=[board[cell]//4 for cell in cells];rng.shuffle(pieces)
        for cell,piece in zip(cells,pieces):
            board[cell]=rng.choice([state for state in problem.domains[cell] if state//4==piece])
    evaluator=AssignmentEvaluator(problem,backend='cpu')
    if not evaluator.structurally_valid(board):
        raise RuntimeError('Repair-state diagnostic produced an invalid fixture')
    cells=independent_cells(problem,seed=17)
    plain=evaluator.evaluate([(board,cells)])[0]
    tied=evaluator.evaluate([(board,cells)],tie_seed=17)[0]
    if not np.array_equal(plain['weights'],tied['weights']):
        raise RuntimeError('Neutral matching changed a primary placement score')
    costs=assignment_arc_costs(tied)
    baseline=assignment_repair(problem,board,cells)
    neutral=assignment_repair(problem,board,cells,tie_seed=17)
    if (costs.dtype!=np.int64 or not baseline['optimal'] or not neutral['optimal']
            or baseline['score']!=neutral['score'] or not evaluator.structurally_valid(neutral['board'])):
        raise RuntimeError('Neutral matching failed its primary-optimum witness check')
    results=[{'engine':'neutral-matching','primary_optimum_preserved':True,
              'legal_working_board_validated':True,'reused_structure_executed':True}]

    pair=PairSelector(problem).choose([known,board],seed=17)
    if pair is None or {pair['left'],pair['right']}!={0,1} or pair['examined_pairs']!=1:
        raise RuntimeError('Repair pair-selection diagnostic failed')
    policy=AdaptiveRegions(max_seconds=.25)
    plan=policy.plan(elapsed_seconds=2.,remaining_seconds=1.)
    if plan is None:
        raise RuntimeError('Adaptive-region diagnostic did not schedule its initial arm')
    policy.observe(plan['arm'],elapsed_seconds=.02,score_gain=1)
    restored_policy=AdaptiveRegions(max_seconds=.25)
    restored_policy.restore_state(policy.export_state())
    if restored_policy.export_state()!=policy.export_state():
        raise RuntimeError('Adaptive-region diagnostic failed its state round trip')
    results.append({'engine':'repair-policy','bounded_pair_selection_executed':True,
                    'adaptive_policy_round_trip_validated':True})

    settings=dict(seed=17,diversify=False,region_repair_enabled=False,pair_selection=False)
    options=dict(settings,seconds=max(1.,seconds),on_checkpoint=lambda state:None)
    uninterrupted=solve(problem,board,max_batches=2,**options)
    first=solve(problem,board,max_batches=1,**options)
    # The only file is a bounded synthetic checkpoint in a fresh temporary folder.
    with tempfile.TemporaryDirectory(prefix='eternity-repair-checkpoint-diagnostic-') as temporary:
        path=Path(temporary)/'repair.json'
        path.write_text(json.dumps(first['checkpoint'],allow_nan=False),encoding='utf-8')
        checkpoint=load_checkpoint(path)
        codec=CheckpointCodec(problem,checkpoint_settings(**settings))
        decoded=codec.decode(checkpoint)
        if decoded['best']!=first['board']:
            raise RuntimeError('Repair checkpoint lost its validated best board')
        resumed=solve(problem,board,checkpoint=checkpoint,max_batches=1,**options)
    expected=deepcopy(uninterrupted['checkpoint']['state'])
    actual=deepcopy(resumed['checkpoint']['state'])
    expected.pop('elapsed_total');actual.pop('elapsed_total')
    if (not resumed['resumed'] or expected!=actual or
            not evaluator.structurally_valid(resumed['board'])):
        raise RuntimeError('Repair checkpoint changed the fixed-batch CPU continuation')
    results.append({'engine':'repair-checkpoint','json_round_trip_validated':True,
                    'cpu_fixed_batch_continuation_validated':True,
                    'comparison_excludes_elapsed_time':True})
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
