"""Offline exact repair neighborhoods; neither operator proves full infeasibility.

Independent-set matching follows Schaus/Deville (JFPC 2008). Cycle crossover
chooses whole components of two cell/piece permutations, retaining uniqueness.
"""
from __future__ import annotations
from dataclasses import replace
from contextlib import closing
import math
import itertools
from pathlib import Path
import random
import sqlite3
import struct
import threading
import time
from exact_model import check_solution, constraint_edges, grid_edges


def structurally_valid(problem, board):
    return check_solution(replace(problem, edges=()), board)


def score(problem, board):
    return sum(problem.faces[board[a]][sa] == problem.faces[board[b]][sb]
               for a, b, sa, sb in grid_edges(problem.size))


def _require(problem, board):
    if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
        raise ValueError('Repair requires a full-grid problem; omitted edges are not permitted')
    if not structurally_valid(problem, board):
        raise ValueError('Repair parent violates pieces, domains, frame or fixed clues')


def independent_cells(problem, seed=0, parity=None):
    """Mixed-parity greedy sets avoid permanently locking pieces to one parity."""
    order = [c for c in range(problem.size**2) if c not in problem.fixed]
    if parity is not None:
        if type(parity) is not int or parity not in (0, 1):
            raise ValueError('parity must be zero or one')
        return [c for c in order if (c//problem.size+c%problem.size)%2 == parity]
    random.Random(seed).shuffle(order)
    neighbors = {c:set() for c in order}
    for a,b,_,_ in grid_edges(problem.size):
        if a in neighbors: neighbors[a].add(b)
        if b in neighbors: neighbors[b].add(a)
    chosen = set()
    for cell in order:
        if not chosen.intersection(neighbors[cell]): chosen.add(cell)
    return sorted(chosen)


def assignment_repair(problem, board, cells, *, should_stop=None):
    """Exact maximum-score permutation/rotation of a nonadjacent piece pool.

    Building costs is interruptible; the small native assignment call is not.
    Every accepted result is independently checked and rescored.
    """
    _require(problem, board)
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    cells = list(cells)
    n = problem.size**2
    if len(set(cells)) != len(cells) or any(type(c) is not int or not 0 <= c < n for c in cells):
        raise ValueError('Invalid independent cells')
    chosen = set(cells)
    if chosen.intersection(problem.fixed) or any(a in chosen and b in chosen for a,b,_,_ in grid_edges(problem.size)):
        raise ValueError('Repair cells must be nonadjacent and exclude fixed clues')
    before = score(problem, board)
    baseline = {'board':list(board), 'score':before, 'reference_score':before,
                'optimal':False, 'neighborhood_size':len(cells), 'operator':'independent-matching'}
    def stopped(): return bool(should_stop and should_stop())
    if stopped(): return dict(baseline, termination='stopped')
    if not cells: return dict(baseline, optimal=True, termination='optimal')
    from ortools.graph.python import linear_sum_assignment
    if stopped(): return dict(baseline, termination='stopped')
    pieces = [board[c]//4 for c in cells]
    neighbors = {c:[] for c in cells}
    for a,b,sa,sb in grid_edges(problem.size):
        if a in chosen: neighbors[a].append((b,sa,sb))
        if b in chosen: neighbors[b].append((a,sb,sa))
    matching = linear_sum_assignment.SimpleLinearSumAssignment()
    states = {}
    for i,cell in enumerate(cells):
        if stopped(): return dict(baseline, termination='stopped')
        allowed = set(problem.domains[cell])
        for j,piece in enumerate(pieces):
            if j % 32 == 0 and stopped(): return dict(baseline, termination='stopped')
            candidates = [s for s in range(4*piece,4*piece+4) if s in allowed]
            # Parent legality alone does not imply every domain state obeys frame.
            from exact_model import frame_ok
            candidates = [s for s in candidates if frame_ok(problem.size,cell,problem.faces[s])]
            if not candidates: continue
            def weight(s):
                return sum(problem.faces[s][sa] == problem.faces[board[other]][sb]
                           for other,sa,sb in neighbors[cell])
            state = max(candidates,key=lambda s:(weight(s),s==board[cell],-s))
            states[i,j] = state
            matching.add_arc_with_cost(i,j,-weight(state))
    if stopped(): return dict(baseline, termination='stopped')
    outcome = matching.solve()
    if outcome != matching.OPTIMAL:
        raise RuntimeError('Feasible parent lost by assignment model')
    candidate = list(board)
    for i,cell in enumerate(cells): candidate[cell] = states[i,matching.right_mate(i)]
    actual = score(problem,candidate)
    if not structurally_valid(problem,candidate) or actual < before:
        raise RuntimeError('Assignment repair failed independent validation')
    return dict(baseline,board=candidate,score=actual,optimal=True,termination='optimal')


def permutation_cycles(problem, parent_a, parent_b):
    _require(problem,parent_a); _require(problem,parent_b)
    locations = {s//4:c for c,s in enumerate(parent_a)}
    mapping = [locations[s//4] for s in parent_b]
    unseen = set(range(len(mapping))); cycles=[]
    while unseen:
        start = min(unseen); cycle=[]; cell=start
        while cell in unseen:
            unseen.remove(cell); cycle.append(cell); cell=mapping[cell]
        if cell != start: raise RuntimeError('Parent mapping was not a permutation')
        cycles.append(cycle)
    return cycles


def cycle_recombine(problem, parent_a, parent_b, *, seconds=1.0, seed=0,
                    should_stop=None):
    """Optimize whole alternating cycles with an exact binary CP-SAT objective.

    On a timeout, retain at least the better parent. 'optimal' refers solely to
    this two-parent neighborhood, never to the puzzle or other arrangements.
    """
    started=time.monotonic()
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    deadline=started+seconds
    cycles=permutation_cycles(problem,parent_a,parent_b)
    before=max(score(problem,parent_a),score(problem,parent_b))
    best=list(parent_a if score(problem,parent_a)>=score(problem,parent_b) else parent_b)
    active=[cycle for cycle in cycles if any(parent_a[c]!=parent_b[c] for c in cycle)]
    result={'board':best,'score':before,'reference_score':before,'cycles':len(cycles),
            'active_cycles':len(active),'optimal':False,'operator':'cycle-recombination'}
    def stopped(): return bool(should_stop and should_stop())
    if stopped(): return dict(result,termination='stopped')
    if not active: return dict(result,optimal=True,termination='optimal')
    if seconds==0: return dict(result,termination='timeout')
    if len(active)==1: return dict(result,optimal=True,termination='optimal')
    if time.monotonic()>=deadline: return dict(result,termination='timeout')
    from ortools.sat.python import cp_model
    if stopped(): return dict(result,termination='stopped')
    if time.monotonic()>=deadline: return dict(result,termination='timeout')
    model=cp_model.CpModel()
    choices=[model.new_bool_var('cycle_'+str(i)) for i in range(len(active))]
    component={c:i for i,cycle in enumerate(active) for c in cycle}
    terms=[]
    for a,b,sa,sb in grid_edges(problem.size):
        if stopped(): return dict(result,termination='stopped')
        if time.monotonic()-started >= seconds: return dict(result,termination='timeout')
        relevant=sorted({component[c] for c in (a,b) if c in component})
        if not relevant:
            terms.append(int(problem.faces[parent_a[a]][sa]==problem.faces[parent_a[b]][sb])); continue
        rows=[]
        for mask in range(1<<len(relevant)):
            bits=[(mask>>j)&1 for j in range(len(relevant))]
            assignment=dict(zip(relevant,bits))
            s1=(parent_b if assignment.get(component.get(a),0) else parent_a)[a]
            s2=(parent_b if assignment.get(component.get(b),0) else parent_a)[b]
            rows.append(bits+[int(problem.faces[s1][sa]==problem.faces[s2][sb])])
        match=model.new_bool_var('match_'+str(a)+'_'+str(b))
        model.add_allowed_assignments([choices[i] for i in relevant]+[match],rows)
        terms.append(match)
    model.maximize(sum(terms))
    prefer_b=score(problem,parent_b)>score(problem,parent_a)
    for var in choices: model.add_hint(var,int(prefer_b))
    model.add(sum(terms)>=before)
    if stopped(): return dict(result,termination='stopped')
    if time.monotonic()>=deadline: return dict(result,termination='timeout')
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=max(0,deadline-time.monotonic())
    solver.parameters.num_search_workers=1
    solver.parameters.random_seed=seed & 0x7FFFFFFF
    if stopped(): return dict(result,termination='stopped')
    if time.monotonic()>=deadline: return dict(result,termination='timeout')
    done=threading.Event();interrupted=threading.Event();errors=[]
    def watch():
        while not done.wait(.02):
            try:
                if stopped(): interrupted.set()
                if interrupted.is_set() or time.monotonic()>=deadline:
                    # Repeat through native startup, including deadline stops.
                    solver.stop_search()
            except BaseException as exc:
                errors.append(exc);interrupted.set();solver.stop_search()
    watcher=threading.Thread(target=watch,name='eternity-cycle-cancel',daemon=True);watcher.start()
    try: status=solver.solve(model)
    finally: done.set();watcher.join()
    if errors: raise RuntimeError('Cycle repair cancellation callback failed') from errors[0]
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        candidate=list(parent_a)
        for i,cycle in enumerate(active):
            if solver.value(choices[i]):
                for cell in cycle: candidate[cell]=parent_b[cell]
        actual=score(problem,candidate)
        if not structurally_valid(problem,candidate) or actual<before or actual!=round(solver.objective_value):
            raise RuntimeError('Cycle repair failed independent validation')
        result.update(board=candidate,score=actual)
    elif status in (cp_model.INFEASIBLE,cp_model.MODEL_INVALID):
        raise RuntimeError('Feasible parents lost by cycle model')
    return dict(result,optimal=status==cp_model.OPTIMAL,
                termination='stopped' if interrupted.is_set() or stopped() else 'optimal' if status==cp_model.OPTIMAL else 'timeout',
                branches=solver.num_branches)


def cached_parents(problem, database, *, limit=32):
    """Read a bounded selection from the local cache, with no cache mutations."""
    if type(limit) is not int or not 1<=limit<=256: raise ValueError('limit must be 1..256')
    database=Path(database)
    if not database.is_file(): return []
    result=[]
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True,timeout=1)) as db:
        for (blob,) in db.execute('SELECT placement FROM boards WHERE placement IS NOT NULL ORDER BY score DESC,public_sha LIMIT ?', (limit,)):
            if not isinstance(blob,bytes) or len(blob)!=2*problem.size**2: continue
            board=list(struct.unpack('<'+str(problem.size**2)+'H',blob))
            if structurally_valid(problem,board) and board not in result: result.append(board)
    return result



def solve(problem, reference, *, seconds=60, seed=0, workers=1, parents=None,
          should_stop=None, on_improvement=None, on_progress=None):
    """Interruptible portfolio of exact repair neighborhoods, not exhaustive DFS.

    Stops at the supplied time limit or cancellation. A timeout, plateau or
    exhausted parent neighborhood says nothing about full-puzzle feasibility.
    Parent preparation and backend imports consume the same wall budget.
    Callbacks and parent iterators must return promptly. At most 64 supplied
    parents are inspected, including duplicate or invalidly repeated entries.
    """
    start=time.monotonic()
    if workers != 1 or type(workers) is not int:
        raise ValueError('Matching repair uses one CPU worker')
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
        raise ValueError('seconds must be finite and nonnegative')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')
    for name,callback in (('should_stop',should_stop),('on_improvement',on_improvement),('on_progress',on_progress)):
        if callback is not None and not callable(callback): raise ValueError(name+' must be callable')
    _require(problem,reference)
    deadline=start+seconds;rng=random.Random(seed)
    best=list(reference);best_score=score(problem,best);initial=best_score
    bank=[best];seen=set();completed=0;cycle_calls=0;rounds=0
    max_score=len(grid_edges(problem.size));last_progress=float('-inf')
    def stopped(): return bool(should_stop and should_stop())
    def ended(): return stopped() or time.monotonic()>=deadline
    def accept(candidate):
        nonlocal best,best_score
        if not structurally_valid(problem,candidate): raise RuntimeError('Invalid repair result')
        value=score(problem,candidate)
        if value>best_score:
            best,best_score=list(candidate),value
            if on_improvement:
                on_improvement({'board':best.copy(),'score':value,'elapsed_seconds':time.monotonic()-start,
                                'backend':'exact-matching-and-cycle-repair','branches':None,'conflicts':None})
        return value
    if not ended():
        # islice bounds duplicate streams too; limiting the number of distinct
        # boards alone could consume an unbounded repeating parent iterator.
        for parent in itertools.islice(parents or (),64):
            if ended(): break
            _require(problem,parent)
            if parent not in bank: bank.append(list(parent))
            if len(bank)>=64: break
    # Cached parents can improve a stale local incumbent. That is reuse, not
    # discovery; the worker retains local-cache provenance for saved boards.
    for board in bank:
        if ended(): break
        accept(board)
    while not ended() and best_score<max_score:
        index=rounds%len(bank);parent=bank[index]
        cells=independent_cells(problem,seed=rng.randrange(2**32),
                                parity=(rounds//len(bank))%2 if rounds<2*len(bank) else None)
        key=(tuple(parent),tuple(cells))
        if key not in seen:
            repaired=assignment_repair(problem,parent,cells,should_stop=ended)
            if repaired['optimal']:
                completed+=1;seen.add(key)
                if len(seen)>4096: seen.clear()
                bank[index]=repaired['board'];accept(repaired['board'])
        rounds+=1
        if rounds%4==0 and len(bank)>1 and not ended():
            other=bank[rng.randrange(len(bank))]
            combined=cycle_recombine(problem,best,other,seconds=min(.25,max(0,deadline-time.monotonic())),
                                     seed=rng.randrange(2**31),should_stop=ended)
            cycle_calls+=1;accept(combined['board'])
            if combined['score']>score(problem,bank[index]): bank[index]=combined['board']
        now=time.monotonic()
        if on_progress and now-last_progress>=.25:
            last_progress=now
            on_progress({'phase':'matching_and_cycle_repair','neighborhoods':completed,'recombinations':cycle_calls,
                         'best_score':best_score,'elapsed_seconds':now-start,
                         'backend':'CPU matching + CP-SAT cycle choices'})
    termination='perfect_board_validated' if best_score==max_score else 'stopped' if stopped() else 'timeout'
    return {'outcome':'solved' if best_score==max_score else 'stopped' if termination=='stopped' else 'timeout',
            'termination':termination,'board':best,'score':best_score,'reference_score':initial,
            'improved':best_score>initial,'max_score':max_score,'neighborhoods':completed,
            'recombinations':cycle_calls,'elapsed_seconds':time.monotonic()-start,
            'proof_scope':'completed_independent_or_two_parent_neighborhoods_only',
            'coverage_verified':False,'branches':None,'conflicts':None}
