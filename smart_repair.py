"""Offline exact repair neighborhoods; neither operator proves full infeasibility.

Independent-set matching follows Schaus/Deville (JFPC 2008). Cycle crossover
chooses whole components of two cell/piece permutations, retaining uniqueness.
"""
from __future__ import annotations
from dataclasses import replace
import math
import itertools
import random
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


def _assignment_from_matrix(problem, matrix, *, should_stop=None, bulk_arcs=True, structural_problem=None):
    """Match a device/CPU matrix; independently check all selected states/costs."""
    if type(bulk_arcs) is not bool: raise ValueError('bulk_arcs must be boolean')
    board=matrix['board'];cells=matrix['cells'];before=score(problem,board)
    baseline={'board':list(board),'score':before,'reference_score':before,
              'optimal':False,'neighborhood_size':len(cells),'operator':'independent-matching',
              'cost_backend':matrix['backend'],'cost_device':matrix['device']}
    def stopped(): return bool(should_stop and should_stop())
    if stopped(): return dict(baseline,termination='stopped')
    if not cells: return dict(baseline,optimal=True,termination='optimal')
    from ortools.graph.python import linear_sum_assignment
    if stopped(): return dict(baseline,termination='stopped')
    matching=linear_sum_assignment.SimpleLinearSumAssignment()
    states=matrix['states'];weights=matrix['weights'];n=len(cells)
    if states.shape!=(n,n) or weights.shape!=(n,n): raise RuntimeError('Invalid assignment matrix shape')
    add_bulk=getattr(matching,'add_arcs_with_cost',None)
    if bulk_arcs and callable(add_bulk):
        import numpy as np
        left,right=np.nonzero(states>=0)
        costs=-weights[left,right]
        # Bounded chunks preserve prompt cancellation and canonical row-major
        # arc ordering. Runtime failures propagate; only API absence falls back.
        for start in range(0,len(left),4096):
            if stopped(): return dict(baseline,termination='stopped')
            add_bulk(left[start:start+4096],right[start:start+4096],costs[start:start+4096])
        baseline['matching_arc_mode']='bulk'
    else:
        for i in range(n):
            if stopped(): return dict(baseline,termination='stopped')
            for j in range(n):
                if int(states[i,j])>=0: matching.add_arc_with_cost(i,j,-int(weights[i,j]))
        baseline['matching_arc_mode']='scalar'
    if stopped(): return dict(baseline,termination='stopped')
    outcome=matching.solve()
    if outcome!=matching.OPTIMAL: raise RuntimeError('Feasible parent lost by assignment model')
    candidate=list(board);chosen=set(cells);selected_weight=0
    for i,cell in enumerate(cells):
        j=matching.right_mate(i);state=int(states[i,j])
        if not 0<=state<len(problem.faces) or state//4!=board[cells[j]]//4:
            raise RuntimeError('Assignment matrix returned wrong physical piece')
        candidate[cell]=state;selected_weight+=int(weights[i,j])
    actual=score(problem,candidate)
    fixed_score=sum(problem.faces[board[a]][sa]==problem.faces[board[b]][sb]
                    for a,b,sa,sb in grid_edges(problem.size) if a not in chosen and b not in chosen)
    valid=structurally_valid(problem,candidate) if structural_problem is None else check_solution(structural_problem,candidate)
    if not valid or actual<before or actual!=fixed_score+selected_weight:
        raise RuntimeError('Assignment repair failed independent validation')
    return dict(baseline,board=candidate,score=actual,optimal=True,termination='optimal')


def assignment_repair(problem, board, cells, *, should_stop=None, bulk_arcs=True):
    """Exact CPU maximum-score permutation/rotation of a nonadjacent piece pool."""
    from gpu_assignment import AssignmentCancelled,cpu_cost_matrix
    if type(bulk_arcs) is not bool: raise ValueError('bulk_arcs must be boolean')
    _require(problem,board)
    cells=list(cells)
    try: matrix=cpu_cost_matrix(problem,board,cells,should_stop=should_stop)
    except AssignmentCancelled:
        return {'board':list(board),'score':score(problem,board),'reference_score':score(problem,board),
                'optimal':False,'neighborhood_size':len(cells),'operator':'independent-matching',
                'termination':'stopped','cost_backend':'cpu','cost_device':'CPU'}
    return _assignment_from_matrix(problem,matrix,should_stop=should_stop,bulk_arcs=bulk_arcs)


def assignment_repair_batch(problem, neighborhoods, *, evaluator, should_stop=None, bulk_arcs=True,
                            reuse_validation=True):
    """Bounded GPU costs and exact CPU matchings against an immutable snapshot.

    Full board legality and scores are checked for every input and output. Only
    rebuilding/validating the unchanged problem definition is reused. Source
    mutations before, during or after a callback are rejected before returning.
    reuse_validation=False retains JSON identity and repeated model validation
    for bounded ablations; it never weakens the board checks.
    """
    if type(bulk_arcs) is not bool: raise ValueError('bulk_arcs must be boolean')
    if type(reuse_validation) is not bool: raise ValueError('reuse_validation must be boolean')
    if reuse_validation:
        compatible=evaluator.matches_problem(problem)
    else:
        from exact_model import problem_hash
        compatible=problem_hash(problem)==problem_hash(evaluator.problem)
    if not compatible: raise ValueError('Assignment evaluator belongs to a different problem')
    matrices=evaluator.evaluate(neighborhoods,reuse_validation=reuse_validation)
    if not evaluator.matches_problem(problem): raise ValueError('Assignment problem changed during matrix evaluation')
    trusted=evaluator.problem
    structure=evaluator.structural_problem if reuse_validation else None
    results=[_assignment_from_matrix(trusted,matrix,should_stop=should_stop,bulk_arcs=bulk_arcs,
                                     structural_problem=structure) for matrix in matrices]
    if not evaluator.matches_problem(problem): raise ValueError('Assignment problem changed during matching')
    return results


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
    component={c:i for i,cycle in enumerate(active) for c in cycle}
    unary=[[0,0] for _ in active];pairs={};constant=0;edge_tables=[]
    for a,b,sa,sb in grid_edges(problem.size):
        if stopped(): return dict(result,termination='stopped')
        if time.monotonic()>=deadline: return dict(result,termination='timeout')
        relevant=sorted({component[c] for c in (a,b) if c in component})
        if not relevant:
            constant+=int(problem.faces[parent_a[a]][sa]==problem.faces[parent_a[b]][sb]);continue
        rows=[]
        for bits in itertools.product((0,1),repeat=len(relevant)):
            assignment=dict(zip(relevant,bits))
            s1=(parent_b if assignment.get(component.get(a),0) else parent_a)[a]
            s2=(parent_b if assignment.get(component.get(b),0) else parent_a)[b]
            rows.append(list(bits)+[int(problem.faces[s1][sa]==problem.faces[s2][sb])])
        edge_tables.append((a,b,relevant,rows))
        if len(relevant)==1:
            for bit in (0,1): unary[relevant[0]][bit]+=rows[bit][-1]
        else:
            pair=pairs.setdefault(tuple(relevant),[0,0,0,0])
            for index,row in enumerate(rows): pair[index]+=row[-1]
    if stopped(): return dict(result,termination='stopped')
    if time.monotonic()>=deadline: return dict(result,termination='timeout')
    from binary_pairwise import maximize
    binary=maximize(unary,{key:tuple(value) for key,value in pairs.items()},
                    seconds=min(.05,max(0,deadline-time.monotonic())),should_stop=should_stop)
    if binary['optimal']:
        candidate=list(parent_a)
        for i,cycle in enumerate(active):
            if binary['assignment'][i]:
                for cell in cycle: candidate[cell]=parent_b[cell]
        actual=score(problem,candidate)
        if not structurally_valid(problem,candidate) or actual<before or actual!=constant+binary['value']:
            raise RuntimeError('Cycle component fast path failed independent validation')
        return dict(result,board=candidate,score=actual,optimal=True,termination='optimal',
                    branches=0,cycle_backend='component-dp-and-enumeration',
                    tree_components=binary.get('tree_components',0),
                    enumerated_components=binary.get('enumerated_components',0))
    if stopped(): return dict(result,termination='stopped')
    if time.monotonic()>=deadline: return dict(result,termination='timeout')
    from ortools.sat.python import cp_model
    if stopped(): return dict(result,termination='stopped')
    if time.monotonic()>=deadline: return dict(result,termination='timeout')
    model=cp_model.CpModel()
    choices=[model.new_bool_var('cycle_'+str(i)) for i in range(len(active))]
    terms=[constant]
    for a,b,relevant,rows in edge_tables:
        if stopped(): return dict(result,termination='stopped')
        if time.monotonic()>=deadline: return dict(result,termination='timeout')
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
                branches=solver.num_branches,cycle_backend='cp-sat')


def cached_parents(problem, database, *, limit=32, seed=0, should_stop=None, **options):
    """Validated, diverse local parents; the cache is always opened read-only."""
    from repair_parents import cached_parents as select_parents
    return select_parents(problem, database, limit=limit, seed=seed,
                          should_stop=should_stop, **options)


def solve(problem, reference, *, seconds=60, seed=0, workers=1, parents=None,
          should_stop=None, on_improvement=None, on_progress=None,
          gpu_backend=None, gpu_batch_size=16, gpu_cache_dir=None, gpu_allow_cpu_fallback=False,
          bulk_arcs=True, diversify=True, plateau_rounds=8,
          region_repair_enabled=True, region_every=8, region_seconds=.25):
    """Bounded structured repair with diversification and score-target regions.

    Only working parents may lose score. The independently checked best-ever
    board is copied separately and callbacks receive only strict improvements.
    A failed region applies only to its frozen complement, never the full puzzle.
    Optional switches support reproducible ablations; all three operators are
    enabled in the normal matching and GPU-matching worker modes.
    """
    from collections import deque
    start=time.monotonic()
    if workers != 1 or type(workers) is not int:
        raise ValueError('Matching repair uses one CPU worker')
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
        raise ValueError('seconds must be finite and nonnegative')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')
    for name,callback in (('should_stop',should_stop),('on_improvement',on_improvement),('on_progress',on_progress)):
        if callback is not None and not callable(callback): raise ValueError(name+' must be callable')
    for name,value in (('bulk_arcs',bulk_arcs),('diversify',diversify),
                       ('region_repair_enabled',region_repair_enabled),
                       ('gpu_allow_cpu_fallback',gpu_allow_cpu_fallback)):
        if type(value) is not bool: raise ValueError(name+' must be boolean')
    for name,value in (('plateau_rounds',plateau_rounds),('region_every',region_every)):
        if type(value) is not int or not 1<=value<=1024: raise ValueError(name+' must be 1..1024')
    if type(region_seconds) not in (int,float) or not math.isfinite(region_seconds) or not 0<region_seconds<=5:
        raise ValueError('region_seconds must be positive and at most 5')
    if gpu_backend not in (None,'auto','cuda','opencl'):
        raise ValueError('gpu_backend must be None, auto, cuda or opencl')
    if type(gpu_batch_size) is not int or not 1<=gpu_batch_size<=64:
        raise ValueError('gpu_batch_size must be 1..64 independent neighborhoods')
    _require(problem,reference)
    deadline=start+seconds;rng=random.Random(seed)
    best=list(reference);best_score=score(problem,best);initial=best_score
    bank=[best.copy()];seen=set();completed=0;cycle_calls=0;rounds=0;batches=0
    counts=dict(diversification_attempts=0,diversifications=0,tabu_rejections=0,
                region_calls=0,region_improvements=0,region_seconds=0.)
    max_score=len(grid_edges(problem.size));last_progress=float('-inf')
    def stopped(): return bool(should_stop and should_stop())
    def ended(): return stopped() or time.monotonic()>=deadline
    evaluator=None
    if gpu_backend is not None:
        from gpu_assignment import AssignmentEvaluator
        evaluator=AssignmentEvaluator(problem,backend=gpu_backend,cache_dir=gpu_cache_dir,
                                      allow_cpu_fallback=gpu_allow_cpu_fallback,
                                      batch_size=gpu_batch_size,should_stop=ended)
    def cost_stats():
        return evaluator.stats() if evaluator is not None else {'cost_backend':'cpu','cost_device':'CPU',
                'cost_batches':0,'cost_pairs':0,'cost_kernel_seconds':0.,'cost_fallback_reason':None}
    def accept(candidate):
        nonlocal best,best_score
        if not structurally_valid(problem,candidate): raise RuntimeError('Invalid repair result')
        value=score(problem,candidate)
        if value>best_score:
            best,best_score=list(candidate),value
            if on_improvement:
                on_improvement({'board':best.copy(),'score':value,'elapsed_seconds':time.monotonic()-start,
                                'backend':'structured-repair-portfolio','branches':None,'conflicts':None})
        return value
    if not ended():
        for parent in itertools.islice(parents or (),64):
            if ended(): break
            _require(problem,parent)
            if parent not in bank: bank.append(list(parent))
            if len(bank)>=64: break
    for board in bank:
        if ended(): break
        accept(board)
    bank_scores=[score(problem,b) for b in bank]
    peaks=list(bank_scores);stalls=[0]*len(bank)
    recent=[deque([tuple(b)],maxlen=12) for b in bank]
    diversifier=None
    def progress(phase='matching_and_cycle_repair', force=False):
        nonlocal last_progress
        now=time.monotonic()
        if on_progress and (force or now-last_progress>=.25):
            last_progress=now
            on_progress({'phase':phase,'neighborhoods':completed,'recombinations':cycle_calls,
                         'best_score':best_score,'elapsed_seconds':now-start,'parent_count':len(bank),
                         'backend':'GPU costs + CPU matching, cycles and regions'
                         if evaluator is not None and evaluator.backend in ('cuda','opencl')
                         else 'CPU matching, cycles and regions',**counts,**cost_stats()})
    while not ended() and best_score<max_score:
        pending=[];pending_keys=set();touched=set();advanced=set();batches+=1
        for _ in range(gpu_batch_size if evaluator is not None else 1):
            if ended(): break
            index=rounds%len(bank);parent=bank[index];touched.add(index)
            cells=independent_cells(problem,seed=rng.randrange(2**32),
                                    parity=(rounds//len(bank))%2 if rounds<2*len(bank) else None)
            key=(tuple(parent),tuple(cells));rounds+=1
            if key not in seen and key not in pending_keys:
                pending.append((index,parent,cells,key));pending_keys.add(key)
        def adopt(index,candidate):
            value=accept(candidate)
            key=tuple(candidate)
            if key==tuple(bank[index]): return
            # A strict new local record overrides tabu (aspiration). Tabu only
            # steers working parents; it never suppresses a verified global best.
            if diversify and key in recent[index] and value<=peaks[index]:
                counts['tabu_rejections']+=1;return
            if value>=bank_scores[index]:
                bank[index]=list(candidate);bank_scores[index]=value;recent[index].append(key)
                if value>peaks[index]:
                    peaks[index]=value;advanced.add(index)
        repaired_list=[]
        if pending:
            if evaluator is None:
                repaired_list=[assignment_repair(problem,pending[0][1],pending[0][2],should_stop=ended,bulk_arcs=bulk_arcs)]
            else:
                from gpu_assignment import AssignmentCancelled
                try:
                    repaired_list=assignment_repair_batch(problem,[(parent,cells) for _,parent,cells,_ in pending],
                                                         evaluator=evaluator,should_stop=ended,bulk_arcs=bulk_arcs)
                except AssignmentCancelled: break
            for (index,parent,cells,key),repaired in zip(pending,repaired_list):
                if repaired['optimal']:
                    completed+=1;seen.add(key)
                    if len(seen)>4096: seen.clear()
                    adopt(index,repaired['board'])
        if rounds%4==0 and len(bank)>1 and not ended() and best_score<max_score:
            index=rng.randrange(len(bank))
            combined=cycle_recombine(problem,best,bank[index],seconds=min(.25,max(0,deadline-time.monotonic())),
                                     seed=rng.randrange(2**31),should_stop=ended)
            cycle_calls+=1;adopt(index,combined['board']);touched.add(index)
        # All pending GPU snapshots are consumed before any intentional worsening.
        for index in sorted(touched):
            stalls[index]=0 if index in advanced else stalls[index]+1
            if not diversify or stalls[index]<plateau_rounds or ended() or best_score==max_score: continue
            if diversifier is None:
                from repair_diversify import Diversifier
                diversifier=Diversifier(problem)
            counts['diversification_attempts']+=1
            changed=diversifier.perturb(bank[index],seed=rng.randrange(2**32),
                        moves=2+(counts['diversification_attempts']%7),
                        forbidden_boards=recent[index],should_stop=ended)
            stalls[index]=0
            if changed['changed']:
                counts['diversifications']+=1
                bank[index]=list(changed['board']);bank_scores[index]=accept(bank[index])
                recent[index].append(tuple(bank[index]))
                peaks[index]=max(peaks[index],bank_scores[index])
        if region_repair_enabled and batches%region_every==0 and not ended() and best_score<max_score:
            # At most one short region per interval and about 20% of elapsed
            # time in this CPU operator, so GPU matching retains its role.
            allowance=min(region_seconds, deadline-time.monotonic(),
                          max(0., .2*(time.monotonic()-start)-counts['region_seconds']))
            if allowance>=.02:
                from repair_regions import select_region, region_repair
                region_started=time.monotonic();region_deadline=region_started+allowance
                def region_stop(): return ended() or time.monotonic()>=region_deadline
                cells=select_region(problem,best,seed=rng.randrange(2**32),
                                    region_size=8+4*(counts['region_calls']%4),
                                    donor_count=2+2*((counts['region_calls']//4)%3),should_stop=region_stop)
                if cells and not region_stop():
                    counts['region_calls']+=1
                    progress('connected_region_repair',force=True)
                    result=region_repair(problem,best,cells,seconds=max(0.,region_deadline-time.monotonic()),
                                         seed=rng.randrange(2**31),should_stop=region_stop)
                    previous=best_score;accept(result['board'])
                    if best_score>previous:
                        counts['region_improvements']+=1
                        # Feed a regional improvement back into GPU matching.
                        index=min(range(len(bank)),key=bank_scores.__getitem__)
                        bank[index]=best.copy();bank_scores[index]=best_score
                        peaks[index]=max(peaks[index],best_score);stalls[index]=0
                        recent[index].append(tuple(best))
                counts['region_seconds']+=time.monotonic()-region_started
        progress()
    termination='perfect_board_validated' if best_score==max_score else 'stopped' if stopped() else 'timeout'
    return {'outcome':'solved' if best_score==max_score else 'stopped' if termination=='stopped' else 'timeout',
            'termination':termination,'board':best,'score':best_score,'reference_score':initial,
            'improved':best_score>initial,'max_score':max_score,'neighborhoods':completed,
            'recombinations':cycle_calls,'elapsed_seconds':time.monotonic()-start,'parent_count':len(bank),
            'proof_scope':'completed_repair_neighborhoods_with_frozen_complements_only',
            'coverage_verified':False,'branches':None,'conflicts':None,**counts,**cost_stats()}
