"""Bounded regional master with certified local rejection cuts.

The CP-SAT master allocates every physical piece to exactly one region, fixes
region capacities, and chooses the color of every cross-region constraint edge.
A local exact DFS supplies each region's interior or proves that particular
allocation/interface impossible. Only a completed local proof creates a cut.
No model, certificate, or counter is imported from disk; runs start fresh.
"""
from __future__ import annotations
from dataclasses import replace
import math
import threading
import time
from types import MappingProxyType
from exact_model import check_solution, constraint_edges, initial_domains, problem_hash


class _Limit(Exception):
    def __init__(self,outcome):self.outcome=outcome


def _integer(value,name,low,high):
    if type(value) is not int or not low<=value<=high:raise ValueError(f'Invalid {name}')
    return value


def _conserve_colors(model,problem,regions,memberships,internal,incident,crossvars,crosscolors,counts,tick,legal,coupling):
    """Regions count colored sides, with slack only on omitted in-board sides."""
    n=problem.size**2
    by_color={}
    unstable=set()
    for piece in range(n):
        signatures=[]
        for state in range(4*piece,4*piece+4):
            signature={}
            for color in problem.faces[state]:
                tick()
                if color:signature[color]=1+signature.get(color,0)
            signatures.append(signature)
        palette=set().union(*signatures)
        for color in sorted(palette):
            tick(4)
            value=signatures[0].get(color,0)
            if any(signature.get(color,0)!=value for signature in signatures[1:]):
                unstable.add(color)
            elif value:by_color.setdefault(color,[]).append((piece,value))
    # ExactProblem also supports non-rotational state tables. The identity
    # is only meaningful for a color invariant under the declared rotations.
    palette=sorted(by_color.keys()-unstable)
    indicator={}
    size=problem.size
    for r,cells in enumerate(regions):
        unpaired=0
        for c in cells:
            tick()
            y,x=divmod(c,size)
            unpaired+=(x>0)+(x+1<size)+(y>0)+(y+1<size)
        unpaired-=2*len(internal[r])+len(incident[r])
        supported={};internal_totals=[];loose_totals=[]
        if coupling=='edge_colors':
            for a,b,sa,sb in internal[r]:
                tick(1+len(legal[a])+len(legal[b]))
                possible={problem.faces[s][sa] for s in legal[a]} & {problem.faces[s][sb] for s in legal[b]}
                if not possible:
                    model.add_bool_or([]);counts['master_constraints']+=1
                    return
                if len(possible)==1:
                    supported.setdefault(next(iter(possible)),[]).append(1)
                    continue
                choices=[]
                for color in sorted(possible):
                    tick()
                    flag=model.new_bool_var(f'internal{a}_{b}_color{color}')
                    supported.setdefault(color,[]).append(flag);choices.append(flag)
                model.add_exactly_one(choices);counts['master_constraints']+=1
        for color in palette:
            supply=[];exposed=[]
            for piece,value in by_color[color]:
                tick()
                var=memberships.get((piece,r))
                if var is not None:supply.append(value*var)
            for edge,_,_ in incident[r]:
                colors=crosscolors[edge]
                tick(1+len(colors))
                if color not in colors:continue
                if len(colors)==1:exposed.append(1);continue
                key=edge,color
                if key not in indicator:
                    bit=model.new_bool_var(f'edge{edge}_color{color}')
                    model.add(crossvars[edge]==color).only_enforce_if(bit)
                    model.add(crossvars[edge]!=color).only_enforce_if(bit.Not())
                    indicator[key]=bit;counts['master_constraints']+=2
                exposed.append(indicator[key])
            if not supply and not exposed and coupling!='edge_colors':continue
            if coupling=='edge_colors':
                inside=sum(supported.get(color,()))
                tick(1+len(supported.get(color,())))
            else:
                inside=model.new_int_var(0,len(internal[r]),f'internal{r}_color{color}')
                internal_totals.append(inside)
            # Each omitted adjacency exposes a free half-edge on this side.
            loose=model.new_int_var(0,unpaired,f'loose{r}_color{color}') if unpaired else 0
            tick(1+len(supply)+len(exposed))
            model.add(sum(supply)==2*inside+sum(exposed)+loose)
            counts['master_constraints']+=1;counts['conservation_equations']+=1
            if unpaired:loose_totals.append(loose)
        if coupling=='counts' and internal[r]:
            tick(1+len(internal_totals))
            if unstable:model.add(sum(internal_totals)<=len(internal[r]))
            else:model.add(sum(internal_totals)==len(internal[r]))
            counts['master_constraints']+=1
        if coupling!='none' and unpaired:
            tick(1+len(loose_totals))
            if unstable:model.add(sum(loose_totals)<=unpaired)
            else:model.add(sum(loose_totals)==unpaired)
            counts['master_constraints']+=1


def solve(problem,seconds=30.,seed=0,workers=1,should_stop=None,on_progress=None,*,
          regions=None,max_work=100000,max_master_queries=64,regional_cuts=True,
          minimize_interfaces=False,color_conservation=False,allocation_cores=False,
          hall_capacity=False,cache_witnesses=False,local_oracle="dfs",color_coupling='none'):
    """Solve exactly the supplied edge scope, or return a bounded unknown.

    max_work caps explicit Python preparation, oracle and cut operations. Native
    master work has a shared wall deadline and is reported separately. Native
    branch counts are not equated with local candidate placement counts.
    Optional interface minimization keeps the exact piece pool and removes a
    boundary assumption only with a complete local infeasibility explanation.
    It reuses the original local DFS proof without another search. Explanation
    operations are included in oracle_work; explanation_work is a subset, not
    additional uncounted work. Cores are sufficient, not necessarily minimal.
    """
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid seconds')
    _integer(seed,'seed',0,2**32-1);_integer(workers,'workers',1,64)
    _integer(max_work,'max_work',0,10000000);_integer(max_master_queries,'master query cap',1,10000)
    if type(regional_cuts) is not bool:raise ValueError('regional_cuts must be boolean')
    if type(minimize_interfaces) is not bool:raise ValueError('minimize_interfaces must be boolean')
    for flag in (color_conservation,allocation_cores,hall_capacity,cache_witnesses):
        if type(flag) is not bool:raise ValueError('Invalid regional reasoning option')
    if color_coupling not in ('none','counts','edge_colors'):raise ValueError('Invalid regional color coupling')
    if color_coupling!='none' and not color_conservation:raise ValueError('Coupled colors require color conservation')
    if local_oracle not in ('dfs','sat'):raise ValueError('Invalid regional local oracle')
    if allocation_cores and not minimize_interfaces:
        raise ValueError('Allocation cores require interface explanations')
    if minimize_interfaces and not regional_cuts:raise ValueError('Interface minimization requires regional cuts')
    for callback in (should_stop,on_progress):
        if callback is not None and not callable(callback):raise ValueError('Invalid callback')
    p=replace(problem,faces=tuple(tuple(f) for f in problem.faces),domains=tuple(tuple(d) for d in problem.domains),
              fixed=MappingProxyType(dict(problem.fixed)),edges=None if problem.edges is None else tuple(tuple(e) for e in problem.edges))
    n=p.size**2
    if regions is None:regions=tuple(tuple(range(r*p.size,(r+1)*p.size)) for r in range(p.size))
    if not isinstance(regions,(tuple,list)) or not 1<=len(regions)<=n:raise ValueError('Invalid region partition')
    cleaned=[]
    for group in regions:
        if not isinstance(group,(tuple,list)) or not group or len(group)>n or any(type(c) is not int or not 0<=c<n for c in group):raise ValueError('Invalid region cells')
        cleaned.append(tuple(sorted(group)))
    if sorted(c for group in cleaned for c in group)!=list(range(n)):raise ValueError('Regions must partition every cell exactly once')
    regions=tuple(cleaned);where={c:r for r,g in enumerate(regions) for c in g}
    started=time.monotonic();deadline=started+seconds;phase='building';last_progress=started
    counts=dict(setup_work=0,oracle_work=0,cut_work=0,master_queries=0,master_branches=0,master_conflicts=0,
                local_queries=0,local_placements=0,local_failures=0,regional_cuts_added=0,full_cuts_added=0,
                allocation_literals=0,interface_variables=0,master_constraints=0,master_deterministic_time=0.,
                interface_literals_removed=0,interface_literals_in_cuts=0,
                explanation_work=0,extracted_cores=0,allocation_cores_added=0,
                conservation_equations=0,hall_inequalities=0,witness_cache_hits=0,
                local_sat_decisions=0,local_sat_conflicts=0,local_sat_propagations=0,local_sat_restarts=0)
    edges=tuple(constraint_edges(p));identity=problem_hash(p);native_status='NOT_RUN'
    def work():return counts['setup_work']+counts['oracle_work']+counts['cut_work']
    def result(outcome,board=None):
        return dict(outcome='timeout' if outcome in ('work_limit','query_limit') else outcome,termination=outcome,board=board,backend='regional-master',phase=phase,
                    branches=counts['local_placements']+counts['local_sat_decisions'],conflicts=counts['local_failures'],
                    elapsed_seconds=time.monotonic()-started,workers=1,requested_workers=workers,
                    complete=outcome in ('solved','infeasible'),proof_scope='supplied-problem-edges-and-domains',
                    problem_hash=identity,regions=[list(g) for g in regions],work=work(),
                    solver_status=native_status,regional_cuts=regional_cuts,minimize_interfaces=minimize_interfaces,
                    resume_capability='fresh-run-only',color_conservation=color_conservation,color_coupling=color_coupling,
                    allocation_cores=allocation_cores,hall_capacity=hall_capacity,cache_witnesses=cache_witnesses,local_oracle=local_oracle,
                    reasoning_counters_session=dict(counts,explicit_work=work()),**counts)
    def progress(force=False):
        nonlocal last_progress
        now=time.monotonic()
        if on_progress is not None and (force or now-last_progress>=.25):
            last_progress=now;on_progress(result('running'))
    def check():
        if should_stop is not None and should_stop():raise _Limit('stopped')
        now=time.monotonic()
        if now>=deadline:raise _Limit('timeout')
        if on_progress is not None and now-last_progress>=.25:progress()
    def tick(kind,amount=1):
        check()
        if work()+amount>max_work:raise _Limit('work_limit')
        counts[kind]+=amount
    def local(region,pool,boundary,explain=False):
        counts['local_queries']+=1;chosen=regions[region];pool=set(pool);domains={};reasons={}
        def combine(target,source):
            if explain:
                tick('oracle_work',len(source));counts['explanation_work']+=len(source)
                target.update(source)
        for c in chosen:
            allowed=[];reasons[c]=set()
            for s in legal[c]:
                tick('oracle_work')
                if s//4 not in pool:
                    if allocation_cores:
                        # Negative codes record a sufficient excluded-piece
                        # premise, disjoint from nonnegative interface indices.
                        tick('oracle_work');counts['explanation_work']+=1
                        reasons[c].add(~(s//4))
                    continue
                blocked=False
                for side,color,edge in boundary.get(c,()):
                    tick('oracle_work')
                    if p.faces[s][side]!=color:
                        # A single violated boundary equality suffices to
                        # explain removing this state. Pool exclusions are
                        # unconditional within this exact-pool local oracle.
                        blocked=True
                        if explain:
                            tick('oracle_work');counts['explanation_work']+=1
                            reasons[c].add(edge)
                        break
                if not blocked:allowed.append(s)
            if not allowed:return None,reasons[c]
            domains[c]=tuple(allowed)
        adjacent={c:[] for c in chosen}
        for a,b,sa,sb in internal[region]:adjacent[a].append((b,sa,sb));adjacent[b].append((a,sb,sa))
        assigned={};used=set()
        def search():
            if len(assigned)==len(chosen):return dict(assigned),set()
            options=[]
            for c in chosen:
                if c in assigned:continue
                available=[]
                for s in domains[c]:
                    tick('oracle_work')
                    if s//4 in used:continue
                    valid=True
                    for other,side,opposite in adjacent[c]:
                        tick('oracle_work')
                        if other in assigned and p.faces[s][side]!=p.faces[assigned[other]][opposite]:valid=False;break
                    if valid:available.append(s)
                # Given the current temporary placements, states ruled out by
                # uniqueness/adjacency need no boundary premise. The remaining
                # removed states have the recorded boundary explanations.
                if not available:return None,reasons[c]
                options.append((len(available),c,available))
            _,cell,candidates=min(options)
            core=set();combine(core,reasons[cell])
            for s in candidates:
                tick('oracle_work');counts['local_placements']+=1
                assigned[cell]=s;used.add(s//4)
                answer,child_core=search()
                if answer is not None:return answer,set()
                combine(core,child_core)
                used.remove(s//4);del assigned[cell]
            # All selected-cell alternatives are now resolved. The union is
            # sufficient for failure under the ancestor placements; at root,
            # no temporary placement assumptions remain.
            return None,core
        return search()
    local_solvers={}
    try:
        check();progress(True)
        # Expensive native import is lazy and consumes the same wall budget.
        from ortools.sat.python import cp_model
        check()
        # Charge the entire bounded raw-domain scan before filtering. Charging
        # first permits cancellation/caps without partially filtered proof data.
        for domain in p.domains:tick('setup_work',len(domain))
        legal=initial_domains(p)
        if any(not d for d in legal):return result('infeasible')
        eligible=[set() for _ in range(n)]
        for c,domain in enumerate(legal):
            for state in domain:tick('setup_work');eligible[state//4].add(where[c])
        if any(not choices for choices in eligible):return result('infeasible')
        model=cp_model.CpModel();allocation=[];memberships={}
        for piece,choices in enumerate(eligible):
            tick('setup_work');var=model.new_int_var_from_domain(cp_model.Domain.from_values(sorted(choices)),f'p{piece}')
            allocation.append(var)
            for r in sorted(choices):
                tick('setup_work');b=model.new_bool_var(f'p{piece}_r{r}');memberships[piece,r]=b
                model.add(var==r).only_enforce_if(b);model.add(var!=r).only_enforce_if(b.Not())
                counts['allocation_literals']+=1;counts['master_constraints']+=2
        for r,group in enumerate(regions):
            tick('setup_work');model.add(sum(b for (piece,rr),b in memberships.items() if rr==r)==len(group));counts['master_constraints']+=1
        crosscolors=[];cross=[];crossvars=[];incident=[[] for _ in regions];internal=[[] for _ in regions]
        for edge in edges:
            a,b,sa,sb=edge
            if where[a]==where[b]:internal[where[a]].append(edge);continue
            left=set();right=set()
            for s in legal[a]:tick('setup_work');left.add(p.faces[s][sa])
            for s in legal[b]:tick('setup_work');right.add(p.faces[s][sb])
            colors=sorted(left&right)
            if not colors:return result('infeasible')
            crosscolors.append(colors);i=len(cross);cross.append(edge);crossvars.append(model.new_int_var_from_domain(cp_model.Domain.from_values(colors),f'e{i}'))
            incident[where[a]].append((i,a,sa));incident[where[b]].append((i,b,sb));counts['interface_variables']+=1
        if color_conservation:
            _conserve_colors(model,p,regions,memberships,internal,incident,crossvars,crosscolors,counts,
                             lambda amount=1:tick('setup_work',amount),legal,color_coupling)
        if hall_capacity:
            for r,cells in enumerate(regions):
                allowed=[]
                for cell in cells:
                    tick('setup_work',len(legal[cell]))
                    allowed.append(frozenset(state//4 for state in legal[cell]))
                for group in dict.fromkeys(allowed):
                    required=0
                    for choices in allowed:
                        tick('setup_work',1+min(len(group),len(choices)))
                        required+=choices<=group
                    if required==len(cells):continue
                    tick('setup_work',1+len(group))
                    model.add(sum(memberships[piece,r] for piece in sorted(group))>=required)
                    counts['hall_inequalities']+=1;counts['master_constraints']+=1
        witness_cache={}
        if local_oracle!='dfs':from regional_sat_oracle import RegionalSAT
        while counts['master_queries']<max_master_queries:
            check();phase='master_search';progress(True)
            solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
            solver.parameters.random_seed=seed&0x7fffffff
            solver.parameters.max_time_in_seconds=max(.000001,deadline-time.monotonic())
            finished=threading.Event();failure=[];interrupted=[None]
            def watch():
                while not finished.wait(.02):
                    try:check();progress()
                    except _Limit as exc:interrupted[0]=exc.outcome;solver.stop_search()
                    except Exception as exc:failure.append(exc);solver.stop_search()
            watcher=threading.Thread(target=watch,name='eternity-regional-master-stop',daemon=True);watcher.start()
            try:status=solver.solve(model)
            finally:finished.set();watcher.join()
            if failure:raise failure[0]
            counts['master_queries']+=1;counts['master_branches']+=solver.num_branches;counts['master_conflicts']+=solver.num_conflicts
            counts['master_deterministic_time']+=solver.response_proto.deterministic_time
            native_status=solver.status_name(status)
            if status==cp_model.MODEL_INVALID:raise RuntimeError('Invalid regional master model')
            if status==cp_model.INFEASIBLE:return result('infeasible')
            if interrupted[0]:return result(interrupted[0])
            if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):return result('timeout')
            allocation_value=[solver.value(v) for v in allocation];color_value=[solver.value(v) for v in crossvars]
            board=[None]*n;failed=None;phase='regional_oracles';progress(True)
            for r,group in enumerate(regions):
                check();pool=[piece for piece,value in enumerate(allocation_value) if value==r]
                boundary={}
                for i,c,side in incident[r]:boundary.setdefault(c,[]).append((side,color_value[i],i))
                cache_key=None
                if cache_witnesses:
                    tick('oracle_work',1+len(pool)+len(incident[r]))
                    cache_key=(r,tuple(pool),tuple(color_value[i] for i,_,_ in incident[r]))
                if cache_key is not None and cache_key in witness_cache:
                    answer=witness_cache[cache_key];core=set();counts['witness_cache_hits']+=1
                else:
                    if local_oracle=='dfs':
                        answer,core=local(r,pool,boundary,minimize_interfaces)
                    else:
                        if r not in local_solvers:
                            local_solvers[r]=RegionalSAT(p,legal,group,internal[r],incident[r],
                                                         lambda n=1:tick('oracle_work',n),counts)
                        counts['local_queries']+=1
                        answer,core=local_solvers[r].query(pool,boundary,check,progress)
                    if cache_key is not None and answer is not None and len(witness_cache)<4096:
                        tick('oracle_work',1+len(answer));witness_cache[cache_key]=answer
                if answer is None:counts['local_failures']+=1;failed=(r,pool,core);break
                for c,s in answer.items():board[c]=s
            if failed is None:
                if not check_solution(p,board):raise RuntimeError('Regional composition failed independent full validation')
                return result('solved',board)
            r,pool,core=failed
            if regional_cuts:
                crossing=sorted({i for i,_,_ in incident[r]})
                if minimize_interfaces:
                    excluded={~literal for literal in core if literal<0}
                    core={literal for literal in core if literal>=0}
                    if allocation_cores:tick('cut_work',len(excluded)+len(core))
                    counts['interface_literals_removed']+=len(crossing)-len(core)
                    crossing=sorted(core);counts['extracted_cores']+=1
                counts['interface_literals_in_cuts']+=len(crossing)
                variables=[allocation[piece] for piece in pool]+[crossvars[i] for i in crossing]
                values=[r]*len(pool)+[color_value[i] for i in crossing]
                # If the proof needs fewer exclusions, block those absences
                # directly. Any assigned-piece variable omitted here is free.
                if allocation_cores and len(excluded)<len(pool):
                    variables=[memberships[piece,r] for piece in sorted(excluded)]+[crossvars[i] for i in crossing]
                    values=[0]*len(excluded)+[color_value[i] for i in crossing]
                    counts['allocation_cores_added']+=1
            else:variables=allocation+crossvars;values=allocation_value+color_value
            tick('cut_work',len(variables))
            # Exact capacity makes positive membership for every pool piece
            # equivalent to this region's whole physical-piece allocation.
            if variables:model.add_forbidden_assignments(variables,[values])
            else:model.add_bool_or([])
            counts['master_constraints']+=1
            counts['regional_cuts_added' if regional_cuts else 'full_cuts_added']+=1
        return result('query_limit')
    except _Limit as exc:return result(exc.outcome)
    finally:
        for engine in local_solvers.values():engine.close()
