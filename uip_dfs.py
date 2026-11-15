"""Finite-domain placement DFS with explicit implication reasons and first-UIP learning.

A signed placement literal denotes choosing or excluding a physical orientation
in a cell. Every inference carries a clause over these original literals. The
chronological ablation uses identical propagation without learning a clause.
"""
import math
import time
from exact_model import check_solution, constraint_edges, initial_domains


class _Limit(Exception):
    pass


def solve(problem,seconds=30.,seed=0,workers=1,should_stop=None,on_progress=None,*,
          learning=True,max_work=2500000,max_clauses=8192,include_proofs=False,channels=False,ordering="baseline",trace_conflicts=False):
    if type(seconds) not in (float,int) or not math.isfinite(seconds) or seconds<0:
        raise ValueError('Invalid seconds')
    if type(trace_conflicts) is not bool or type(channels) is not bool or type(learning) is not bool or type(include_proofs) is not bool or type(max_work) is not int or max_work<0:
        raise ValueError('Invalid UIP policy')
    if type(max_clauses) is not int or not 0<=max_clauses<=65536 or workers!=1:
        raise ValueError('Invalid clause cap or worker count')
    if ordering not in ('baseline','density','adaptive') or (ordering=='adaptive' and not channels):
        raise ValueError('Adaptive branching requires broad channels')
    start=time.perf_counter()
    counts=dict(work=0,branches=0,positive_decisions=0,conflicts=0,inferences=0,
                setup_steps=0,group_steps=0,clause_steps=0,analysis_steps=0,
                rollback_steps=0,ordering_steps=0,learned_clauses=0,backjumped_levels=0)
    def tick(name,n=1):
        if should_stop and should_stop():raise _Limit('stopped')
        if time.perf_counter()-start>=seconds:raise _Limit('timeout')
        if counts['work']+n>max_work:raise _Limit('work_limit')
        counts['work']+=n
        counts[name]+=n
    n=problem.size**2
    atoms=[None]
    learned=[]
    conflict_traces=[]
    observations={}
    board=None
    outcome=termination='timeout'
    try:
        tick('setup_steps',0)
        initial=initial_domains(problem)
        cells=[[] for _ in initial]
        pieces=[[] for _ in initial]
        for c,d in enumerate(initial):
            for s in d:
                tick('setup_steps')
                v=len(atoms);atoms.append((c,s))
                cells[c].append(v);pieces[s//4].append(v)
        groups=cells+pieces
        branch_groups=[('cell',g) for g in cells]
        placement_limit=len(atoms)
        arcs=[]
        for a,b,sa,sb in constraint_edges(problem):
            for left,right,side,facing in ((a,b,sa,sb),(b,a,sb,sa)):
                by_color={}
                for v in cells[right]:
                    tick('setup_steps')
                    s=atoms[v][1]
                    by_color.setdefault(problem.faces[s][facing],[]).append(v)
                for v in cells[left]:
                    s=atoms[v][1]
                    supports=by_color.get(problem.faces[s][side],())
                    tick('setup_steps',len(supports)+1)
                    arcs.append(tuple([-v]+[t for t in supports if atoms[t][1]//4!=s//4]))
        if channels:
            def channel(name, alternatives):
                tick('setup_steps', 1+len(alternatives))
                if len(alternatives)==1:return alternatives[0]
                v=len(atoms)
                atoms.append(name)
                arcs.append(tuple([-v]+alternatives))
                arcs.extend((-x,v) for x in alternatives)
                return v
            for piece,placements in enumerate(pieces):
                at_cells={}
                for v in placements:
                    tick('setup_steps')
                    at_cells.setdefault(atoms[v][0],[]).append(v)
                rows={}
                for c,options in at_cells.items():
                    v=channel(('piece',c,piece),options)
                    rows.setdefault(c//problem.size,[]).append(v)
                locations=[v for options in rows.values() for v in options]
                groups.append(locations)
                branch_groups.append(('piece',locations))
                row_group=[channel(('row',r,piece),options) for r,options in rows.items()]
                groups.append(row_group)
                branch_groups.append(('row',row_group))
            for edge,(a,b,sa,sb) in enumerate(constraint_edges(problem)):
                left={};right={}
                for c,side,sets in ((a,sa,left),(b,sb,right)):
                    for v in cells[c]:
                        tick('setup_steps')
                        sets.setdefault(problem.faces[atoms[v][1]][side],[]).append(v)
                colors=[]
                for color in sorted(left.keys()|right.keys()):
                    options=left.get(color,[])
                    # Keep an explicit shared atom even for a singleton left end.
                    v=len(atoms);atoms.append(('color',edge,color))
                    colors.append(v)
                    for options in (options,right.get(color,[])):
                        tick('setup_steps',1+len(options))
                        arcs.append(tuple([-v]+options));arcs.extend((-x,v) for x in options)
                groups.append(colors)
                branch_groups.append(('color',colors))
        adjacency=[[] for _ in cells]
        for a,b,sa,sb in constraint_edges(problem):
            adjacency[a].append((b,sa,sb))
            adjacency[b].append((a,sb,sa))
        value=[0]*len(atoms)
        level=[0]*len(atoms)
        reason=[None]*len(atoms)
        trail=[];levels=[];frames=[]
        def assign(lit,why):
            v=abs(lit)
            assert not value[v]
            value[v]=1 if lit>0 else -1
            level[v]=len(levels)
            reason[v]=why
            trail.append(lit)
            if why is not None:counts['inferences']+=1

        def rollback(to):
            while len(levels)>to:
                cut=levels.pop()
                while len(trail)>cut:
                    tick('rollback_steps')
                    v=abs(trail.pop())
                    value[v]=0;level[v]=0;reason[v]=None

        def propagate():
            while True:
                old=len(trail)
                for group in groups:
                    tick('group_steps',len(group)+1)
                    chosen=None;undecided=[]
                    for v in group:
                        if value[v]>0:
                            if chosen is not None:return (-chosen,-v)
                            chosen=v
                        elif not value[v]:undecided.append(v)
                    if chosen is not None:
                        for v in undecided:assign(-v,(-chosen,-v))
                    elif not undecided:return tuple(group)
                    elif len(undecided)==1:assign(undecided[0],tuple(group))
                for database in (arcs,learned):
                    for clause in database:
                        possible=None
                        for lit in clause:
                            tick('clause_steps')
                            v=value[abs(lit)]
                            if v==(1 if lit>0 else -1):break
                            if not v:
                                if possible is not None:break
                                possible=lit
                        else:
                            if possible is None:return clause
                            assign(possible,clause)
                if old==len(trail):return None

        def analyze(conflict):
            current=set(conflict)
            tick('analysis_steps',len(current))
            if not current:return None
            highest=max(level[abs(l)] for l in current)
            if not highest:return None
            if len(levels)>highest:rollback(highest)
            at_level=sum(level[abs(l)]==highest for l in current)
            for assigned in reversed(trail):
                if at_level<=1:break
                tick('analysis_steps')
                if -assigned not in current or level[abs(assigned)]!=highest:continue
                antecedent=reason[abs(assigned)]
                if antecedent is None:raise RuntimeError('A decision cannot resolve itself')
                current.remove(-assigned)
                at_level-=1
                for lit in antecedent:
                    tick('analysis_steps')
                    if lit!=assigned and lit not in current:
                        if -lit in current:raise RuntimeError('Tautological conflict')
                        current.add(lit)
                        at_level+=int(level[abs(lit)]==highest)
            if at_level!=1:raise RuntimeError('Conflict analysis did not reach the first UIP')
            asserting=next(l for l in current if level[abs(l)]==highest)
            jump=max((level[abs(l)] for l in current if l!=asserting),default=0)
            return tuple(sorted(current)),asserting,jump

        def decision(lit):
            counts['branches']+=1
            counts['positive_decisions']+=int(lit>0)
            levels.append(len(trail));assign(lit,None)

        last_observation=None
        while True:
            conflict=propagate()
            if last_observation is not None:
                kind,before=last_observation
                tick('ordering_steps',placement_limit)
                after=sum(value[v]==0 for v in range(1,placement_limit))
                contraction=1.0 if conflict is not None else (before-after)/max(1,before)
                total,seen=observations.get(kind,(0.,0))
                observations[kind]=(total+contraction,seen+1)
                last_observation=None
            if conflict is not None:
                counts['conflicts']+=1
                if trace_conflicts and len(conflict_traces)<128:
                    tick('analysis_steps',len(trail))
                    path=[atoms[l] for l in trail if l>0 and reason[l] is None and l<placement_limit]
                    if path:conflict_traces.append(path)
                if learning:
                    analyzed=analyze(conflict)
                    if analyzed is None:
                        outcome=termination='infeasible'
                        break
                    clause,asserting,jump=analyzed
                    if len(learned)>=max_clauses:raise _Limit('clause_limit')
                    learned.append(clause);counts['learned_clauses']+=1
                    counts['backjumped_levels']+=max(0,len(levels)-jump-1)
                    rollback(jump)
                    assign(asserting,clause)
                else:
                    while frames:
                        lit=frames.pop()
                        rollback(len(frames))
                        if lit>0:
                            frames.append(-lit);decision(-lit)
                            break
                    else:
                        outcome=termination='infeasible'
                        break
                continue
            chosen=0;smallest=len(atoms)
            for group in cells:
                tick('ordering_steps',len(group)+1)
                if any(value[v]>0 for v in group):continue
                options=[v for v in group if not value[v]]
                if len(options)<smallest:chosen=options[0];smallest=len(options)
            selected_kind='cell'
            if chosen and ordering=='density':
                c=atoms[chosen][0]
                # Count compatible values of each incident adjacency given
                # this placement. Product of independent support fractions
                # ranks local completions; it cannot delete a real choice.
                def density(v):
                    s=atoms[v][1];prob=1.
                    for other,side,facing in adjacency[c]:
                        allowed=matching=0
                        for t in cells[other]:
                            tick('ordering_steps')
                            if value[t]<0:continue
                            allowed+=1
                            state=atoms[t][1]
                            matching+=int(s//4!=state//4 and problem.faces[s][side]==problem.faces[state][facing])
                        prob*=matching/max(1,allowed)
                    return prob
                chosen=max((v for v in cells[c] if not value[v]),key=density)
            elif chosen and ordering=='adaptive':
                candidates={}
                for kind,group in branch_groups:
                    tick('ordering_steps',len(group)+1)
                    if any(value[v]>0 for v in group):continue
                    free=[v for v in group if not value[v]]
                    if not 2<=len(free)<=smallest+1:continue
                    if kind not in candidates or len(free)<candidates[kind][0]:
                        candidates[kind]=(len(free),free[0])
                def rating(kind):
                    total,seen=observations.get(kind,(0.,0))
                    return (total/seen+1/math.sqrt(seen) if seen else 3., -candidates[kind][0])
                selected_kind=max(candidates,key=rating)
                chosen=candidates[selected_kind][1]
                tick('ordering_steps',placement_limit)
                before=sum(value[v]==0 for v in range(1,placement_limit))
                last_observation=(selected_kind,before)
            if not chosen:
                board=[next(atoms[v][1] for v in group if value[v]>0) for group in cells]
                if not check_solution(problem,board):raise RuntimeError('UIP search returned invalid witness')
                outcome=termination='solved'
                break
            if not learning:frames.append(chosen)
            decision(chosen)
    except _Limit as exc:
        termination=str(exc)
        outcome='stopped' if termination=='stopped' else 'timeout'
    result=dict(outcome=outcome,termination=termination,board=board,backend='first-uip-placement-dfs',
                elapsed_seconds=time.perf_counter()-start,learning=learning,channels=channels,ordering=ordering,branch_type_observations=observations,workers=1,
                learned_channel_clauses=sum(any(len(atoms[abs(l)])>2 for l in c) for c in learned),**counts)
    if trace_conflicts:result['conflict_traces']=conflict_traces
    if include_proofs:
        result['atoms']=atoms
        result['clauses']=learned
    if on_progress:on_progress(result)
    return result
