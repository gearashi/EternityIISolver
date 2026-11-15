"""Bounded exact completion of independent residual cell/piece components.

An edge or a shared eligible physical piece joins components. Only a complete
search can prove failure. Interrupted local searches never alter caller domains.
"""
from dataclasses import replace
from types import MappingProxyType
from exact_model import constraint_edges, check_solution, frame_ok


def _bits(mask):
    while mask:
        bit=mask & -mask;mask-=bit;yield bit.bit_length()-1


class _Limit(Exception): pass


def solve_components(problem, domains, *, max_work=20000, max_component_cells=24,
                     should_stop=None, decompose=True):
    """Return a full witness, a scoped failure, or an inconclusive result.

    decompose=False is an ablation using the same local propagation and ordering.
    Work counts domain-state visits, compatibility tests, graph incidences and
    candidate placements separately and under one conservative operation cap.
    """
    if type(max_work) is not int or not 0<=max_work<=1000000: raise ValueError('Invalid component work cap')
    if type(max_component_cells) is not int or not 1<=max_component_cells<=256: raise ValueError('Invalid component size cap')
    if type(decompose) is not bool: raise ValueError('decompose must be boolean')
    if should_stop is not None and not callable(should_stop): raise ValueError('Invalid stop callback')
    n=problem.size**2
    if len(domains)!=n or any(type(d) is not int or d<0 or d>>len(problem.faces) for d in domains):
        raise ValueError('Expected bounded state bitsets')
    original=tuple(domains)
    p=replace(problem,faces=tuple(tuple(f) for f in problem.faces),
              domains=tuple(tuple(d) for d in problem.domains),fixed=MappingProxyType(dict(problem.fixed)),
              edges=None if problem.edges is None else tuple(tuple(e) for e in problem.edges))
    counters=dict(candidate_placements=0,domain_state_visits=0,compatibility_checks=0,graph_incidences=0)
    groups=[];completed=0;board=None
    def tick(kind,amount=1):
        if should_stop is not None and should_stop(): raise _Limit('stopped')
        if sum(counters.values())+amount>max_work: raise _Limit('work_limit')
        counters[kind]+=amount
    def result(outcome):
        return dict(outcome=outcome,board=board,complete=outcome in ('solved','infeasible'),
                    component_count=len(groups),component_sizes=[len(g) for g in groups],
                    completed_components=completed,work=sum(counters.values()),**counters,
                    proof_scope='supplied_problem_and_exact_residual_domains',decompose=decompose)
    try:
        tick('domain_state_visits',0)
        allowed=[]
        for c,d in enumerate(original):
            legal=set(p.domains[c]);kept=0
            for s in _bits(d):
                tick('domain_state_visits')
                if s not in legal: raise ValueError('Residual states escape supplied domain')
                if frame_ok(p.size,c,p.faces[s]) and (c not in p.fixed or p.fixed[c]==s): kept|=1<<s
            allowed.append(kept)
        edges=tuple(constraint_edges(p));neighbors=[[] for _ in range(n)]
        for a,b,sa,sb in edges:
            tick('graph_incidences');neighbors[a].append((b,sa,sb));neighbors[b].append((a,sb,sa))
        def propagate(ds,cells):
            cells=tuple(sorted(cells))
            while True:
                changed=False;owners={}
                if any(ds[c]==0 for c in cells): return False
                for c in cells:
                    pieces=set()
                    for s in _bits(ds[c]): tick('domain_state_visits');pieces.add(s//4)
                    if len(pieces)==1:
                        piece=next(iter(pieces))
                        if piece in owners and owners[piece]!=c:return False
                        owners[piece]=c
                for c in cells:
                    keep=ds[c]
                    for piece,owner in owners.items():
                        tick('domain_state_visits')
                        if owner!=c:keep &= ~(15<<(piece*4))
                    if not keep:return False
                    if keep!=ds[c]:ds[c]=keep;changed=True
                for c in cells:
                    for other,side,oside in neighbors[c]:
                        keep=0
                        for state in _bits(ds[c]):
                            tick('domain_state_visits')
                            for support in _bits(ds[other]):
                                tick('compatibility_checks')
                                if state//4!=support//4 and p.faces[state][side]==p.faces[support][oside]:
                                    keep|=1<<state;break
                        if not keep:return False
                        if keep!=ds[c]:ds[c]=keep;changed=True
                if not changed:return True
        if not propagate(allowed,range(n)):return result('infeasible')
        unresolved=[c for c,d in enumerate(allowed) if d.bit_count()>1]
        if not unresolved:
            board=[d.bit_length()-1 for d in allowed]
            if not check_solution(p,board): raise RuntimeError('Invalid singleton component completion')
            return result('solved')
        parents={c:c for c in unresolved}
        def find(c):
            while parents[c]!=c:parents[c]=parents[parents[c]];c=parents[c]
            return c
        def join(a,b):
            a,b=find(a),find(b)
            if a!=b:parents[b]=a
        if decompose:
            for a,b,_,_ in edges:
                tick('graph_incidences')
                if a in parents and b in parents:join(a,b)
            owner={}
            for c in unresolved:
                piece_set=set()
                for state in _bits(allowed[c]):
                    tick('domain_state_visits');piece_set.add(state//4)
                for piece in sorted(piece_set):
                    tick('graph_incidences')
                    if piece in owner:join(c,owner[piece])
                    else:owner[piece]=c
            grouped={}
            for c in unresolved:grouped.setdefault(find(c),[]).append(c)
            groups=sorted(grouped.values(),key=lambda g:(len(g),min(g)))
            if len(groups)<2:return result('not_decomposable')
        else:groups=[unresolved]
        if any(len(g)>max_component_cells for g in groups):return result('component_limit')
        # Every physical piece must occur once, so an isolated component's
        # eligible inventory must have exactly as many pieces as cells.
        assigned={ (allowed[c].bit_length()-1)//4 for c in range(n) if c not in parents }
        all_pool=set(assigned)
        for group in groups:
            pool=set()
            for c in group:
                for s in _bits(allowed[c]):tick('graph_incidences');pool.add(s//4)
            if len(pool)!=len(group) or pool & all_pool:return result('infeasible')
            all_pool|=pool
        if len(all_pool)!=n:return result('infeasible')
        def dfs(ds,group):
            if not propagate(ds,group):return None
            todo=[c for c in group if ds[c].bit_count()>1]
            if not todo:return ds
            cell=min(todo,key=lambda c:(ds[c].bit_count(),-len(neighbors[c]),c))
            for state in _bits(ds[cell]):
                tick('candidate_placements')
                child=ds.copy();child[cell]=1<<state
                solved=dfs(child,group)
                if solved is not None:return solved
            return None
        for group in groups:
            solved=dfs(allowed.copy(),group)
            if solved is None:return result('infeasible')
            for c in group:allowed[c]=solved[c]
            completed+=1
        board=[d.bit_length()-1 for d in allowed]
        if not check_solution(p,board) or any(not (original[c] & (1<<s)) for c,s in enumerate(board)):
            raise RuntimeError('Independent component witness failed validation')
        return result('solved')
    except _Limit as exc:return result(str(exc))
