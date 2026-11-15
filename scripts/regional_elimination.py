"""Admissible mismatch bounds via bounded variable elimination.

Split buckets duplicate the *choice* of an eliminated cell among groups, which
can only decrease their summed minimum. Every original edge appears once.
Duplicate-piece factors are hard, including pairs without a grid edge.
"""
from dataclasses import dataclass
from itertools import product

from exact_model import constraint_edges
from relational_regions import Work, _Limit, bits, inputs


@dataclass
class Factor:
    cells: tuple
    costs: dict


def bound(problem, domains=None, *, width=3, max_work=150000, should_stop=None):
    """Return a lower bound and, if elimination was exact, a checked optimum witness.

    A missing optional residual mask means all legal states. The score covers
    only constraint_edges(problem). A resource cutoff returns unknown.
    """
    if type(width) is not int or not 2 <= width <= 16:
        raise ValueError('Bucket width must be between 2 and 16')
    work=Work(max_work,should_stop)
    splits=0
    def result(complete,reason,lower=None,board=None,impossible=False):
        return dict(complete=complete,termination=reason,lower_bound=lower,board=board,
                    exact=complete and splits==0,structurally_infeasible=impossible,
                    split_buckets=splits,work=work.total,work_counts=dict(work.counts))
    try:
        work.take('input_states',2*sum(len(d) for d in problem.domains))
        p,_,masks,_=inputs(problem,domains,[])
        choices=[tuple(bits(mask)) for mask in masks]
        if any(not d for d in choices):return result(True,'empty_domain',impossible=True)
        pcs=[]
        for domain in choices:
            work.take('piece_incidences',len(domain))
            pcs.append({s//4 for s in domain})
        edges=constraint_edges(p)
        top=len(edges)+1
        edge_map={(a,b):(sa,sb) for a,b,sa,sb in edges}
        factors=[]
        for a,d in enumerate(choices):
            for b in range(a+1,len(choices)):
                work.take('pair_tests',min(len(pcs[a]),len(pcs[b]))+1)
                sides=edge_map.get((a,b))
                if sides is None and pcs[a].isdisjoint(pcs[b]):continue
                costs={}
                for s,t in product(d,choices[b]):
                    work.take('initial_factor_entries')
                    if s//4==t//4:cost=top
                    elif sides is None:cost=0
                    else:cost=int(p.faces[s][sides[0]]!=p.faces[t][sides[1]])
                    if cost:costs[s,t]=cost
                if costs:factors.append(Factor((a,b),costs))
        # Small domains first, then geometric order. Scores are diagnostics;
        # omitted scopes in mini-buckets do not prove feasibility.
        order=sorted(range(len(choices)),key=lambda c:(len(choices[c]),c))
        records=[]
        constant=0
        for cell in order:
            bucket=[];remaining=[]
            for f in factors:
                work.take('scope_tests',len(f.cells))
                (bucket if cell in f.cells else remaining).append(f)
            factors=remaining
            if not bucket:
                records.append((cell,(),{():choices[cell][0]}))
                continue
            groups=[]
            for factor in sorted(bucket,key=lambda f:-len(f.cells)):
                for scope,members in groups:
                    work.take('scope_tests',len(scope)+len(factor.cells))
                    if len(scope.union(factor.cells))<=width:
                        scope.update(factor.cells);members.append(factor)
                        break
                else:groups.append((set(factor.cells),[factor]))
            splits+=len(groups)-1
            for scope,members in groups:
                rest=tuple(sorted(scope-{cell}))
                full=(cell,)+rest
                slots=[]
                for f in members:
                    work.take('scope_tests',len(full)*len(f.cells))
                    slots.append(tuple(full.index(c) for c in f.cells))
                new_costs={};best_choice={}
                for other_states in product(*(choices[c] for c in rest)):
                    best=top;arg=None
                    for s in choices[cell]:
                        work.take('elimination_placements')
                        states=(s,)+other_states
                        cost=0
                        for f,indices in zip(members,slots):
                            work.take('factor_lookups',1+len(indices))
                            key=tuple(states[i] for i in indices)
                            cost+=f.costs.get(key,0)
                            if cost>=best:break
                        if cost<best:
                            best=cost;arg=s
                            if best==0:break
                    if best:new_costs[other_states]=best
                    if arg is not None and not splits:best_choice[other_states]=arg
                if rest:
                    if new_costs:factors.append(Factor(rest,new_costs))
                else:constant+=new_costs.get((),0)
                if not splits:records.append((cell,rest,best_choice))
        if constant>=top:return result(True,'infeasible',impossible=True)
        if splits:return result(True,'relaxed_lower_bound',constant)
        board=[None]*len(choices)
        for cell,rest,lookup in reversed(records):
            work.take('verification',1+len(rest))
            board[cell]=lookup[tuple(board[c] for c in rest)]
        work.take('verification',2*len(board)+len(edges))
        if (len(set(s//4 for s in board))!=len(board)
                or any(s not in choices[c] for c,s in enumerate(board))
                or sum(p.faces[board[a]][sa]!=p.faces[board[b]][sb] for a,b,sa,sb in edges)!=constant):
            raise RuntimeError('Invalid elimination witness')
        return result(True,'exact_minimum',constant,board)
    except _Limit as error:return result(False,str(error))
