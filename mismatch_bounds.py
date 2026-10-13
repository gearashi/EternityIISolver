"""Admissible integer mismatch bounds from edge-disjoint relaxed regions.

Each region ignores outside constraints and competition for pieces with other
regions. This enlarges feasibility, so its proved minimum mismatch cost is a
lower bound. Regions partition edges, preventing double-counting. A work cap
skips a square entirely and falls back to independent edge bounds.
"""
from __future__ import annotations
import itertools
import math
from exact_model import grid_edges, initial_domains


def mismatch_lower_bound(problem, *, domains=None, square_limit=4096,
                         total_limit=16384, check=lambda:None):
    if type(square_limit) is not int or not 0<=square_limit<=1000000:
        raise ValueError('square_limit must be a bounded nonnegative integer')
    if type(total_limit) is not int or not 0<=total_limit<=1000000:
        raise ValueError('total_limit must be a bounded nonnegative integer')
    check()
    base=initial_domains(problem)
    if domains is None: domains=base
    elif len(domains)!=problem.size**2: raise ValueError('One domain per cell is required')
    else:
        clean=[]
        for allowed,supplied in zip(base,domains):
            supplied=tuple(supplied)
            if any(type(s) is not int for s in supplied) or not set(supplied)<=set(allowed):
                raise ValueError('Bound domains must be subsets of legal initial domains')
            clean.append(tuple(dict.fromkeys(supplied)))
        domains=tuple(clean)
    check()
    edges=grid_edges(problem.size);faces=problem.faces
    if any(not d for d in domains):
        return {'lower_bound':None,'structurally_infeasible':True,'groups':[], 'enumerated':0,'skipped_squares':0}
    # Supports retain physical identity. A same-color pair reusing one piece
    # cannot satisfy even this relaxed single-edge subproblem.
    side_support=[]
    for cell,domain in enumerate(domains):
        check(); sides=[{} for _ in range(4)]
        for index,state in enumerate(domain):
            if index%64==0: check()
            for side,color in enumerate(faces[state]):
                sides[side][color]=sides[side].get(color,0)|(1<<(state//4))
        side_support.append(sides)
    edge_min=[]
    for a,b,sa,sb in edges:
        check();left,right=side_support[a][sa],side_support[b][sb]
        possible=any((mask|right[color]).bit_count()>=2 for color,mask in left.items() if color in right)
        edge_min.append(0 if possible else 1)
    remaining=set(range(len(edges)));groups=[];enumerated=0;skipped=0
    for row in range(0,problem.size-1,2):
        for col in range(0,problem.size-1,2):
            check();a=row*problem.size+col
            cells=(a,a+1,a+problem.size,a+problem.size+1)
            indices=[i for i,(u,v,_,_) in enumerate(edges) if u in cells and v in cells]
            work=math.prod(len(domains[c]) for c in cells)
            if work>square_limit or work>total_limit-enumerated:
                skipped+=1;continue
            best=None
            target=sum(edge_min[i] for i in indices)
            offsets={c:i for i,c in enumerate(cells)}
            for values in itertools.product(*(domains[c] for c in cells)):
                enumerated+=1
                if enumerated%64==0: check()
                if len({s//4 for s in values})!=4: continue
                cost=sum(faces[values[offsets[u]]][su]!=faces[values[offsets[v]]][sv]
                         for i in indices for u,v,su,sv in (edges[i],))
                best=cost if best is None else min(best,cost)
                # Reaching a separately established lower bound proves optimality
                # without enumerating the remaining combinations.
                if best==target: break
            check()
            if best is None:
                return {'lower_bound':None,'structurally_infeasible':True,'groups':groups,
                        'enumerated':enumerated,'skipped_squares':skipped}
            groups.append({'edges':indices,'minimum_mismatches':best,'kind':'square-relaxation'})
            remaining.difference_update(indices)
    groups.extend({'edges':[i],'minimum_mismatches':edge_min[i],'kind':'edge-relaxation'} for i in sorted(remaining))
    return {'lower_bound':sum(g['minimum_mismatches'] for g in groups),
            'structurally_infeasible':False,'groups':groups,'enumerated':enumerated,'skipped_squares':skipped}


def can_beat(problem, incumbent_score, **kwargs):
    maximum=len(grid_edges(problem.size))
    if type(incumbent_score) is not int or not 0<=incumbent_score<=maximum:
        raise ValueError('Invalid incumbent score')
    result=mismatch_lower_bound(problem,**kwargs)
    result['can_beat_bound']=not result['structurally_infeasible'] and result['lower_bound']<maximum-incumbent_score
    # True is only unknown/possibly improvable; False is the useful certificate.
    return result
