"""Necessary connected-component conditions for strict improvement within a radius."""
from collections import Counter
import numpy as np

from exact_model import constraint_edges, grid_edges, initial_domains
from exact_optimize import evaluate_board


def screen(problem,reference,prices,*,radius=6,max_shapes=250000,keep=32,should_stop=None,observe_component=None):
    """List promising shapes and, on completion, count all that can contribute.

    A positive global score delta needs a component with positive reduced gain.
    Prices for all physical-piece transfers cancel in a complete board. For a
    component, maximize its exterior matches independently and optimistically
    match all its internal edges. A size-m component inside a radius-k change
    can use at most k-m donor pieces from outside that component.
    """
    n=len(reference)
    if type(radius) is not int or not 1<=radius<=min(8,n):
        raise ValueError('radius outside bounded component study')
    if type(max_shapes) is not int or max_shapes<1 or type(keep) is not int or not 0<=keep<=1024:
        raise ValueError('invalid shape or output cap')
    if not evaluate_board(problem,reference)['valid'] or set(constraint_edges(problem))!=set(grid_edges(problem.size)):
        raise ValueError('legal full-grid reference required')
    if len(prices)!=n or any(type(x) is not int or not 0<=x<=4*n for x in prices):
        raise ValueError('invalid integral transfer potentials')
    import heapq
    faces=np.array(problem.faces,dtype=np.int16)
    pi=np.array(prices,dtype=np.int16)
    source=np.empty(n,dtype=np.int16)
    for c,s in enumerate(reference):
        source[s//4]=c
    adjacency=[{} for _ in reference]
    mismatch_count=[0]*n
    for a,b,sa,sb in grid_edges(problem.size):
        x,y=faces[reference[a],sa],faces[reference[b],sb]
        old=int(x==y)
        adjacency[a][b]=(sa,sb,old)
        adjacency[b][a]=(sb,sa,old)
        if not old:
            mismatch_count[a]+=1
            mismatch_count[b]+=1
    include=np.array([[int(not (mask&(1<<side))) for mask in range(16)] for side in range(4)],dtype=np.int16)
    profiles=np.full((n,n,16),-30000,dtype=np.int16)
    mutable=[]
    setup=0
    for c,row in enumerate(initial_domains(problem)):
        choices=np.asarray([s for s in row if s!=reference[c]],dtype=np.int16)
        if not len(choices):
            continue
        mutable.append(c)
        origins=source[choices//4]
        delta=np.zeros((len(choices),4),dtype=np.int16)
        for d,(side,other,old) in adjacency[c].items():
            color=faces[reference[d],other]
            delta[:,side]=(faces[choices,side]==color).astype(np.int16)-old
        values=(delta@ include)+(pi[c]-pi[origins])[:,None]
        if np.any(values[:,0]>0):
            raise ValueError('transfer potential violates a frozen-neighbor inequality')
        np.maximum.at(profiles[c],origins,values)
        setup+=len(choices)*16
    maximum=profiles.max(axis=1)
    allowed=set(mutable)
    shapes={(c,) for c in mutable}
    visits=extensions=origin_checks=0
    candidates=Counter()
    totals=Counter()
    examples=[]
    def finish(status):
        return dict(status=status,radius=radius,shapes=visits,extensions=extensions,
                    recipient_origin_visits=origin_checks,orientation_mask_scores=setup,
                    tested_by_size=dict(sorted(totals.items())),viable_by_size=dict(sorted(candidates.items())),
                    samples=[dict(bound=int(ub),cells=list(shape)) for ub,shape in sorted(examples,reverse=True)])
    for length in range(1,radius+1):
        following=set()
        for shape in sorted(shapes):
            if (visits%256==0 and should_stop and should_stop()) or visits>=max_shapes:
                return finish('incomplete')
            visits+=1
            totals[length]+=1
            cluster=set(shape)
            missing=0
            local_total=0
            improvements=[]
            boundary=set()
            for c in shape:
                mask=0
                for d,(side,_,old) in adjacency[c].items():
                    if d in cluster:
                        mask|=1<<side
                        missing+=int(not old and c<d)
                    elif d in allowed:
                        boundary.add(d)
                # At most k-m of these cells can simultaneously use a donor
                # from outside the cluster. Maxima may reuse a physical piece,
                # yielding an admissible upper bound.
                best_in=max(int(profiles[c,d,mask]) for d in shape)
                any_origin=int(maximum[c,mask])
                origin_checks+=len(shape)
                local_total+=best_in
                improvements.append(any_origin-best_in)
            bonus=sum(sorted(improvements,reverse=True)[:radius-length])
            upper=missing+local_total+bonus
            if observe_component is not None:
                observe_component(shape,upper)
            if upper>0:
                candidates[length]+=1
                if keep:
                    item=(upper,shape)
                    if len(examples)<keep:
                        heapq.heappush(examples,item)
                    elif item>examples[0]:
                        heapq.heapreplace(examples,item)
            if length<radius:
                for d in boundary:
                    extensions+=1
                    following.add(tuple(sorted((*shape,d))))
                    if len(following)>max_shapes:
                        return finish('incomplete')
        shapes=following
    return finish('complete')
