"""Soft parent ranking and archive selection using verified repair outcomes."""
from itertools import combinations
from exact_model import grid_edges
from repair_policy import PairSelector


def features(problem,board):
    edges=grid_edges(problem.size)
    mask=frozenset(i for i,(a,b,sa,sb) in enumerate(edges)
                   if problem.faces[board[a]][sa]!=problem.faces[board[b]][sb])
    vertices={v for i in mask for v in edges[i][:2]}
    graph={v:set() for v in vertices}
    for i in mask:
        a,b,_,_=edges[i]
        graph[a].add(b)
        graph[b].add(a)
    remaining=set(vertices)
    sizes=[]
    while remaining:
        pending=[remaining.pop()]
        size=0
        while pending:
            v=pending.pop()
            size+=1
            for u in graph[v]:
                if u in remaining:
                    remaining.remove(u)
                    pending.append(u)
        sizes.append(size)
    return mask,(len(mask),len(vertices),len(sizes),max(sizes,default=0))


def vector(problem,selector,left,right,fa,fb):
    details,_=selector._features(left,right)
    a,b=fa[1],fb[1]
    scale=max(1,len(grid_edges(problem.size)))
    top,bottom=sorted(details['parent_scores'],reverse=True)
    return (1.,top/scale,bottom/scale,details['active_cycles']/len(left[0]),
            details['nontrivial_cycles']/len(left[0]),details['largest_cycle']/len(left[0]),
            details['mismatch_cycles']/len(left[0]),len(fa[0]&fb[0])/max(1,len(fa[0]|fb[0])),
            abs(a[2]-b[2])/scale,max(a[3],b[3])/len(left[0]))


def fit(xs,ys):
    import numpy as np
    x=np.asarray(xs,dtype=float)
    y=np.asarray(ys,dtype=float)
    if x.ndim!=2 or not len(x):
        raise ValueError('empty training data')
    return tuple(float(v) for v in np.linalg.solve(x.T@x+.25*np.eye(x.shape[1]),x.T@y))


def difficulty_archive(rows,bank,limit):
    # Reserve the top-scoring board, then maximize the minimum distance in
    # mismatch count, affected-cell count, component count and largest component.
    # Ties retain the better true score and stable board index.
    order=sorted(range(len(bank)),key=lambda i:(rows[i][1][0],i))
    if not order or limit<1:return []
    chosen=[order.pop(0)]
    while order and len(chosen)<limit:
        def key(i):
            delta=min(sum(abs(x-y) for x,y in zip(rows[i][1],rows[k][1])) for k in chosen)
            return delta,-rows[i][1][0],-i
        winner=max(order,key=key)
        chosen.append(winner)
        order.remove(winner)
    return chosen
