"""Integral transfer potentials for changes on pairwise nonadjacent cells."""
from exact_model import constraint_edges, grid_edges, initial_domains
from exact_optimize import evaluate_board


def certificate(problem, reference, *, max_work=None, should_stop=None):
    """Check or refute integral transfer prices, optionally under one hard work limit."""
    if max_work is not None and (type(max_work) is not int or max_work<0):
        raise ValueError('max_work must be nonnegative')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('invalid should_stop')
    edges=grid_edges(problem.size)
    if not evaluate_board(problem,reference)['valid'] or set(edges)!=set(constraint_edges(problem)):
        raise ValueError('valid full-grid reference required')
    board=tuple(reference)
    n=len(board)
    counts=dict(domain_setup_visits=0,state_visits=0,edge_tests=0,relaxation_visits=0,verification_visits=0)
    work=0
    class Limit(Exception):
        pass
    def charge(field,amount):
        nonlocal work
        if should_stop and should_stop():
            raise Limit('stopped')
        if max_work is not None and work+amount>max_work:
            raise Limit('work_limit')
        counts[field]+=amount
        work+=amount
    origin={s//4:c for c,s in enumerate(board)}
    incident=[[] for _ in board]
    for a,b,sa,sb in edges:
        x,y=problem.faces[board[a]][sa],problem.faces[board[b]][sb]
        incident[a].append((sa,y,int(x==y)))
        incident[b].append((sb,x,int(x==y)))
    weights=[]
    try:
        charge('domain_setup_visits',sum(map(len,problem.domains)))
        for c,domain in enumerate(initial_domains(problem)):
            # Reserve a row as one chunk and report visits only after both
            # reservations fit, so an interrupted row cannot claim work.
            needed=len(domain)*(1+len(incident[c]))
            if max_work is not None and work+needed>max_work:
                raise Limit('work_limit')
            charge('state_visits',len(domain))
            counts['edge_tests']+=len(domain)*len(incident[c])
            work+=len(domain)*len(incident[c])
            best={}
            for s in domain:
                value=sum(int(problem.faces[s][side]==color)-was for side,color,was in incident[c])
                q=origin[s//4]
                if q not in best or value>best[q][0]:
                    best[q]=(value,s)
            weights.extend((c,q,w,s) for q,(w,s) in best.items())
        pi=[0]*n
        pred=[None]*n
        for _ in range(n):
            changed=None
            for offset in range(0,len(weights),256):
                chunk=weights[offset:offset+256]
                charge('relaxation_visits',len(chunk))
                for c,q,w,s in chunk:
                    if pi[q]<pi[c]+w:
                        pi[q]=pi[c]+w
                        pred[q]=(c,w,s)
                        changed=q
            if changed is None:
                charge('verification_visits',len(weights))
                if not all(w<=pi[q]-pi[c] for c,q,w,_ in weights):
                    raise RuntimeError('potential certificate failed')
                return dict(certified=True,status='certified',potentials=pi,work=work,**counts)
        cur=changed
        for _ in range(n):
            cur=pred[cur][0]
        cycle=[]
        candidate=list(board)
        total=0
        while cur not in cycle:
            charge('verification_visits',1)
            cycle.append(cur)
            prev,w,s=pred[cur]
            candidate[prev]=s
            total+=w
            cur=prev
        if cur!=cycle[0] or total<=0:
            raise RuntimeError('positive-cycle extraction failed')
        verified=evaluate_board(problem,candidate)
        if not verified['valid']:
            raise RuntimeError('transfer witness illegal')
        return dict(certified=False,status='positive_cycle',relaxed_gain=total,cycle=cycle,board=candidate,
                    score=verified['score'],work=work,**counts)
    except Limit as exc:
        return dict(certified=False,status=str(exc),work=work,**counts)
