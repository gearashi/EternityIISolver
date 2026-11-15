"""Bounded exact traversal of equal-score boards under legal one/two-cell moves."""
from collections import deque
from itertools import combinations, permutations, product

from exact_model import constraint_edges, grid_edges, initial_domains
from exact_optimize import evaluate_board


def explore(problem, reference, *, max_boards=32, max_work=20_000_000, should_stop=None):
    if type(max_boards) is not int or max_boards < 1 or type(max_work) is not int or max_work < 0:
        raise ValueError('positive board limit and nonnegative work limit required')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('invalid should_stop')
    initial=evaluate_board(problem, reference)
    edges=grid_edges(problem.size)
    if not initial['valid'] or set(constraint_edges(problem))!=set(edges):
        raise ValueError('valid full-grid reference required')
    parent=tuple(reference)
    seen={parent}
    todo=deque([parent])
    level=initial['score']
    expanded=0
    neutral_moves=0
    counts=dict(domain_visits=0, cell_sets=0, allocation_tests=0, state_visits=0,
                orientation_combinations=0, edge_tests=0, validations=1)
    work=0

    class Limit(Exception):
        pass

    def charge(key,n=1):
        nonlocal work
        if work+n > max_work:
            raise Limit('work_limit')
        if work//256 != (work+n)//256 and should_stop and should_stop():
            raise Limit('stopped')
        counts[key]+=n
        work+=n

    def answer(why,board=parent,score=level):
        return dict(termination=why,score=score,reference_score=level,board=list(board),
                    improved=score>level,boards_discovered=len(seen),boards_fully_expanded=expanded,
                    neutral_transitions=neutral_moves,work=work,counts=counts.copy(),
                    scope='reachable equal-score component and strict exits using one/two-cell moves')

    def handle(b,cells,candidate,gain):
        nonlocal neutral_moves
        changed=list(b)
        for cell,s in zip(cells,candidate):
            changed[cell]=s
        new=tuple(changed)
        if gain==0:
            neutral_moves+=1
            if new in seen:
                return None
        checked=evaluate_board(problem,new)
        counts['validations']+=1
        if not checked['valid'] or checked['score']!=level+gain:
            raise RuntimeError('plateau neighbor failed full validation')
        if gain>0:
            return answer('improved',new,checked['score'])
        if len(seen)>=max_boards:
            raise Limit('board_limit')
        seen.add(new)
        todo.append(new)
        return None

    try:
        if should_stop and should_stop():
            return answer('stopped')
        grouped=[]
        for values in initial_domains(problem):
            charge('domain_visits',len(values))
            row={}
            for s in values:
                row.setdefault(s//4,[]).append(s)
            grouped.append(row)
        mutable=tuple(c for c in range(len(parent)) if c not in problem.fixed and len(problem.domains[c])>1)
        while todo:
            b=todo.popleft()
            local=[[] for _ in b]
            shared={}
            for c,d,sc,sd in edges:
                charge('edge_tests')
                x,y=problem.faces[b[c]][sc],problem.faces[b[d]][sd]
                equal=int(x==y)
                local[c].append((sc,y,equal))
                local[d].append((sd,x,equal))
                shared[c,d]=sc,sd,x,y
            profiles={}
            def choices(c,origin):
                key=(c,origin)
                if key not in profiles:
                    rows=[]
                    for s in grouped[c].get(b[origin]//4,()):
                        charge('state_visits')
                        if s!=b[c]:
                            charge('edge_tests',len(local[c]))
                            gain=sum(int(problem.faces[s][side]==other)-old for side,other,old in local[c])
                            rows.append((gain,s))
                    profiles[key]=sorted(rows,reverse=True)
                return profiles[key]
            for c in mutable:
                charge('cell_sets')
                for gain,s in choices(c,c):
                    if gain>=0:
                        found=handle(b,(c,),(s,),gain)
                        if found:
                            return found
            for c,d in combinations(mutable,2):
                charge('cell_sets')
                link=shared.get((c,d))
                for first,second in permutations((c,d)):
                    charge('allocation_tests')
                    left=choices(c,first)
                    right=choices(d,second)
                    if not left or not right:
                        continue
                    # A shared edge contributes at most +2 after adding both
                    # single-cell deltas. Only affected edges are counted once.
                    if left[0][0]+right[0][0]+(2 if link else 0)<0:
                        continue
                    for (gx,s),(gy,t) in product(left,right):
                        charge('orientation_combinations')
                        gain=gx+gy
                        if link:
                            sc,sd,x,y=link
                            u,v=problem.faces[s][sc],problem.faces[t][sd]
                            charge('edge_tests',4)
                            gain+=int(u==v)+int(x==y)-int(u==y)-int(x==v)
                        if gain>=0:
                            found=handle(b,(c,d),(s,t),gain)
                            if found:
                                return found
            expanded+=1
        return answer('exhausted')
    except Limit as exc:
        return answer(str(exc))
