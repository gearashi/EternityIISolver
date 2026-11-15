"""Exact bounded score search with directional cost projection on a spanning forest."""
import time
from exact_model import grid_edges, initial_domains
from exact_optimize import _inputs, evaluate_board


class _Limit(Exception):
    pass


def optimize(problem,reference,seconds=30.,seed=0,workers=1,should_stop=None,
             on_improvement=None,on_progress=None,*,target_score=None,directional=True,max_work=1000000):
    start=time.perf_counter()
    base,maximum=_inputs(problem,reference,seconds,seed,workers,should_stop,on_improvement,on_progress)
    target=min(maximum,base+1) if target_score is None else target_score
    if type(target) is not int or not 0<=target<=maximum or workers!=1:
        raise ValueError('Invalid target or worker count')
    if type(max_work) is not int or max_work<0 or type(directional) is not bool:
        raise ValueError('Invalid weighting configuration')
    counts=dict(work=0,bound_pair_checks=0,unary_checks=0,domain_scans=0,
                branches=0,singleton_placements=0,bound_prunes=0)
    def tick(name,count=1):
        if should_stop and should_stop():raise _Limit('stopped')
        if time.perf_counter()-start>=seconds:raise _Limit('timeout')
        if counts['work']+count>max_work:raise _Limit('work_limit')
        counts['work']+=count;counts[name]+=count
    edges=grid_edges(problem.size)
    faces=problem.faces
    def mismatches(domains,assigned):
        unary={c:{s:0 for s in d} for c,d in domains.items()}
        fixed_cost=0
        parent={}
        for a,b,sa,sb in edges:
            tick('unary_checks')
            if a in assigned and b in assigned:
                fixed_cost+=int(faces[assigned[a]][sa]!=faces[assigned[b]][sb])
                continue
            if a in assigned or b in assigned:
                x,y,sx,sy=(b,a,sb,sa) if a in assigned else (a,b,sa,sb)
                color=faces[assigned[y]][sy]
                for s in domains[x]:
                    tick('unary_checks')
                    unary[x][s]+=int(faces[s][sx]!=color)
                continue
            # Canonical grid edges always point right or down, hence a < b.
            if directional and b not in parent:
                parent[b]=a,sa,sb
                continue
            # Allocate each non-tree edge cost at most once to its later cell.
            minima=[]
            for t in domains[b]:
                best=maximum+1
                for s in domains[a]:
                    tick('bound_pair_checks')
                    if s//4!=t//4:
                        best=min(best,int(faces[s][sa]!=faces[t][sb]))
                        if best==0:break
                if directional:unary[b][t]+=best
                else:minima.append(best)
            if not directional:fixed_cost+=min(minima)
        total=fixed_cost
        for b in sorted(domains, reverse=True):
            if b not in parent:
                total+=min(unary[b].values())
                continue
            a,sa,sb=parent[b]
            for s in domains[a]:
                best=maximum+1
                for t,weight in unary[b].items():
                    tick('bound_pair_checks')
                    if s//4!=t//4:
                        best=min(best,weight+int(faces[s][sa]!=faces[t][sb]))
                unary[a][s]+=best
        return total

    def visit(domains,assigned):
        while True:
            if any(not d for d in domains.values()):return None
            forced=[c for c,d in domains.items() if len(d)==1]
            if not forced:break
            tick('singleton_placements',len(forced))
            reserved={domains[c][0]//4 for c in forced}
            if len(reserved)!=len(forced):return None
            for c in forced:
                assigned[c]=domains.pop(c)[0]
            for c,d in domains.items():
                tick('domain_scans', len(d))
                domains[c]=[t for t in d if t//4 not in reserved]
        if mismatches(domains,assigned)>maximum-target:
            counts['bound_prunes']+=1
            return None
        if not domains:
            candidate=[assigned[c] for c in range(len(assigned))]
            report=evaluate_board(problem,candidate)
            if not report['valid'] or report['score']<target:raise RuntimeError('Invalid weighted witness')
            return candidate
        c=min(domains,key=lambda i:(len(domains[i]),i))
        choices=sorted(domains[c],key=lambda s:(s!=reference[c],s))
        for s in choices:
            tick('branches')
            child=dict(domains)
            child[c]=[s]
            solution=visit(child,assigned.copy())
            if solution is not None:return solution
        return None

    best=list(reference);score=base;outcome=termination='timeout'
    try:
        tick('domain_scans',0)
        if score>=target:outcome=termination='solved'
        else:
            source=initial_domains(problem)
            tick('domain_scans',sum(map(len,problem.domains)))
            found=visit({c:list(d) for c,d in enumerate(source)}, {})
            if found is None:outcome=termination='infeasible'
            else:
                best=found;score=evaluate_board(problem,best)['score'];outcome=termination='solved'
                if on_improvement:on_improvement(dict(board=best.copy(),score=score,backend='weighted-forest'))
    except _Limit as error:
        termination=str(error);outcome='stopped' if termination=='stopped' else 'timeout'
    result=dict(outcome=outcome,termination=termination,board=best,score=score,target_score=target,
                backend='weighted-forest',elapsed_seconds=time.perf_counter()-start,workers=1,**counts)
    if on_progress:on_progress(result)
    return result
