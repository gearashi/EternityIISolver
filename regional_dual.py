"""Checked Lagrangian lower bounds and per-state fixings for edge mismatches.

For each directed edge pick f(color) in [0,D]. The difference f(a)-f(b)
is at most D times the mismatch of its two endpoints. Every physical piece
appears exactly once in a complete assignment, so arbitrary per-piece prices
cancel. The resulting row minima give a provable score bound. The native LP
only proposes potentials; all search-relevant consequences use exact integers.
"""
import math
from exact_model import constraint_edges
from relational_regions import Work,_Limit,inputs,bits


def certify(problem,ds,scale,edge_prices,piece_prices,work):
    if type(scale) is not int or not 1<=scale<=1000000:raise ValueError('Invalid dual scale')
    n=len(ds);edges=constraint_edges(problem)
    if (len(piece_prices)!=n or any(type(x) is not int for x in piece_prices)
            or len(edge_prices)!=len(edges)):raise ValueError('Invalid dual vectors')
    incidence=[[] for _ in ds]
    for i,(a,b,sa,sb) in enumerate(edges):
        work.take('dual_validation',1+len(edge_prices[i]))
        if any(type(k) is not int or type(v) is not int or not 0<=v<=scale for k,v in edge_prices[i].items()):
            raise ValueError('Invalid edge potential')
        incidence[a].append((i,sa,1));incidence[b].append((i,sb,-1))
    rows=[]
    for c,domain in enumerate(ds):
        values={}
        for state in domain:
            work.take('dual_terms',1+len(incidence[c]))
            values[state]=piece_prices[state//4]+sum(sign*edge_prices[e].get(problem.faces[state][side],0)
                                                    for e,side,sign in incidence[c])
        if not values:raise ValueError('Cannot certify an empty domain')
        rows.append(values)
    minima=[min(values.values()) for values in rows]
    raw=sum(minima)-sum(piece_prices)
    def ceil_div(x):return -(-x//scale)
    local=tuple({state:max(0,ceil_div(raw+value-minimum)) for state,value in values.items()}
                for values,minimum in zip(rows,minima))
    work.take('fixings',sum(map(len,rows))+len(ds))
    return max(0,ceil_div(raw)),local


def bound(problem,domains=None,*,scale=4096,max_work=200000,seconds=1.0,should_stop=None):
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=60:
        raise ValueError('Invalid LP proposal limit')
    if type(scale) is not int or not 1<=scale<=1000000:raise ValueError('Invalid potential scale')
    work=Work(max_work,should_stop)
    solver=None;status=None;iterations=0
    def result(complete,termination,lower=None,fixings=None,certificate=None):
        return dict(complete=complete,termination=termination,lower_bound=lower,by_state=fixings,
                    work=work.total,work_counts=dict(work.counts),native_status=status,
                    simplex_iterations=iterations,certificate=certificate)
    try:
        work.take('input_states',2*sum(len(d) for d in problem.domains))
        p,_,masks,_=inputs(problem,domains,[])
        ds=[tuple(bits(mask)) for mask in masks]
        if any(not domain for domain in ds):return result(True,'empty_domain')
        from ortools.linear_solver import pywraplp
        solver=pywraplp.Solver.CreateSolver('GLOP')
        if solver is None:return result(False,'lp_unavailable')
        solver.SetTimeLimit(max(1,int(1000*seconds)))
        inf=solver.infinity()
        q=[solver.NumVar(0,0,'')]+[solver.NumVar(-inf,inf,'') for _ in ds[1:]]
        t=[solver.NumVar(-inf,inf,'') for _ in ds]
        work.take('lp_variables',len(q)+len(t))
        edge_vars=[];incident=[[] for _ in ds]
        for i,(a,b,sa,sb) in enumerate(constraint_edges(p)):
            work.take('edge_domains',len(ds[a])+len(ds[b]))
            palette={p.faces[s][sa] for s in ds[a]}|{p.faces[s][sb] for s in ds[b]}
            edge_vars.append({k:solver.NumVar(0,1,'') for k in sorted(palette)})
            work.take('lp_variables',len(palette));incident[a].append((i,sa,1));incident[b].append((i,sb,-1))
        for c,domain in enumerate(ds):
            for s in domain:
                work.take('lp_coefficients',3+len(incident[c]))
                row=solver.RowConstraint(-inf,0,'')
                row.SetCoefficient(t[c],1);row.SetCoefficient(q[s//4],-1)
                for edge,side,sign in incident[c]:row.SetCoefficient(edge_vars[edge][p.faces[s][side]],-sign)
        objective=solver.Objective()
        for v in t:objective.SetCoefficient(v,1)
        for v in q:objective.SetCoefficient(v,-1)
        objective.SetMaximization();work.take('lp_coefficients',len(t)+len(q))
        work.take('lp_solve_calls')
        status=solver.Solve();iterations=solver.iterations()
        work.take('certificate_reads',len(q)+sum(map(len,edge_vars)))
        if status not in (pywraplp.Solver.OPTIMAL,pywraplp.Solver.FEASIBLE):return result(False,'no_dual_proposal')
        raw_pieces=[variable.solution_value() for variable in q]
        raw_edges=[{color:variable.solution_value() for color,variable in group.items()}
                   for group in edge_vars]
        if (any(not math.isfinite(v) for v in raw_pieces)
                or any(not math.isfinite(v) for group in raw_edges for v in group.values())):
            return result(False,'invalid_price_proposal')
        # Any finite piece prices certify a bound. Clipping before scaling
        # also keeps a poorly conditioned native proposal representable.
        price_limit=max(1,len(edge_vars))
        prices=[round(scale*min(price_limit,max(-price_limit,v))) for v in raw_pieces]
        fx=[{color:round(scale*min(1,max(0,value))) for color,value in group.items()}
            for group in raw_edges]
        lower,fixings=certify(p,ds,scale,fx,prices,work)
        return result(True,'certified',lower,fixings,dict(scale=scale,edges=fx,pieces=prices))
    except _Limit as error:return result(False,str(error))
