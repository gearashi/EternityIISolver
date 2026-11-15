"""Exact admissible mismatch bounds from cost-partitioned overlapping regions.

Every edge owns one unit of mismatch cost. Fractions assigned across regions
never exceed that unit. Summed regional minima are therefore a lower bound even
though regions relax outside constraints and cross-region piece competition.
"""
from fractions import Fraction
from math import ceil
from exact_model import constraint_edges
from relational_regions import inputs,table,Work,_Limit,bits


def lower_bound(problem,domains=None,*,regions=None,max_work=50000,max_tuples=20000,
                strategy='optimized',should_stop=None):
    if strategy not in ('max','disjoint','uniform','optimized'):raise ValueError('Invalid partition strategy')
    if type(max_tuples) is not int or not 1<=max_tuples<=100000:raise ValueError('Invalid max_tuples')
    p,original,cleaned,groups=inputs(problem,domains,regions)
    work=Work(max_work,should_stop);masks=[];edge_sets=[];allocations=0;accepted=0
    best=Fraction(0);best_weights=None;best_parts=[];edges=constraint_edges(p)
    def result(complete,termination,infeasible=False):
        weights=[] if best_weights is None else [{str(e):[v.numerator,v.denominator] for e,v in row.items() if v} for row in best_weights]
        return dict(complete=complete,termination=termination,structurally_infeasible=infeasible,
            lower_bound=None if not complete or infeasible else ceil(best),
            rational_bound=None if not complete or infeasible else [best.numerator,best.denominator],
            region_minima=[[x.numerator,x.denominator] for x in best_parts] if complete else [],
            region_cells=[list(g) for g in groups],region_edges=[list(e) for e in edge_sets],
            cost_partition=weights if complete else [],work=work.total,work_counts=dict(work.counts),
            allocation_evaluations=allocations,accepted_transfers=accepted,strategy=strategy,
            setup_state_visits=2*sum(len(d) for d in p.domains),
            proof_scope='supplied_puzzle_constraints_and_state_domains')
    def evaluate(weights):
        nonlocal allocations
        # This independent exact check protects the load-bearing allocation invariant.
        for i,row in enumerate(weights):
            for e,value in row.items():
                work.take('weight_feasibility_checks')
                if e not in edge_sets[i] or value<0:raise RuntimeError('Invalid regional edge weight')
        for e in range(len(edges)):
            work.take('weight_feasibility_checks')
            if sum((row.get(e,Fraction(0)) for row in weights),Fraction(0))>1:raise RuntimeError('Overcounted edge cost')
        parts=[]
        for row,patterns in zip(weights,masks):
            costs=[]
            for mask in patterns:
                work.take('weighted_pattern_evaluations');value=Fraction(0)
                for e in bits(mask):work.take('weighted_edge_terms');value+=row.get(e,Fraction(0))
                costs.append(value)
            parts.append(min(costs))
        allocations+=1;return sum(parts,Fraction(0)),parts
    def consider(weights):
        nonlocal best,best_weights,best_parts
        value,parts=evaluate(weights)
        if best_weights is None or value>best:best=value;best_weights=[dict(row) for row in weights];best_parts=parts
    try:
        work.take('setup_checks',0)
        if any(not d for d in cleaned):return result(True,'domain_contradiction',True)
        for group in groups:
            rows,local_edges=table(p,cleaned,group,work,max_tuples,False)
            if not rows:return result(True,'empty_structural_table',True)
            masks.append(tuple(sorted({mask for _,mask in rows})));edge_sets.append(local_edges)
        owners={e:[i for i,es in enumerate(edge_sets) if e in es] for e in range(len(edges))}
        if not groups:best_weights=[];return result(True,'empty_partition')
        if strategy in ('max','optimized'):
            for i,es in enumerate(edge_sets):
                weights=[{} for _ in groups];weights[i]={e:Fraction(1) for e in es};consider(weights)
        if strategy in ('disjoint','optimized'):
            for order in ([*range(len(groups))],list(reversed(range(len(groups))))):
                weights=[{} for _ in groups];owned=set()
                for i in order:
                    for e in edge_sets[i]:
                        if e not in owned:weights[i][e]=Fraction(1);owned.add(e)
                consider(weights)
        if strategy in ('uniform','optimized'):
            weights=[{} for _ in groups]
            for e,indices in owners.items():
                for i in indices:weights[i][e]=Fraction(1,len(indices))
            consider(weights)
        if strategy=='optimized':
            # Two bounded coordinate sweeps; exact quarter-unit transfers. This
            # optimizes over feasible allocations but does not claim global LP optimality.
            for _ in range(2):
                changed=False
                for e,indices in owners.items():
                    for source in indices:
                        for dest in indices:
                            if source==dest or best_weights[source].get(e,0)<=0:continue
                            weights=[dict(row) for row in best_weights]
                            amount=min(Fraction(1,4),weights[source].get(e,Fraction(0)))
                            weights[source][e]-=amount;weights[dest][e]=weights[dest].get(e,Fraction(0))+amount
                            old=best;consider(weights)
                            if best>old:accepted+=1;changed=True
                if not changed:break
        return result(True,'completed')
    except _Limit as error:return result(False,str(error))
