"""Exact min-marginal costs in a relaxed edge-color diagram.

One transition places one allowed state. Every scoped interior edge contributes
its mismatch cost when its later endpoint is assigned. Tracking a subset of
physical pieces relaxes the other uniqueness constraints, so per-state minima
are lower bounds. The map can serve multiple score thresholds on this problem.
"""
from dataclasses import dataclass
from exact_model import constraint_edges,problem_hash
from relational_regions import Work,_Limit,inputs,bits


@dataclass
class MinMarginals:
    problem_hash: str
    domains: tuple
    minimum: int
    by_state: tuple

    def supported(self,max_mismatches):
        if type(max_mismatches) is not int or max_mismatches<0:raise ValueError('Invalid mismatch target')
        return tuple(tuple(s for s in domain if s in row and row[s]<=max_mismatches)
                     for domain,row in zip(self.domains,self.by_state))


def compile_costs(problem,domains=None,*,tracked=(),max_work=200000,max_arcs=30000,should_stop=None):
    if type(max_arcs) is not int or not 1<=max_arcs<=200000:raise ValueError('Invalid arc cap')
    work=Work(max_work,should_stop);placements=0;arcs=0
    def result(complete,why,table=None):
        return dict(complete=complete,termination=why,table=table,placements=placements,arcs=arcs,
                    work=work.total,work_counts=dict(work.counts))
    try:
        work.take('input_states',2*sum(len(domain) for domain in problem.domains))
        p,_,masks,_=inputs(problem,domains,[])
        ds=tuple(tuple(bits(mask)) for mask in masks)
        if any(not domain for domain in ds):return result(True,'empty_domain')
        edges=constraint_edges(p)
        n=len(ds)
        selected=set(tracked)
        if any(type(piece) is not int or not 0<=piece<n for piece in selected):raise ValueError('Invalid tracked identity')
        selected_mask=sum(1<<piece for piece in selected)
        work.take('identities',len(selected))
        live=[0]*(n+1)
        for cell in reversed(range(n)):
            mask=live[cell+1]
            for s in ds[cell]:
                work.take('identities')
                mask|=1<<(s//4)
            live[cell]=mask
        if selected_mask & ~live[0]:return result(True,'missing_tracked_identity')
        layers=[];frontier=();root=((),0);layer={root:0}
        for cell,states in enumerate(ds):
            opening={};closing={}
            for idx,(a,b,sa,sb) in enumerate(edges):
                work.take('scope_tests')
                if a==cell:opening[idx]=sa
                if b==cell:closing[idx]=sb
            before={edge:i for i,edge in enumerate(frontier)}
            frontier=tuple(sorted((set(frontier)-closing.keys())|opening.keys()))
            required=selected_mask & live[cell] & ~live[cell+1]
            next_layer={};layer_arcs=[]
            for old,prefix in layer.items():
                colors,used=old
                for s in states:
                    work.take('placements');placements+=1
                    pubit=1<<(s//4)
                    if used&pubit:continue
                    successor_used=used|(pubit&selected_mask)
                    if required & ~successor_used:continue
                    face=p.faces[s]
                    work.take('edge_checks',len(closing)+len(frontier))
                    delta=sum(face[side]!=colors[before[edge]] for edge,side in closing.items())
                    new=tuple(face[opening[e]] if e in opening else colors[before[e]] for e in frontier), successor_used&live[cell+1]
                    work.take('arc_updates')
                    if arcs>=max_arcs:return result(False,'arc_limit')
                    arcs+=1;layer_arcs.append((old,new,s,delta,prefix))
                    cost=prefix+delta
                    if new not in next_layer or cost<next_layer[new]:next_layer[new]=cost
            if not next_layer:return result(True,'empty_relaxation')
            layers.append(layer_arcs);layer=next_layer
        minimum=min(layer.values())
        suffix={key:0 for key in layer};marginals=[{} for _ in ds]
        for cell in reversed(range(n)):
            prior={};row=marginals[cell]
            for old,new,s,delta,prefix in layers[cell]:
                work.take('backward_arcs')
                if new not in suffix:continue
                cost=delta+suffix[new]
                prior[old]=min(prior.get(old,cost),cost)
                total=prefix+cost
                row[s]=min(row.get(s,total),total)
            suffix=prior
        if suffix[root]!=minimum:raise RuntimeError('Inconsistent forward/backward minimum')
        return result(True,'completed',MinMarginals(problem_hash(p),ds,minimum,tuple(marginals)))
    except _Limit as error:return result(False,str(error))
