"""Complete local tuple tables with bounded relational overlap propagation.

Only completed enumeration and joins justify deletion. On interruption the
original supplied domains are returned unchanged. Local tables relax constraints
outside each region; global physical-piece competition is not discarded by any
claimed completion, because this API only propagates necessary conditions.
"""
from __future__ import annotations
from dataclasses import replace
from types import MappingProxyType
import itertools
from exact_model import initial_domains, constraint_edges


class _Limit(Exception):pass


class Work:
    def __init__(self,maximum,stop=None):
        if type(maximum) is not int or not 0<=maximum<=10000000:raise ValueError('Invalid max_work')
        if stop is not None and not callable(stop):raise ValueError('Invalid stop callback')
        self.maximum=maximum;self.stop=stop;self.counts={}
    def take(self,name,amount=1):
        if self.stop is not None and self.stop():raise _Limit('stopped')
        if sum(self.counts.values())+amount>self.maximum:raise _Limit('work_limit')
        self.counts[name]=self.counts.get(name,0)+amount
    @property
    def total(self):return sum(self.counts.values())


def bits(mask):
    while mask:
        flag=mask&-mask;yield flag.bit_length()-1;mask^=flag


def inputs(problem,domains,regions):
    p=replace(problem,faces=tuple(tuple(f) for f in problem.faces),
              domains=tuple(tuple(d) for d in problem.domains),fixed=MappingProxyType(dict(problem.fixed)),
              edges=None if problem.edges is None else tuple(tuple(e) for e in problem.edges))
    raw=[sum(1<<s for s in dict.fromkeys(d)) for d in p.domains]
    if domains is None:original=raw
    else:
        original=list(itertools.islice(iter(domains),p.size*p.size+1))
        if len(original)!=len(raw) or any(type(d) is not int or d<0 or d&~raw[c] for c,d in enumerate(original)):
            raise ValueError('Domains must be legal state bitset subsets')
    legal=[sum(1<<s for s in d) for d in initial_domains(p)]
    cleaned=[d&allowed for d,allowed in zip(original,legal)]
    if regions is None:
        regions=[(r*p.size+c,r*p.size+c+1,(r+1)*p.size+c,(r+1)*p.size+c+1)
                 for r in range(p.size-1) for c in range(p.size-1)]
    else:
        try:regions=list(itertools.islice(iter(regions),257))
        except TypeError as error:raise ValueError('Invalid regions') from error
    if len(regions)>256:raise ValueError('At most256 regions')
    result=[]
    for group in regions:
        try:group=list(itertools.islice(iter(group),7))
        except TypeError as error:raise ValueError('Invalid region') from error
        if not 1<=len(group)<=6 or any(type(c) is not int or not 0<=c<len(raw) for c in group) or len(set(group))!=len(group):
            raise ValueError('Regions need1..6distinct valid cells')
        group=tuple(sorted(group))
        if group not in result:result.append(group)
    result.sort(key=lambda g:(__import__('math').prod(cleaned[c].bit_count() for c in g),g))
    return p,original,cleaned,result


def table(p,domains,cells,work,max_tuples,matching):
    if type(max_tuples) is not int or not 1<=max_tuples<=100000:raise ValueError('Invalid max_tuples')
    edges=[(i,a,b,sa,sb) for i,(a,b,sa,sb) in enumerate(constraint_edges(p)) if a in cells and b in cells]
    order=sorted(cells,key=lambda c:(domains[c].bit_count(),c));chosen={};used=set();rows=[]
    def visit(index):
        if index==len(order):
            work.take('tuple_rows')
            if len(rows)>=max_tuples:raise _Limit('tuple_limit')
            values=tuple(chosen[c] for c in cells)
            work.take('tuple_edge_checks',len(edges))
            mask=sum(1<<i for i,a,b,sa,sb in edges if p.faces[chosen[a]][sa]!=p.faces[chosen[b]][sb])
            rows.append((values,mask));return
        c=order[index]
        for state in bits(domains[c]):
            work.take('tuple_state_checks')
            if state//4 in used:continue
            valid=True
            if matching:
                for _,a,b,sa,sb in edges:
                    if c==a and b in chosen:
                        work.take('tuple_edge_checks')
                        if p.faces[state][sa]!=p.faces[chosen[b]][sb]:valid=False;break
                    if c==b and a in chosen:
                        work.take('tuple_edge_checks')
                        if p.faces[chosen[a]][sa]!=p.faces[state][sb]:valid=False;break
            if valid:
                chosen[c]=state;used.add(state//4);visit(index+1);used.remove(state//4);del chosen[c]
    visit(0)
    return rows,tuple(e[0] for e in edges)


def propagate(problem,domains=None,*,regions=None,max_work=50000,max_tuples=20000,
              relational=True,should_stop=None):
    if type(relational) is not bool:raise ValueError('relational must be boolean')
    if type(max_tuples) is not int or not 1<=max_tuples<=100000:raise ValueError('Invalid max_tuples')
    p,original,cleaned,groups=inputs(problem,domains,regions)
    work=Work(max_work,should_stop);tables=[];removed_tuples=0;rounds=0
    def result(consistent,complete,termination,output):
        return dict(consistent=consistent,complete=complete,termination=termination,domains=list(output),
            removed_states=sum(a.bit_count()-b.bit_count() for a,b in zip(original,output)),
            table_count=len(tables),tuple_counts=[len(t) for t in tables],removed_tuples=removed_tuples,
            rounds=rounds,work=work.total,work_counts=dict(work.counts),
            setup_state_visits=2*sum(len(d) for d in p.domains),
            proof_scope='supplied_puzzle_rules_and_state_domains',relational=relational)
    try:
        work.take('setup_checks',0)
        if any(not d for d in cleaned):return result(False,True,'domain_contradiction',cleaned)
        for group in groups:
            rows,_=table(p,cleaned,group,work,max_tuples,True)
            tables.append([values for values,_ in rows])
            if not rows:return result(False,True,'empty_complete_table',cleaned)
        pairs=[]
        for i,a in enumerate(groups):
            for j in range(i+1,len(groups)):
                shared=tuple(c for c in a if c in groups[j]);work.take('region_pair_checks')
                if shared:pairs.extend(((i,j,shared),(j,i,shared)))
        while True:
            rounds+=1;changed=False
            if relational:
                for i,j,shared in pairs:
                    ai=[groups[i].index(c) for c in shared];bi=[groups[j].index(c) for c in shared]
                    support=set()
                    for row in tables[j]:
                        work.take('join_signature_cells',len(bi));support.add(tuple(row[k] for k in bi))
                    kept=[]
                    for row in tables[i]:
                        work.take('join_signature_cells',len(ai));work.take('join_rows')
                        if tuple(row[k] for k in ai) in support:kept.append(row)
                    if len(kept)!=len(tables[i]):
                        removed_tuples+=len(tables[i])-len(kept);tables[i]=kept;changed=True
                    if not kept:return result(False,True,'relational_contradiction',cleaned)
            for group,rows in zip(groups,tables):
                for index,c in enumerate(group):
                    support=0
                    for row in rows:work.take('projection_cells');support|=1<<row[index]
                    after=cleaned[c]&support
                    if after!=cleaned[c]:cleaned[c]=after;changed=True
                    if not after:return result(False,True,'unary_contradiction',cleaned)
            for i,(group,rows) in enumerate(zip(groups,tables)):
                kept=[]
                for row in rows:
                    work.take('domain_tuple_checks',len(group))
                    if all(cleaned[c]>>s&1 for c,s in zip(group,row)):kept.append(row)
                if len(kept)!=len(rows):removed_tuples+=len(rows)-len(kept);tables[i]=kept;changed=True
            if not changed:return result(True,True,'fixed_point',cleaned)
    except _Limit as error:
        return result(True,False,str(error),original)
