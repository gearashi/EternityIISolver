"""Construct complete positive-sum catalogs of separated signed components."""
from array import array
from collections import Counter, defaultdict
from functools import lru_cache
from math import comb
from time import monotonic

from repair_components import screen


class PlanCutoff(Exception):
    """Internal control flow for a budgeted family plan."""


def plans(hist, radius, charge=lambda units=1: None):
    charge(2*len(hist)+1)
    best=max(0,max((g for n,g in hist),default=0))
    groups=tuple(sorted(key for key in hist if key[1]+(radius-key[0])*best>0))

    @lru_cache(None)
    def ceiling(i,left):
        charge()
        if not left:return 0
        if i==len(groups):return -1000000
        n,g=groups[i]
        limit=min(hist[n,g],left//n)
        charge(limit+1)
        return max(r*g+ceiling(i+1,left-r*n) for r in range(limit+1))

    out=[]
    def build(i,left,value,taken):
        charge()
        if not left:
            if value>0:
                if len(out)>=100000:raise PlanCutoff('signed_cover_limit')
                charge(1+len(taken))
                out.append(tuple(taken))
            return
        if i>=len(groups) or value+ceiling(i,left)<=0:return
        n,g=groups[i]
        for r in range(min(hist[n,g],left//n)+1):
            charge(1+r)
            build(i+1,left-r*n,value+r*g,(*taken,*(i for _ in range(r))))
    build(0,radius,0,())
    ordered=[]
    for ids in out:
        charge(1+3*len(ids))
        counts=Counter(ids)
        order=sorted(counts,key=lambda i:(comb(hist[groups[i]],counts[i]),i))
        ordered.append(tuple(i for i in order for _ in range(counts[i])))
    return groups,tuple(ordered)


def prepare(problem,parent,prices,radius,*,max_shapes=850000,deadline=float('inf'),max_work=2**63-1):
    setup=16*sum(max(0,len(d)-1) for d in problem.domains)
    per_shape=2+radius*radius+4*radius
    shape_cap=min(max_shapes,(max_work-setup)//per_shape)
    if shape_cap<1:return {'status':'work_limit','work':0},''
    values=defaultdict(lambda:array('H'))
    hist=Counter()
    def save(shape,gain):
        # A connected component of m cells can gain at most 4*m after
        # transfer prices. Any other components together gain at most 4*(k-m).
        if gain+4*(radius-len(shape))>0:
            key=(len(shape),gain)
            values[key].extend(shape)
            hist[key]+=1
    stats=screen(problem,parent,prices,radius=radius,keep=0,max_shapes=shape_cap,
                 should_stop=lambda:monotonic()>=deadline,observe_component=save)
    work=sum(stats[key] for key in ('shapes','extensions','recipient_origin_visits','orientation_mask_scores'))+stats['shapes']
    if work>max_work:raise RuntimeError('signed-cover scan exceeded budget')
    public=dict(screen=stats,work=work)
    if monotonic()>=deadline:return dict(public,status='timeout'),''
    if stats['status']!='complete':
        return dict(public,status='work_limit' if shape_cap<max_shapes else 'signed_cover_limit'),''
    planning_work=0
    def charge(amount=1):
        nonlocal planning_work
        if monotonic()>=deadline:raise PlanCutoff('timeout')
        if work+planning_work+amount>max_work:raise PlanCutoff('work_limit')
        planning_work+=amount
    try:
        possible,patterns=plans(hist,radius,charge)
        charge(1+sum(1+3*len(p) for p in patterns))
        active=sorted({i for p in patterns for i in p})
        ids={old:new for new,old in enumerate(active)}
        charge(1+3*len(active))
        serial=1+sum(3+len(values[possible[i]]) for i in active)
        serial+=1+sum(1+len(p) for p in patterns)
        patterns=[tuple(ids[i] for i in p) for p in patterns]
        groups=[possible[i] for i in active]
        nominal=[]
        for p in patterns:
            charge(1+2*len(p))
            ways=1
            for gid,r in Counter(p).items():ways*=comb(hist[groups[gid]],r)
            nominal.append(ways)
    except PlanCutoff as error:
        return dict(public,status=str(error),work=work+planning_work,planner_work=planning_work),''
    public.update(work=work+planning_work,planner_work=planning_work)
    work+=planning_work
    if work+serial>max_work:return dict(public,status='work_limit'),''
    # Kernel counters use uint64_t. A negative proof requires exact agreement.
    if sum(nominal)>=2**63:return dict(public,status='signed_cover_limit'),''
    buf=[str(len(groups))]
    for n,g in groups:
        data=values[n,g]
        buf.append(f'{n} {g} {hist[n,g]}')
        buf.append(' '.join(map(str,data)))
    buf.append(str(len(patterns)))
    for p in patterns:buf.append(' '.join(map(str,(len(p),*p))))
    at_end=monotonic()
    public.update(work=work+serial,patterns=len(patterns),groups=len(groups),shape_rows=sum(hist[t] for t in groups),
                  nominal=sum(nominal),serialization=serial)
    if at_end>=deadline:return dict(public,status='timeout'),''
    return dict(public,status='complete'),'\n'.join(buf)+'\n'


def decode(stdout,problem,parent,radius,cap,cover):
    """Check catalog cardinality and completed pattern coverage before any proof."""
    from repair_native_radius import _decode
    if not isinstance(cover,dict) or cover.get('status')!='complete':
        raise RuntimeError('A complete signed catalog is required')
    for key,maximum in (('shape_rows',850000),('patterns',100000),('nominal',2**63-1)):
        if type(cover.get(key)) is not int or not 0<=cover[key]<=maximum:
            raise RuntimeError('Invalid signed catalog cardinality')
    if cover['patterns']>cover['nominal'] or bool(cover['patterns'])!=bool(cover['shape_rows']):
        raise RuntimeError('Inconsistent signed catalog cardinalities')
    lines=stdout.splitlines()
    if len(lines)<2:
        raise RuntimeError('signed-family receipt is missing')
    tail=lines[-1].split()
    if len(tail)!=5 or tail[0]!='signed_rows':
        raise RuntimeError('signed-family receipt is missing')
    rows,began,ended,covered=map(int,tail[1:])
    status,counts,board=_decode('\n'.join(lines[:-1]),problem,parent,radius,cap,conditional=True)
    if status not in ('family_complete','work_limit'):
        raise RuntimeError('unexpected signed helper scope')
    # A work cap can interrupt the exact evaluator after it charges one
    # set but before it finishes that leaf. It cannot cover a skipped leaf.
    interrupted_leaf = int(status=='work_limit' and began>ended)
    if (not 0<=rows<=cover['shape_rows'] or not 0<=ended<=began<=cover['patterns']
            or began-ended>1 or (began and rows!=cover['shape_rows']) or (covered and not began)
            or not 0<=covered<=cover['nominal'] or counts['tested_sets']>covered+interrupted_leaf):
        raise RuntimeError('inconsistent signed cover receipt')
    if status=='family_complete':
        if rows!=cover['shape_rows']:
            raise RuntimeError('signed catalog was incompletely loaded')
        if counts['gain']:
            if began!=ended+1 or not covered:
                raise RuntimeError('signed witness has no covered family')
        elif ended!=cover['patterns'] or covered!=cover['nominal']:
            raise RuntimeError('signed cover does not exhaust all declared families')
    return status,dict(counts,shape_rows=rows,patterns_started=began,patterns_completed=ended,
                       nominal_covered=covered),board
