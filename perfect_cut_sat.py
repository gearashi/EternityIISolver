"""Verified Hamming-distance cuts for exact edge matching under an optional edit radius.

A certificate names the parent, puzzle domains, oriented clues and exact edge
scope. It is rechecked on every invocation. No score optimization entry point
accepts these hard restrictions.
"""
from dataclasses import replace
from hashlib import sha256
from itertools import combinations
import json
import math
import time

from exact_model import check_solution, constraint_edges, initial_domains, problem_hash
from exact_sat import AssumptionSession


def _board_hash(parent):
    return sha256(json.dumps(list(parent),separators=(',',':')).encode('ascii')).hexdigest()


def certificate(problem,parent,rows,scale):
    """Tag candidate rows for transport; the consumer verifies them independently."""
    return {'schema':1,'objective':'all-supplied-edges-perfect','problem':problem_hash(problem),
            'parent':_board_hash(parent),'scale':scale,'rows':rows}


class _Limit(Exception):
    pass


def _changed_assignment(problem, local, candidates, tick, local_masks):
    # Independent exact check on surviving changed cells. Each placement
    # simultaneously removes its tile and constrains every changed neighbor.
    if not candidates:
        return True
    cell=min(candidates,key=lambda c:(candidates[c].bit_count(),c))
    states=candidates[cell]
    while states:
        bit=states&-states
        states^=bit
        state=bit.bit_length()-1
        tick('verified_placements')
        rest={c:mask&~(15<<(4*(state//4))) for c,mask in candidates.items() if c!=cell}
        for a,b,sa,sb in local:
            tick('verified_edges')
            if a==cell and b in rest:
                rest[b]&=local_masks[sb].get(problem.faces[state][sa],0)
            elif b==cell and a in rest:
                rest[a]&=local_masks[sa].get(problem.faces[state][sb],0)
        if all(rest.values()) and _changed_assignment(problem,local,rest,tick,local_masks):
            return True
    return False


def _verify(problem, parent, allowed, cert, tick):
    if (type(cert) is not dict or type(cert.get('schema')) is not int or cert['schema']!=1 or cert.get('objective')!='all-supplied-edges-perfect'
            or cert.get('problem')!=problem_hash(problem) or cert.get('parent')!=_board_hash(parent)):
        raise ValueError('Distance certificate does not match the exact task and parent')
    scale=cert.get('scale')
    if type(scale) is not int or scale<=0 or not isinstance(cert.get('rows'),(list,tuple)):
        raise ValueError('Invalid certificate scale or rows')
    n=len(parent)
    masks=[{} for _ in range(4)]
    for state,face in enumerate(problem.faces):
        tick('verification_setup',4)
        for side,color in enumerate(face):
            masks[side][color]=masks[side].get(color,0)|(1<<state)
    base=[sum(1<<s for s in domain)&~(1<<parent[c]) for c,domain in enumerate(allowed)]
    tick('verification_setup',sum(map(len,allowed)))
    edges=constraint_edges(problem)
    loads=[0]*n
    numerator=0
    cuts=[]
    for row in cert['rows']:
        if not isinstance(row,(tuple,list)) or len(row)!=3:
            raise ValueError('Invalid certificate row')
        items,minimum,weight=row
        if not isinstance(items,(tuple,list)) or not 1<=len(items)<=12 or any(type(c) is not int for c in items) or len(set(items))!=len(items):
            raise ValueError('Invalid distance cut cells')
        cells=tuple(items)
        if (any(type(c) is not int or not 0<=c<n for c in cells) or type(minimum) is not int
            or not 0<=minimum<=len(cells) or type(weight) is not int or not 1<=weight<=scale):
            raise ValueError('Invalid distance cut')
        inside=set(cells)
        tick('verification_setup',len(edges))
        local=[e for e in edges if e[0] in inside and e[1] in inside]
        for k in range(minimum):
            for changed_tuple in combinations(cells,k):
                tick('verified_subsets')
                changed=set(changed_tuple);fixed=inside-changed
                banned=0
                for c in fixed:
                    banned|=15<<(4*(parent[c]//4))
                candidates={c:base[c]&~banned for c in changed}
                for a,b,sa,sb in local:
                    tick('verified_edges')
                    if a in fixed and b in fixed:
                        if problem.faces[parent[a]][sa]!=problem.faces[parent[b]][sb]:
                            break
                    elif a in fixed:
                        candidates[b]&=masks[sb].get(problem.faces[parent[a]][sa],0)
                    elif b in fixed:
                        candidates[a]&=masks[sa].get(problem.faces[parent[b]][sb],0)
                else:
                    if all(candidates.values()) and _changed_assignment(problem,local,candidates,tick,masks):
                        # Includes the empty changed set whenever its fixed
                        # edges already match. This relaxation admits it.
                        raise ValueError('Independent distance proof does not imply this cut')
        for c in cells:loads[c]+=weight
        numerator+=weight*minimum
        if minimum:cuts.append((cells,minimum))
    if any(w>scale for w in loads):raise ValueError('Invalid overlapping distance weights')
    return cuts,(numerator+scale-1)//scale


class _DistanceSession(AssumptionSession):
    def __init__(self,p,parent,cuts,radius,**options):
        self.parent=parent;self.bounds=cuts;self.radius=radius
        self.card_clauses=0;self.card_aux=0
        super().__init__(p,**options)

    def _build(self):
        super()._build()
        self._built=False
        kept=[self.placements[c][s] for c,s in enumerate(self.parent)]
        from pysat.card import CardEnc,EncType
        def at_most(literals,limit):
            self.check()
            if limit>=len(literals):return
            before=self.next_var
            encoded=CardEnc.atmost(lits=literals,bound=limit,top_id=before,encoding=EncType.seqcounter)
            self.next_var=max(self.next_var,encoded.nv)
            self.card_aux+=self.next_var-before
            for clause in encoded.clauses:
                self._add(clause)
                self.card_clauses+=1
        for cells,k in self.bounds:
            at_most([kept[c] for c in cells],len(cells)-k)
        if self.radius is not None:
            at_most([-v for v in kept],self.radius)
        self.check()
        self._built=True


def solve(problem,parent,seconds=3.,seed=0,*,cert=None,radius=None,should_stop=None):
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
        raise ValueError('Invalid seconds')
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Invalid seed')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('Invalid cancellation callback')
    start=time.perf_counter()
    p=replace(problem,faces=tuple(tuple(f) for f in problem.faces),domains=tuple(tuple(d) for d in problem.domains),
              fixed=dict(problem.fixed),edges=None if problem.edges is None else tuple(tuple(e) for e in problem.edges))
    parent=tuple(parent)
    n=p.size**2
    if radius is not None and (type(radius) is not int or not 0<=radius<=n):
        raise ValueError('Invalid edit radius')
    if len(parent)!=n or any(type(s) is not int or not 0<=s<len(p.faces) for s in parent) or len({s//4 for s in parent})!=n:
        raise ValueError('Invalid distance parent')
    counts=dict(validation_domain_entries=0,verification_setup=0,verified_subsets=0,verified_edges=0,verified_placements=0)
    def tick(name,amount=1):
        if should_stop and should_stop():raise _Limit('stopped')
        if time.perf_counter()-start>=seconds:raise _Limit('timeout')
        counts[name]+=amount
    lower=None;cuts=[]
    r=dict(board=None,branches=0,conflicts=0,native_propagations=0,variables=0,clauses=0,
           native_started=False,card_clauses=0,card_aux=0)
    try:
        tick('validation_domain_entries',0)
        allowed=initial_domains(p)
        tick('validation_domain_entries',sum(map(len,p.domains)))
        if any(s not in allowed[c] for c,s in enumerate(parent)):
            raise ValueError('Parent state lies outside original frame, domain or clues')
        if cert is not None:
            cuts,lower=_verify(p,parent,allowed,cert,tick)
            tick('verification_setup',0)
        if radius is not None and lower is not None and radius<lower:
            r['outcome']='infeasible'
            r['termination']='verified_distance_exceeds_radius'
        else:
            with _DistanceSession(p,parent,cuts,radius,seconds=max(0.,seconds-(time.perf_counter()-start)),seed=seed,
                                  should_stop=should_stop,piece_channels=True,small_pairwise=True) as engine:
                r=engine.solve()
                r.update(native_started=engine.model_builds>0,card_clauses=engine.card_clauses,card_aux=engine.card_aux)
                r['termination']=r['outcome']
        if r['board'] is not None:
            if not check_solution(p,r['board']) or (radius is not None and sum(s!=t for s,t in zip(parent,r['board']))>radius):
                raise RuntimeError('Distance-constrained SAT witness failed original verification')
    except _Limit as exc:
        r['outcome']=r['termination']=str(exc)
    r.update(radius=radius,lower_bound=lower,verified_local_cuts=len(cuts),
             total_elapsed_seconds=time.perf_counter()-start,**counts)
    return r
