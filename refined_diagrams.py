"""Bounded exact edge matching through progressively refined decision diagrams.

The relaxation forgets physical-piece identity for untracked pieces but keeps
all edge colors, frames, clues and cell domains. Every real solution remains a
path. A repeated-piece witness refines that abstraction; only an independently
validated full path is accepted. Interrupted construction is never infeasibility.
"""
from dataclasses import replace
from types import MappingProxyType
import math,random,time
from exact_model import initial_domains,constraint_edges,check_solution

class _Limit(Exception):
    def __init__(self,reason):self.reason=reason



def _domain_order(problem, domains, edges, abstraction, tick):
    """Rank complete permutations, never remove states or approximate a proof.

    The estimate is log2(color frontier combinations) plus one bit for each
    tracked physical identity eligible on both sides. Untracked identity
    coupling breaks ties. These are ordering estimates, not pruning bounds.
    All estimates and greedy candidate construction are charged to setup_work.
    """
    n=len(domains);size=problem.size;piece_masks=[];pieces=[]
    for domain in domains:
        mask=0
        for state in domain:tick();mask|=1<<(state//4)
        piece_masks.append(mask)
        pieces.append(tuple(piece for piece in range(n) if mask&(1<<piece)))
        tick(n)
    tracked=(1<<n)-1 if abstraction=='exact' else sum(1<<piece for piece in {s//4 for s in problem.fixed.values()})
    adjacent=[[] for _ in range(n)]
    for a,b,sa,sb in edges:
        ca=set();cb=set()
        for state in domains[a]:tick();ca.add(problem.faces[state][sa])
        for state in domains[b]:tick();cb.add(problem.faces[state][sb])
        tick(len(ca)+len(cb))
        weight=math.log2(max(1,len(ca&cb)))
        adjacent[a].append((b,weight));adjacent[b].append((a,weight))
    def score(order):
        suffix=[0]*(n+1)
        for k in range(n-1,-1,-1):tick();suffix[k]=suffix[k+1]|piece_masks[order[k]]
        left=0;placed=set();width=0.;peak=total=0.;coupling_peak=coupling_total=0
        for k,cell in enumerate(order):
            tick();left|=piece_masks[cell]
            for other,weight in adjacent[cell]:
                tick();width+=-weight if other in placed else weight
            placed.add(cell);live=left&suffix[k+1];tick(2)
            estimate=width+(live&tracked).bit_count()
            peak=max(peak,estimate);total+=estimate
            coupling=live.bit_count();coupling_peak=max(coupling_peak,coupling);coupling_total+=coupling
        return (round(peak,12),round(total,12),coupling_peak,coupling_total,order)
    def greedy(reverse):
        left=right=0;remaining=set(range(n));placed=set();counts=[0]*n;order=[];width=0.
        for mask,eligible in zip(piece_masks,pieces):
            tick();right|=mask
            for piece in eligible:tick();counts[piece]+=1
        tick(n)
        last_support=sum(1 << piece for piece,count in enumerate(counts) if count == 1)
        while remaining:
            best=None
            for cell in sorted(remaining,reverse=reverse):
                # Dropping this cell removes exactly the identities with no
                # other support in the remaining cells.
                tick(2);removed=piece_masks[cell] & last_support
                next_right=right&~removed;next_left=left|piece_masks[cell];live=next_left&next_right
                next_width=width;closed=0
                for other,weight in adjacent[cell]:
                    tick();next_width+=-weight if other in placed else weight;closed+=int(other in placed)
                tick(2)
                key=(round(next_width+(live&tracked).bit_count(),12),live.bit_count(),len(domains[cell]),-closed,-cell if reverse else cell)
                if best is None or key<best[0]:best=(key,cell,next_left,next_right,next_width)
            _,cell,left,right,width=best;order.append(cell);remaining.remove(cell);placed.add(cell)
            for piece in pieces[cell]:
                tick(2);counts[piece]-=1
                if counts[piece]==1:last_support |= 1<<piece
                elif counts[piece]==0:last_support &= ~(1<<piece)
        return tuple(order)
    row=tuple(range(n));col=tuple(r*size+j for j in range(size) for r in range(size))
    snake=tuple(r*size+j for r in range(size) for j in (range(size) if r%2==0 else range(size-1,-1,-1)))
    col_snake=tuple(r*size+j for j in range(size) for r in (range(size) if j%2==0 else range(size-1,-1,-1)))
    candidates=tuple(dict.fromkeys((row,col,row[::-1],col[::-1],snake,snake[::-1],col_snake,col_snake[::-1],greedy(False),greedy(True))))
    return min(candidates,key=score),len(candidates)


def solve(problem,seconds=30.,seed=0,workers=1,should_stop=None,on_progress=None,*,
          abstraction='refine',max_work=1000000,max_nodes=20000,max_refinements=256,refine_batch=4,
          ordering='geometric', identity_lifetime='full',groups=False,refinement='frequency'):
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid seconds')
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid seed')
    if workers!=1 or type(workers) is not int:raise ValueError('Decision diagrams use one worker')
    if abstraction not in ('refine','exact'):raise ValueError('Invalid abstraction')
    if ordering not in ('geometric','domain'):raise ValueError('Invalid diagram ordering')
    if type(groups) is not bool:raise ValueError('Invalid diagram group capacity setting')
    if refinement not in ('frequency','local'):raise ValueError('Invalid diagram refinement')
    if identity_lifetime not in ('full','project','require'):
        raise ValueError('Invalid diagram identity lifetime')
    for name,v,lo,hi in (('max_work',max_work,0,10000000),('max_nodes',max_nodes,1,200000),
                         ('max_refinements',max_refinements,0,256),('refine_batch',refine_batch,1,256)):
        if type(v) is not int or not lo<=v<=hi:raise ValueError('Invalid '+name)
    if should_stop is not None and not callable(should_stop):raise ValueError('Invalid stop callback')
    if on_progress is not None and not callable(on_progress):raise ValueError('Invalid progress callback')
    start=time.monotonic();deadline=start+seconds;last=start;board=None;order=();n=problem.size**2
    p=replace(problem,faces=tuple(tuple(f) for f in problem.faces),domains=tuple(tuple(d) for d in problem.domains),
              fixed=MappingProxyType(dict(problem.fixed)),edges=None if problem.edges is None else tuple(tuple(e) for e in problem.edges))
    c=dict(candidate_placements=0,edge_checks=0,state_operations=0,witness_checks=0,setup_work=0,
           diagrams=0,refinements=0,tracked_pieces=0,merged_paths=0,peak_layer_nodes=0,total_nodes=0,max_depth=0,
           ordering_work=0,ordering_candidates=0,expired_piece_failures=0,group_failures=0,local_pair_failures=0,local_pairs=0)
    counted=('candidate_placements','edge_checks','state_operations','witness_checks','setup_work')
    def work():return sum(c[k] for k in counted)
    def result(outcome,termination):
        return dict(outcome=outcome,termination=termination,board=board,branches=c['candidate_placements'],conflicts=0,
                    max_depth=c['max_depth'],elapsed_seconds=time.monotonic()-start,work=work(),
                    reasoning_counters_session=dict(c,work=work()),abstraction=abstraction,ordering=ordering,
                    cell_order=list(order),identity_lifetime=identity_lifetime,groups=groups,refinement=refinement,
                    proof_scope='full_supplied_problem',coverage_verified=outcome in ('solved','infeasible'),
                    resume_capability='fresh search; no diagram checkpoint')
    def check():
        nonlocal last
        if should_stop and should_stop():raise _Limit('stopped')
        if time.monotonic()>=deadline:raise _Limit('timeout')
        now=time.monotonic()
        if on_progress and now-last>=.25:
            last=now;on_progress(dict(result('running','building_diagram'),phase='building_diagram'))
    def tick(name,amount=1):
        check()
        if work()+amount>max_work:raise _Limit('work_limit')
        c[name]+=amount
    try:
        check()
        for d in p.domains:tick('setup_work',len(d))
        domains=initial_domains(p);edges=constraint_edges(p)
        if any(not d for d in domains):return result('infeasible','empty_domain')
        # Fixed deterministic row/column choices limit color-interface width.
        row=tuple(range(n));col=tuple(r*p.size+j for j in range(p.size) for r in range(p.size))
        def widths(order):
            ranks={v:i for i,v in enumerate(order)}
            maximum=0
            for k in range(n+1):
                width=0
                for a,b,_,_ in edges:
                    tick('setup_work');width+=int(min(ranks[a],ranks[b])<k<=max(ranks[a],ranks[b]))
                maximum=max(maximum,width)
            return maximum
        order=min((row,col),key=lambda order:(widths(order),order))
        if ordering=='domain':
            def order_tick(amount=1):
                tick('setup_work',amount);c['ordering_work']+=amount
            order,c['ordering_candidates']=_domain_order(p,domains,edges,abstraction,order_tick)
        rank={v:i for i,v in enumerate(order)}
        crossings=[]
        for k in range(n+1):
            crossing=[]
            for index,(a,b,sa,sb) in enumerate(edges):
                tick('setup_work')
                if min(rank[a],rank[b])<k<=max(rank[a],rank[b]):crossing.append(index)
            crossings.append(tuple(crossing))
        instructions=[]
        for k,cell in enumerate(order):
            before={edge:i for i,edge in enumerate(crossings[k])};tests=[];carry=[]
            for ei,(a,b,sa,sb) in enumerate(edges):
                tick('setup_work')
                if cell==a and rank[b]<k:tests.append((before[ei],sa))
                if cell==b and rank[a]<k:tests.append((before[ei],sb))
            for ei in crossings[k+1]:
                a,b,sa,sb=edges[ei]
                carry.append((True,sa if cell==a else sb) if cell in (a,b) else (False,before[ei]))
            instructions.append((tests,carry))
        # Fix each cell's value order independently of traversal selection, so
        # an ordering ablation does not also redistribute random draws.
        rng=random.Random(seed);by_cell=[]
        for domain in domains:
            states=list(domain);rng.shuffle(states);by_cell.append(tuple(states))
        ordered=[by_cell[cell] for cell in order]
        # A forgotten bit may no longer occur in any future domain. Nodes still
        # keep a concrete predecessor so a terminal witness can be reconstructed.
        live_after=[0]*n
        expires=[0]*n
        if identity_lifetime!='full':
            seen=0
            for k in reversed(range(n)):
                live_after[k]=seen
                mask=0
                for state in ordered[k]:
                    tick('setup_work')
                    mask |= 1 << (state//4)
                tick('setup_work',2)
                expires[k]=mask & ~seen
                seen |= mask
            if identity_lifetime=='require' and seen.bit_count()!=n:
                return result('infeasible','piece_missing')
        tracked=set(range(n)) if abstraction=='exact' else {s//4 for s in p.fixed.values()}
        group_for_piece={}
        if groups:
            buckets={}
            for piece in range(n):
                tick('setup_work',4)
                face=p.faces[4*piece]
                key=(face.count(0),min((color for color in face if color),default=0))
                buckets.setdefault(key,[]).append(piece)
            offset=0
            for members in buckets.values():
                if not 1<len(members)<n:continue
                capacity=len(members);mask=(1<<capacity.bit_length())-1
                for member in members:
                    tick('setup_work');group_for_piece[member]=(offset,mask,capacity)
                offset+=capacity.bit_length()
        blocked_pairs=[]
        while True:
            tick('state_operations');c['diagrams']+=1;c['tracked_pieces']=len(tracked)
            tracked_mask=sum(1<<piece for piece in tracked)
            starts={};ends={};clears={}
            for index,(start_cell,end_cell,piece) in enumerate(blocked_pairs):
                tick('setup_work',3);flag=1<<index
                starts[start_cell,piece]=starts.get((start_cell,piece),0)|flag
                ends[end_cell,piece]=ends.get((end_cell,piece),0)|flag
                clears[end_cell]=clears.get(end_cell,0)|flag
            nodes=[(-1,-1,0)];layer={(0,(),0,0):0};c['total_nodes']+=1
            for k,cell in enumerate(order):
                next_layer={};tests,carry=instructions[k]
                required=0
                if identity_lifetime=='require':
                    tick('state_operations')
                    required=expires[k] & tracked_mask
                for (used,colors,resource_counts,pair_flags),parent in layer.items():
                    for state in ordered[k]:
                        tick('candidate_placements');piece_bit=1<<(state//4)
                        if used&piece_bit:continue
                        next_counts=resource_counts;next_flags=pair_flags
                        if groups:
                            tick('state_operations')
                            group=group_for_piece.get(state//4)
                            if group:
                                offset,mask,capacity=group;tick('state_operations',3)
                                if (resource_counts>>offset)&mask>=capacity:
                                    c['group_failures']+=1;continue
                                next_counts+=1<<offset
                        if blocked_pairs:
                            tick('state_operations',3)
                            if ends.get((cell,state//4),0)&pair_flags:
                                c['local_pair_failures']+=1;continue
                            next_flags=(pair_flags & ~clears.get(cell,0)) | starts.get((cell,state//4),0)
                        face=p.faces[state];valid=True
                        for position,side in tests:
                            tick('edge_checks')
                            if colors[position]!=face[side]:valid=False;break
                        if not valid:continue
                        new_colors=tuple(face[index] if fresh else colors[index] for fresh,index in carry)
                        successor_used=used|(piece_bit&tracked_mask)
                        if identity_lifetime!='full':
                            if required:
                                tick('state_operations')
                                if required & ~successor_used:
                                    c['expired_piece_failures']+=1
                                    continue
                            tick('state_operations')
                            successor_used &= live_after[k]
                        key=(successor_used,new_colors,next_counts,next_flags);tick('state_operations',len(carry)+1)
                        if key in next_layer:
                            c['merged_paths']+=1;tick('state_operations')
                            # Representative ordering only: prefer diversity, never
                            # remove a path from the relaxed feasibility relation.
                            previous=next_layer[key];all_used=nodes[parent][2]|piece_bit
                            if all_used.bit_count()>nodes[previous][2].bit_count():nodes[previous]=(parent,state,all_used)
                            continue
                        if len(nodes)>=max_nodes:raise _Limit('node_limit')
                        next_layer[key]=len(nodes);nodes.append((parent,state,nodes[parent][2]|piece_bit));c['total_nodes']+=1
                c['peak_layer_nodes']=max(c['peak_layer_nodes'],len(next_layer));c['max_depth']=max(c['max_depth'],k+1)
                if not next_layer:return result('infeasible','relaxation_exhausted')
                layer=next_layer
            node=next(iter(layer.values()));candidate=[None]*n
            for cell in reversed(order):
                tick('witness_checks');node,state,_=nodes[node];candidate[cell]=state
            # A valid witness is retained even if cancellation arrives now.
            if check_solution(p,candidate):board=candidate;return result('solved','validated_solution')
            frequency={}
            for state in candidate:tick('witness_checks');frequency[state//4]=frequency.get(state//4,0)+1
            repeated=sorted((piece for piece,count in frequency.items() if count>1),key=lambda piece:(-frequency[piece],piece))
            if not repeated or any(piece in tracked for piece in repeated):raise RuntimeError('Invalid diagram abstraction witness')
            if c['refinements']>=max_refinements:raise _Limit('refinement_limit')
            if refinement=='local':
                for piece in repeated[:refine_batch]:
                    tick('state_operations',n)
                    occurrences=[cell for cell in order if candidate[cell]//4==piece]
                    blocked_pairs.append((occurrences[0],occurrences[-1],piece))
                c['local_pairs']=len(blocked_pairs)
            else:tracked.update(repeated[:refine_batch])
            c['refinements']+=1
    except _Limit as limit:return result('stopped' if limit.reason=='stopped' else 'timeout',limit.reason)
