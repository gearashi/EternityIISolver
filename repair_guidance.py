"""Seeded search-quality experiments; heuristic scores never establish pruning.

Every candidate preserves physical identities, frame domains and oriented clues.
Ejection work is reported in complete candidate evaluations plus path/local probes,
so a multi-piece proposal is not misreported as one cheap primitive move.
"""
from __future__ import annotations
from collections import deque
from dataclasses import replace
import itertools
import random
from exact_model import check_solution, constraint_edges, grid_edges, problem_hash, frame_ok


def _integer(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value


def _stop_callback(callback):
    if callback is not None and not callable(callback):
        raise ValueError('should_stop must be callable')
    return callback or (lambda: False)


class _Stopped(Exception):
    pass


def weighted_independent_set(size, weights, *, excluded=(), seed=0, should_stop=None):
    """Exact maximum integer-weight independent set of an induced square grid.

    The source/left/right/sink min-cut encodes weighted vertex cover. A bounded
    seeded secondary objective breaks primary ties without sacrificing weight.
    Cancellation returns [], which is not claimed to be an optimum.
    """
    _integer('size', size, 1, 16); _integer('seed', seed, 0, 2**32-1)
    weights=list(itertools.islice(weights, size*size+1))
    if len(weights)!=size*size:
        raise ValueError('One weight per cell is required')
    for weight in weights: _integer('weight', weight, 0, 10_000_000)
    excluded=list(itertools.islice(excluded,size*size+1))
    if len(excluded)>size*size or any(type(c) is not int or not 0<=c<size*size for c in excluded):
        raise ValueError('Invalid excluded cells')
    stopped=_stop_callback(should_stop)
    if stopped(): return []
    excluded=set(excluded); rng=random.Random(seed); n=size*size
    scale=n*255+1
    capacities=[weights[c]*scale+rng.randrange(256) for c in range(n)]
    active=[c for c in range(n) if c not in excluded]
    source=n; sink=n+1; graph=[[] for _ in range(n+2)]
    def add(a,b,capacity):
        graph[a].append([b,len(graph[b]),capacity])
        graph[b].append([a,len(graph[a])-1,0])
    left={c for c in active if (c//size+c%size)%2==0}
    infinite=sum(capacities[c] for c in active)+1
    for c in active:
        add(source,c,capacities[c]) if c in left else add(c,sink,capacities[c])
    for a,b,_,_ in grid_edges(size):
        if a not in excluded and b not in excluded:
            add(a,b,infinite) if a in left else add(b,a,infinite)
    try:
        while True:
            if stopped(): raise _Stopped
            level=[-1]*(n+2);level[source]=0;queue=deque([source])
            while queue:
                a=queue.popleft()
                for b,_,capacity in graph[a]:
                    if capacity and level[b]<0:level[b]=level[a]+1;queue.append(b)
            if level[sink]<0:break
            position=[0]*(n+2)
            def send(a,flow):
                if a==sink:return flow
                while position[a]<len(graph[a]):
                    edge=graph[a][position[a]];b,rev,capacity=edge
                    if capacity and level[b]==level[a]+1:
                        sent=send(b,min(flow,capacity))
                        if sent:
                            edge[2]-=sent;graph[b][rev][2]+=sent;return sent
                    position[a]+=1
                return 0
            while True:
                if stopped():raise _Stopped
                if not send(source,infinite):break
        reached={source};queue=deque([source])
        while queue:
            if stopped():raise _Stopped
            for b,_,capacity in graph[queue.popleft()]:
                if capacity and b not in reached:reached.add(b);queue.append(b)
        # Complement of (left not reachable) union (right reachable).
        return sorted(c for c in active if (c in reached)==(c in left))
    except _Stopped:
        return []


class Guidance:
    """Detached structural data shared by the optional guidance operators."""
    def __init__(self,problem):
        if set(constraint_edges(problem))!=set(grid_edges(problem.size)):
            raise ValueError('Guidance requires the full-grid problem')
        self.problem=replace(problem,faces=tuple(tuple(f) for f in problem.faces),
                             domains=tuple(tuple(d) for d in problem.domains),
                             fixed=dict(problem.fixed),edges=())
        self.size=problem.size;self.n=self.size**2;self.edges=grid_edges(self.size)
        self.mutable=tuple(c for c in range(self.n) if c not in self.problem.fixed)
        self.allowed=tuple(frozenset(s for s in d if frame_ok(self.size,c,self.problem.faces[s])) for c,d in enumerate(self.problem.domains))
        self.incident=[[] for _ in range(self.n)];self.neighbors=[set() for _ in range(self.n)]
        for index,(a,b,sa,sb) in enumerate(self.edges):
            self.incident[a].append((index,b,sa,sb));self.incident[b].append((index,a,sb,sa))
            self.neighbors[a].add(b);self.neighbors[b].add(a)
        self.fingerprint=problem_hash(replace(self.problem,edges=None))

    def require(self,board):
        if not check_solution(self.problem,board):
            raise ValueError('Board violates physical pieces, frame, domains or fixed clues')

    def edge_weights(self,weights):
        if weights is None:return (1,)*len(self.edges)
        values=tuple(itertools.islice(weights,len(self.edges)+1))
        if len(values)!=len(self.edges):raise ValueError('One positive weight per edge is required')
        for weight in values:_integer('edge weight',weight,1,1_000_001)
        return values

    def matches(self,board):
        faces=self.problem.faces
        return tuple(faces[board[a]][sa]==faces[board[b]][sb] for a,b,sa,sb in self.edges)

    def weighted_independent_cells(self,board,*,seed=0,edge_weights=None,should_stop=None):
        self.require(board);weights=self.edge_weights(edge_weights);stopped=_stop_callback(should_stop)
        values=[1]*self.n;matched=self.matches(board)
        for index,(a,b,_,_) in enumerate(self.edges):
            if not matched[index]:values[a]+=weights[index];values[b]+=weights[index]
        # Nonnegative frustration weights; no candidate or branch is deleted.
        return weighted_independent_set(self.size,values,excluded=self.problem.fixed,
                                        seed=seed,should_stop=stopped)

    def corridor_cells(self,board,*,seed=0,max_cells=20,should_stop=None):
        """Connect separated mismatch clusters through mutable cells, then grow.

        The returned set is connected and excludes all clues. If obstacles or
        the cap prevent reaching another cluster, it grows a local region.
        """
        self.require(board);_integer('seed',seed,0,2**32-1);_integer('max_cells',max_cells,1,64)
        stopped=_stop_callback(should_stop);rng=random.Random(seed);mutable=set(self.mutable)
        matched=self.matches(board);bad={c:0 for c in mutable}
        for i,(a,b,_,_) in enumerate(self.edges):
            if not matched[i]:
                if a in bad:bad[a]+=1
                if b in bad:bad[b]+=1
        anchors=[c for c in self.mutable if bad[c]]
        if not anchors or stopped():return []
        tie={c:rng.random() for c in self.mutable}
        start=max(anchors,key=lambda c:(bad[c],tie[c]));cluster={start};queue=deque([start])
        while queue:
            if stopped():return []
            for c in sorted(self.neighbors[queue.popleft()]):
                if c in bad and bad[c] and c not in cluster:cluster.add(c);queue.append(c)
        parent={start:None};distance={start:0};queue=deque([start])
        while queue:
            if stopped():return []
            a=queue.popleft()
            if distance[a]>=max_cells-1:continue
            for b in sorted(self.neighbors[a],key=lambda c:tie.get(c,0)):
                if b in mutable and b not in parent:
                    parent[b]=a;distance[b]=distance[a]+1;queue.append(b)
        targets=[c for c in anchors if c in parent and c not in cluster]
        if not targets:targets=[c for c in anchors if c in parent]
        target=max(targets,key=lambda c:(distance[c],bad[c],tie[c]))
        core=set()
        while target is not None:core.add(target);target=parent[target]
        frontier=set().union(*(self.neighbors[c] for c in core))&mutable-core
        while frontier and len(core)<max_cells:
            if stopped():return []
            cell=max(frontier,key=lambda c:(bad[c],len(self.neighbors[c]&core),tie[c]))
            core.add(cell);frontier=(frontier|self.neighbors[cell])&mutable-core
        return sorted(core)

    def ejection_repair(self,board,*,seed=0,max_cycle=6,beam_width=4,
                        max_evaluations=256,max_expansions=512,rotations_per_cycle=8,
                        edge_weights=None,should_stop=None):
        """Try bounded directed 3..6-piece cycles with legal rotated recipients.

        Local placement estimates rank paths only. Every complete candidate is
        evaluated exactly on the union of changed incident edges. A weighted
        improvement may lower its true score, so callers must protect their own
        independently validated true-score incumbent (best_true_board/score). No neighborhood optimum
        or infeasibility claim is made by this bounded heuristic.
        """
        self.require(board);_integer('seed',seed,0,2**32-1);_integer('max_cycle',max_cycle,3,6)
        _integer('beam_width',beam_width,1,16);_integer('max_evaluations',max_evaluations,1,100_000)
        _integer('max_expansions',max_expansions,1,10_000);_integer('rotations_per_cycle',rotations_per_cycle,1,4096)
        stopped=_stop_callback(should_stop);weights=self.edge_weights(edge_weights);rng=random.Random(seed)
        original=list(board);matches=self.matches(original);initial=sum(matches)
        guided_initial=sum(w for w,m in zip(weights,matches) if m)
        best=original.copy();best_score=initial;best_guided=guided_initial;best_cycle=[]
        best_true=original.copy();best_true_score=initial
        evaluations=expansions=probes=cycles=0;termination='completed_bounded_scan'
        faces=self.problem.faces;recent=set();incoming_cache={}
        bad={c:sum(weights[i] for i,_,_,_ in self.incident[c] if not matches[i]) for c in self.mutable}
        tie={c:rng.random() for c in self.mutable}
        anchors=sorted((c for c in self.mutable if bad[c]),key=lambda c:(-bad[c],tie[c]))[:8]
        def incoming(cell,piece):
            nonlocal probes
            key=(cell,piece)
            if key not in incoming_cache:
                options=[]
                for state in range(4*piece,4*piece+4):
                    if state in self.allowed[cell]:
                        probes+=1
                        value=sum(weights[i] for i,other,side,opposite in self.incident[cell]
                                  if faces[state][side]==faces[original[other]][opposite])
                        options.append((value,rng.random(),state))
                incoming_cache[key]=sorted(options,reverse=True)
            return incoming_cache[key]
        def limit():
            if stopped():return 'stopped'
            if evaluations>=max_evaluations:return 'candidate_limit'
            if expansions>=max_expansions:return 'expansion_limit'
            return None
        for anchor in anchors:
            reason=limit()
            if reason:termination=reason;break
            frontier=[(0,(anchor,))]
            for depth in range(2,min(max_cycle,len(self.mutable))+1):
                next_paths=[]
                for path_score,path in frontier:
                    target=path[-1];candidates=[]
                    current=sum(weights[i] for i,_,_,_ in self.incident[target] if matches[i])
                    for donor in self.mutable:
                        if stopped():termination='stopped';break
                        if donor in path:continue
                        options=incoming(target,original[donor]//4)
                        if options:candidates.append((options[0][0]-current,bad[donor],tie[donor],donor))
                    if termination=='stopped':break
                    for gain,_,_,donor in sorted(candidates,reverse=True)[:beam_width]:
                        reason=limit()
                        if reason:termination=reason;break
                        expansions+=1;next_paths.append((path_score+gain,(*path,donor)))
                    if termination!='completed_bounded_scan':break
                if termination!='completed_bounded_scan':break
                frontier=sorted(next_paths,reverse=True)[:beam_width]
                if depth<3:continue
                for _,path in frontier:
                    reason=limit()
                    if reason:termination=reason;break
                    # Only one representation per directed cycle; reversal remains
                    # a distinct physical permutation for lengths >=3.
                    offset=path.index(min(path));key=path[offset:]+path[:offset]
                    if key in recent:continue
                    recent.add(key);cycles+=1
                    options=[incoming(cell,original[path[(index+1)%len(path)]]//4) for index,cell in enumerate(path)]
                    if not all(options):continue
                    affected=sorted({i for cell in path for i,_,_,_ in self.incident[cell]})
                    baseline=sum(matches[i] for i in affected)
                    weighted_baseline=sum(weights[i] for i in affected if matches[i])
                    for chosen in itertools.islice(itertools.product(*[[item[2] for item in row] for row in options]),rotations_per_cycle):
                        reason=limit()
                        if reason:termination=reason;break
                        candidate=original.copy()
                        for cell,state in zip(path,chosen):candidate[cell]=state
                        evaluations+=1;local=weighted=0
                        for i in affected:
                            a,b,sa,sb=self.edges[i]
                            if faces[candidate[a]][sa]==faces[candidate[b]][sb]:local+=1;weighted+=weights[i]
                        value=initial+local-baseline;guided=guided_initial+weighted-weighted_baseline
                        if value>best_true_score:
                            best_true=candidate.copy();best_true_score=value
                        if (guided,value)>(best_guided,best_score):
                            best=candidate;best_score=value;best_guided=guided;best_cycle=list(path)
                        if best_true_score==len(self.edges):
                            termination='perfect_board_validated';break
                    if termination!='completed_bounded_scan':break
                if termination!='completed_bounded_scan' or not frontier:break
            if termination!='completed_bounded_scan':break
        self.require(best);self.require(best_true)
        if sum(self.matches(best_true))!=best_true_score:
            raise RuntimeError('Ejection true-score incumbent failed independent validation')
        verified=self.matches(best)
        if sum(verified)!=best_score or sum(w for w,m in zip(weights,verified) if m)!=best_guided:
            raise RuntimeError('Ejection affected-edge objective disagrees with full-grid validation')
        return dict(board=best,score=best_score,reference_score=initial,delta=best_score-initial,
                    best_true_board=best_true,best_true_score=best_true_score,
                    guided_score=best_guided,guided_reference_score=guided_initial,changed=best!=original,
                    cycle=best_cycle,relocated_pieces=len(best_cycle),candidate_evaluations=evaluations,
                    path_expansions=expansions,heuristic_state_evaluations=probes,cycles_considered=cycles,
                    optimal=False,termination=termination)


class EdgePenalties:
    """Bounded, decaying integer memory of unresolved edges; never a proof fact."""
    SCHEMA='eternity-edge-penalties/v1'
    def __init__(self,problem,*,decay_every=32,max_penalty=255):
        _integer('decay_every',decay_every,1,65536);_integer('max_penalty',max_penalty,1,1_000_000)
        self.guidance=Guidance(problem);self.decay_every=decay_every;self.max_penalty=max_penalty
        self.observations=0;self.values=[0]*len(self.guidance.edges)

    def observe(self,board):
        self.guidance.require(board)
        if self.observations>=2**53-1:raise ValueError('Penalty observation limit exceeded')
        self.observations+=1
        if self.observations%self.decay_every==0:self.values=[v//2 for v in self.values]
        self.values=[min(self.max_penalty,v+int(not matched)) for v,matched in zip(self.values,self.guidance.matches(board))]

    def weights(self):return tuple(1+v for v in self.values)

    def evaluate(self,board):
        self.guidance.require(board);matched=self.guidance.matches(board)
        return dict(score=sum(matched),guided_score=sum(w for w,m in zip(self.weights(),matched) if m))

    def export_state(self):
        return dict(schema=self.SCHEMA,problem=self.guidance.fingerprint,decay_every=self.decay_every,
                    max_penalty=self.max_penalty,observations=self.observations,values=list(self.values))

    def restore_state(self,state):
        required={'schema','problem','decay_every','max_penalty','observations','values'}
        if type(state) is not dict or set(state)!=required:raise ValueError('Invalid penalty state fields')
        if (state['schema']!=self.SCHEMA or state['problem']!=self.guidance.fingerprint or
                type(state['decay_every']) is not int or state['decay_every']!=self.decay_every or
                type(state['max_penalty']) is not int or state['max_penalty']!=self.max_penalty):
            raise ValueError('Penalty state belongs to another problem or configuration')
        _integer('observations',state['observations'],0,2**53-1)
        values=state['values']
        if type(values) is not list or len(values)!=len(self.values):raise ValueError('Invalid penalty values')
        for value in values:_integer('penalty',value,0,self.max_penalty)
        if any(v>state['observations'] for v in values):raise ValueError('Penalty exceeds observed updates')
        self.observations=state['observations'];self.values=list(values)


def weighted_independent_cells(problem,board,**kwargs):return Guidance(problem).weighted_independent_cells(board,**kwargs)
def corridor_cells(problem,board,**kwargs):return Guidance(problem).corridor_cells(board,**kwargs)
def ejection_repair(problem,board,**kwargs):return Guidance(problem).ejection_repair(board,**kwargs)
