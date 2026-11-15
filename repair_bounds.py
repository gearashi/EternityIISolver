"""Sound bounded regional score upper bounds, with explicit reasoning work.

Assignment relaxes unresolved edge equality but keeps physical-piece uniqueness.
Compatibility graphs enforce equalities; removing one endpoint per mismatched
edge leaves a clique, so a proved clique upper bound also bounds mismatches.
"""
from __future__ import annotations
from dataclasses import replace
from types import MappingProxyType
import itertools
from exact_model import constraint_edges, grid_edges, frame_ok, problem_hash
from exact_optimize import evaluate_board


class _Limit(Exception):
    pass


def _integer(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')


class _Work:
    def __init__(self, maximum, stop, names):
        _integer('max_work', maximum, 0, 1000000)
        if stop is not None and not callable(stop): raise ValueError('should_stop must be callable')
        self.maximum=maximum;self.stop=stop;self.used=0;self.counts=dict.fromkeys(names,0)
    def check(self):
        if self.stop is not None and self.stop(): raise _Limit('stopped')
    def take(self,name):
        self.check()
        if self.used >= self.maximum: raise _Limit('work_limit')
        self.used+=1;self.counts[name]+=1


class RegionBounds:
    """Reusable immutable model of one legal selected-cell/piece-pool region."""
    def __init__(self, problem, reference, cells):
        if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
            raise ValueError('Score bounds require the full grid')
        checked=evaluate_board(problem,reference)
        if not checked['valid']: raise ValueError('Illegal reference board')
        cells=list(itertools.islice(iter(cells),25))
        if (len(cells)>24 or any(type(c) is not int or not 0<=c<len(reference) for c in cells)
                or len(set(cells))!=len(cells) or set(cells)&problem.fixed.keys()):
            raise ValueError('Expected at most24 distinct mutable cells')
        self._problem=replace(problem,faces=tuple(tuple(f) for f in problem.faces),
                domains=tuple(tuple(d) for d in problem.domains),fixed=MappingProxyType(dict(problem.fixed)),edges=None)
        self.fingerprint=problem_hash(self._problem)
        self.reference=tuple(reference);self.cells=tuple(sorted(cells));self.pool=frozenset(reference[c]//4 for c in cells)
        self.domains=MappingProxyType({c:tuple(dict.fromkeys(s for s in self._problem.domains[c]
                            if s//4 in self.pool and frame_ok(self._problem.size,c,self._problem.faces[s])))
                            for c in self.cells})
        self.edges=grid_edges(problem.size);self.faces=self._problem.faces
        selected=set(self.cells)
        self.changing=tuple(e for e in self.edges if e[0] in selected or e[1] in selected)
        self.outside_score=sum(self.faces[self.reference[a]][sa]==self.faces[self.reference[b]][sb]
                               for a,b,sa,sb in self.edges if a not in selected and b not in selected)
        self.incident={c:[] for c in self.cells}
        for a,b,sa,sb in self.edges:
            if a in self.incident:self.incident[a].append((b,sa,sb))
            if b in self.incident:self.incident[b].append((a,sb,sa))
        self.setup_state_evaluations=sum(len(self._problem.domains[c]) for c in self.cells)
        self.setup_edges=3*len(self.edges)  # Changing edges, outside score, incident lists.

    def _prefix(self,assignments,work):
        if type(assignments) is not dict: raise ValueError('Assignments must be a dict')
        prefix=dict(assignments)
        if any(type(c) is not int or c not in self.domains or type(s) is not int or s not in self.domains[c]
               for c,s in prefix.items()):raise ValueError('Illegal prefix placement')
        used={s//4 for s in prefix.values()}
        if len(used)!=len(prefix):raise ValueError('Prefix repeats a physical piece')
        board=list(self.reference)
        for c,s in prefix.items():board[c]=s
        remaining=tuple(c for c in self.cells if c not in prefix);remset=set(remaining)
        available=tuple(sorted(self.pool-used))
        fixed_score=self.outside_score;unresolved=internal=0
        for a,b,sa,sb in self.changing:
            work.take('prefix_edge_checks')
            if a not in remset and b not in remset:fixed_score+=self.faces[board[a]][sa]==self.faces[board[b]][sb]
            else:unresolved+=1
            if a in remset and b in remset:internal+=1
        return board,remaining,remset,available,fixed_score,unresolved,internal

    @staticmethod
    def _result(work,complete,termination,upper=None,impossible=False,**extra):
        return dict(complete=complete,termination=termination,upper_bound=upper,
                    structurally_infeasible=impossible,work=work.used,**work.counts,**extra)

    def assignment_upper(self,assignments,*,max_work=20000,should_stop=None):
        """Max boundary score under a perfect cell/piece assignment + free edges.

        Missing edges have a finite penalty larger than any legal total. The
        integer Hungarian algorithm obtains a global optimum; a forbidden edge
        in that optimum proves that no perfect physical assignment exists.
        """
        w=_Work(max_work,should_stop,('prefix_edge_checks','matrix_state_evaluations','matching_relaxations'))
        try:
            w.check()
            board,cells,selected,pool,fixed_score,unresolved,internal=self._prefix(assignments,w)
            n=len(cells);weights=[]
            if not n:return self._result(w,True,'complete',fixed_score)
            for c in cells:
                scores={p:None for p in pool}
                for state in self.domains[c]:
                    if state//4 not in scores:continue
                    w.take('matrix_state_evaluations')
                    value=sum(self.faces[state][side]==self.faces[board[o]][os]
                              for o,side,os in self.incident[c] if o not in selected)
                    piece=state//4
                    if scores[piece] is None or value>scores[piece]:scores[piece]=value
                weights.append([scores[p] for p in pool])
            if any(all(v is None for v in row) for row in weights):
                return self._result(w,True,'no_physical_assignment',impossible=True)
            big=4*n+1
            costs=[[big if v is None else 4-v for v in row] for row in weights]
            u=[0]*(n+1);v=[0]*(n+1);owner=[0]*(n+1);way=[0]*(n+1)
            for row in range(1,n+1):
                owner[0]=row;j0=0;minimum=[big*(n+2)]*(n+1);visited=[False]*(n+1)
                while True:
                    visited[j0]=True;i0=owner[j0];delta=big*(n+2);j1=0
                    for j in range(1,n+1):
                        if visited[j]:continue
                        w.take('matching_relaxations')
                        reduced=costs[i0-1][j-1]-u[i0]-v[j]
                        if reduced<minimum[j]:minimum[j]=reduced;way[j]=j0
                        if minimum[j]<delta:delta=minimum[j];j1=j
                    for j in range(n+1):
                        if visited[j]:u[owner[j]]+=delta;v[j]-=delta
                        else:minimum[j]-=delta
                    j0=j1
                    if owner[j0]==0:break
                while True:
                    j1=way[j0];owner[j0]=owner[j1];j0=j1
                    if not j0:break
            chosen=[(owner[j]-1,j-1) for j in range(1,n+1)]
            if any(weights[i][j] is None for i,j in chosen):
                return self._result(w,True,'no_physical_assignment',impossible=True)
            upper=fixed_score+internal+sum(weights[i][j] for i,j in chosen)
            return self._result(w,True,'complete',upper,matching_size=n)
        except _Limit as exc:return self._result(w,False,str(exc))

    def clique_upper(self,assignments,*,max_work=20000,max_vertices=256,should_stop=None):
        """Exact bounded maximum clique supplies an admissible mismatch bound.

        Each vertex is one remaining cell/state satisfying all frozen neighbors.
        An edge joins distinct cells/pieces with matching mutual edge, if any.
        If a legal regional board has m unresolved mismatches, discard at most
        one remaining endpoint per mismatch. Its other n-m placements form a
        clique. Thus m >= n-omega. All graph construction and clique search must
        complete; an interrupted result supplies no upper score certificate.
        """
        _integer('max_vertices',max_vertices,1,512)
        w=_Work(max_work,should_stop,('prefix_edge_checks','graph_state_evaluations','graph_pair_checks','graph_color_steps','clique_nodes'))
        vertices=[];best=[];adj=[]
        try:
            w.check()
            board,cells,selected,pool,fixed_score,unresolved,_=self._prefix(assignments,w)
            n=len(cells)
            if not n:return self._result(w,True,'complete',fixed_score,clique_size=0,completion={})
            for c in cells:
                for state in self.domains[c]:
                    if state//4 not in pool:continue
                    w.take('graph_state_evaluations')
                    if all(self.faces[state][side]==self.faces[board[o]][os]
                           for o,side,os in self.incident[c] if o not in selected):
                        if len(vertices)>=max_vertices:raise _Limit('vertex_limit')
                        vertices.append((c,state))
            adj=[0]*len(vertices)
            sides={(a,b):(sa,sb) for a,b,sa,sb in self.edges}
            sides.update({(b,a):(sb,sa) for a,b,sa,sb in self.edges})
            for i,(c,s) in enumerate(vertices):
                for j in range(i):
                    w.take('graph_pair_checks');d,t=vertices[j]
                    if c==d or s//4==t//4:continue
                    edge=sides.get((c,d))
                    if edge is None or self.faces[s][edge[0]]==self.faces[t][edge[1]]:
                        adj[i]|=1<<j;adj[j]|=1<<i
            def color_order(candidates):
                order=[];bounds=[];color=0
                while candidates:
                    color+=1;available=candidates
                    while available:
                        w.take('graph_color_steps')
                        flag=available&-available;index=flag.bit_length()-1
                        candidates^=flag;available^=flag;available&=~adj[index]
                        order.append(index);bounds.append(color)
                return order,bounds
            def expand(path,candidates):
                nonlocal best
                if len(path)+candidates.bit_count()<=len(best):return
                order,bounds=color_order(candidates)
                for k in range(len(order)-1,-1,-1):
                    if len(path)+bounds[k]<=len(best):return
                    vertex=order[k];w.take('clique_nodes')
                    next_path=path+[vertex];next_candidates=candidates&adj[vertex]
                    if not next_candidates:
                        if len(next_path)>len(best):best=next_path
                    else:expand(next_path,next_candidates)
                    if len(best)==n:return
                    candidates&=~(1<<vertex)
            expand([], (1<<len(vertices))-1)
            w.check()
            completion={vertices[i][0]:vertices[i][1] for i in best} if len(best)==n else None
            return self._result(w,True,'complete',fixed_score+unresolved-(n-len(best)),
                                clique_size=len(best),vertices=len(vertices),completion=completion)
        except _Limit as exc:return self._result(w,False,str(exc),vertices=len(vertices),completion=None)
