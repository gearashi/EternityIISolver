"""Bounded explained DFS with conflict-directed backjumping and learned nogoods.

Every removed state carries a set of currently assumed placement literals that
is sufficient to exclude it. Domain, frame/clue, adjacency and physical-piece rules are used here.
Optional matching propagation explains Hall shortages and SCC deletions using
closed Hall sets; no unexplained global deletion enters a learned nogood. A failed domain is explained by all of its initial values;
branch resolution includes values removed before that branching decision.
Learned nogoods are local to this immutable problem and are never imported.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import replace
import math
import random
import time
from types import MappingProxyType

from exact_model import check_solution, constraint_edges, frame_ok, problem_hash


class _Limit(Exception):
    def __init__(self, reason):self.reason=reason


def _bits(mask):
    while mask:
        flag=mask&-mask;yield flag.bit_length()-1;mask^=flag


def _integer(value,name,low,high):
    if type(value) is not int or not low<=value<=high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value


class _Search:
    def __init__(self,problem,seconds,seed,workers,should_stop,on_progress,learning,
                 max_work,max_nogoods,max_nogood_literals,include_nogoods,matching=False):
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
            raise ValueError('seconds must be finite and nonnegative')
        _integer(seed,'seed',0,2**32-1);_integer(workers,'workers',1,64)
        _integer(max_work,'max_work',0,1000000000)
        _integer(max_nogoods,'max_nogoods',0,4096)
        _integer(max_nogood_literals,'max_nogood_literals',1,256)
        if any(type(v) is not bool for v in (learning,include_nogoods,matching)):
            raise ValueError('learning, include_nogoods and matching must be booleans')
        if should_stop is not None and not callable(should_stop):raise ValueError('Invalid stop callback')
        if on_progress is not None and not callable(on_progress):raise ValueError('Invalid progress callback')
        self.started=time.monotonic();self.deadline=self.started+seconds
        self.problem=replace(problem,faces=tuple(tuple(f) for f in problem.faces),
            domains=tuple(tuple(d) for d in problem.domains),fixed=MappingProxyType(dict(problem.fixed)),
            edges=None if problem.edges is None else tuple(tuple(e) for e in problem.edges))
        self.n=problem.size**2;self.faces=self.problem.faces
        self.identity=problem_hash(self.problem);self.seed=seed;self.requested_workers=workers
        self.should_stop=should_stop;self.on_progress=on_progress;self.learning=learning;self.matching=matching
        self.max_work=max_work;self.max_nogoods=max_nogoods;self.max_nogood_literals=max_nogood_literals
        self.include_nogoods=include_nogoods;self.work=0;self.nogoods=OrderedDict()
        self.branches=0;self.conflicts=0;self.max_depth=0;self.last_progress=self.started
        self.counters=dict(setup_state_visits=0,domain_copies=0,reason_copies=0,
            arc_state_checks=0,piece_state_checks=0,reason_literals=0,reason_joins=0,
            removed_states=0,nogood_literal_checks=0,learned_nogoods=0,
            duplicate_nogoods=0,skipped_large_nogoods=0,evicted_nogoods=0,
            nogood_rejections=0,nogood_unit_removals=0,backjumped_levels=0,
            decision_placements=0,propagation_rounds=0,matching_state_visits=0,
            matching_edge_visits=0,matching_graph_visits=0,matching_reason_checks=0,
            matching_checks=0,matching_hall_failures=0,matching_hall_reasons=0,
            matching_removed_states=0)
        self.base=[];self.adj=[[] for _ in range(self.n)];self.colors=[{} for _ in range(4)]
        self.ties=list(range(self.n));random.Random(seed).shuffle(self.ties)
        self.rank={c:i for i,c in enumerate(self.ties)}

    def report(self,phase='searching'):
        return dict(phase=phase,branches=self.branches,conflicts=self.conflicts,
            max_depth=self.max_depth,propagations=self.counters['removed_states'],
            reasoning_counters_session=dict(self.counters,work=self.work),
            elapsed_seconds=time.monotonic()-self.started,learning=self.learning,matching=self.matching)

    def check(self):
        if self.should_stop is not None and self.should_stop():raise _Limit('stopped')
        if time.monotonic()>=self.deadline:raise _Limit('timeout')
        now=time.monotonic()
        if self.on_progress is not None and now-self.last_progress>=.25:
            self.last_progress=now;self.on_progress(self.report())

    def tick(self,kind,amount=1):
        if amount<0:raise AssertionError('Negative work')
        self.check()
        if self.work+amount>self.max_work:raise _Limit('work_limit')
        self.work+=amount;self.counters[kind]+=amount

    def union(self,reasons):
        result=set()
        for reason in reasons:
            self.tick('reason_joins')
            self.tick('reason_literals',len(reason))
            result.update(reason)
        return frozenset(result)

    def setup(self):
        self.check()
        fixed_pieces={s//4 for s in self.problem.fixed.values()}
        for c,domain in enumerate(self.problem.domains):
            mask=0
            for s in domain:
                self.tick('setup_state_visits')
                if (frame_ok(self.problem.size,c,self.faces[s]) and
                    (s==self.problem.fixed[c] if c in self.problem.fixed else s//4 not in fixed_pieces)):
                    mask|=1<<s
            self.base.append(mask)
        for s,face in enumerate(self.faces):
            self.tick('setup_state_visits')
            for side,color in enumerate(face):self.colors[side][color]=self.colors[side].get(color,0)|(1<<s)
        for a,b,sa,sb in constraint_edges(self.problem):
            self.adj[a].append((b,sa,sb));self.adj[b].append((a,sb,sa))
        return self.base.copy(),[{} for _ in range(self.n)]

    def unavailable(self,c,mask,reasons):
        # All requested states must actually have been removed from this cell.
        return self.union(reasons[c][s] for s in _bits(mask))

    def force_reason(self,c,state,domains,reasons):
        assert domains[c]==1<<state
        return self.unavailable(c,self.base[c]&~(1<<state),reasons)

    def remove(self,c,state,why,domains,reasons):
        flag=1<<state
        if not domains[c]&flag:return False
        self.tick('removed_states')
        domains[c]^=flag;reasons[c][state]=why
        return True

    def wipeout(self,c,reasons):
        return self.unavailable(c,self.base[c],reasons)

    def learn(self,why):
        if not self.learning or not self.max_nogoods or not why:return
        if len(why)>self.max_nogood_literals:
            self.counters['skipped_large_nogoods']+=1;return
        clause=tuple(sorted(why))
        self.tick('reason_literals',len(clause))
        if clause in self.nogoods:
            self.nogoods.move_to_end(clause);self.counters['duplicate_nogoods']+=1;return
        self.nogoods[clause]=None;self.counters['learned_nogoods']+=1
        if len(self.nogoods)>self.max_nogoods:
            self.nogoods.popitem(last=False);self.counters['evicted_nogoods']+=1

    def conflict(self,why):
        self.conflicts+=1;self.learn(why);return why

    def hall_reason(self,cells,pieces,reasons):
        """Explain why these cells have no physical choices outside pieces.

        Every initially legal orientation outside the Hall pool must have been
        deleted, and all of its reasons must be retained. Inside-pool deletions
        are irrelevant: enlarging those domains cannot break the Hall argument.
        """
        lost=[]
        for cell in sorted(cells):
            for state in _bits(self.base[cell]):
                self.tick('matching_reason_checks')
                if state//4 not in pieces:lost.append(reasons[cell][state])
        self.counters['matching_hall_reasons']+=1
        return self.union(lost)

    def matching_filter(self,domains,reasons):
        """Return (changed, conflict) using a perfect matching and Hall reasons.

        All-different is on physical piece identities, never orientations.
        In this balanced graph, a non-matched edge is supported exactly when it
        belongs to an alternating cycle. For each unsupported edge c -> p, the
        cells reachable from p's owner form a tight Hall set excluding c. Its
        removed outside-pool alternatives explain why c cannot use p.
        """
        self.counters['matching_checks']+=1
        candidates=[]
        for domain in domains:
            pieces=set()
            for state in _bits(domain):
                self.tick('matching_state_visits');pieces.add(state//4)
            candidates.append(tuple(sorted(pieces)))
        matched=[-1]*self.n;owners=[-1]*self.n
        for root in sorted(range(self.n),key=lambda c:len(candidates[c])):
            queue=[root];parents={root:-1};seen_pieces=set();cursor=0;found=False
            while cursor<len(queue) and not found:
                cell=queue[cursor];cursor+=1
                self.tick('matching_graph_visits')
                for piece in candidates[cell]:
                    self.tick('matching_edge_visits')
                    if piece in seen_pieces:continue
                    seen_pieces.add(piece);owner=owners[piece]
                    if owner<0:
                        while cell>=0:
                            self.tick('matching_graph_visits')
                            old=matched[cell];matched[cell]=piece;owners[piece]=cell
                            cell=parents[cell];piece=old
                        found=True;break
                    if owner not in parents:parents[owner]=cell;queue.append(owner)
            if not found:
                # No reachable piece is free: the visited cells outnumber their
                # entire current neighbor pool. This is an exact Hall failure.
                assert len(parents)>len(seen_pieces)
                why=self.hall_reason(parents,seen_pieces,reasons)
                self.counters['matching_hall_failures']+=1
                return False,why
        # Collapse each matched piece->cell edge. Directed cell->owner(piece)
        # reachability and SCCs preserve all alternating-cycle information.
        graph=[[] for _ in range(self.n)];reverse=[[] for _ in range(self.n)]
        for cell,choices in enumerate(candidates):
            for piece in choices:
                self.tick('matching_edge_visits');owner=owners[piece]
                graph[cell].append(owner);reverse[owner].append(cell)
        seen=set();order=[]
        for root in range(self.n):
            if root in seen:continue
            seen.add(root);stack=[(root,0)]
            while stack:
                self.tick('matching_graph_visits');cell,index=stack[-1]
                if index==len(graph[cell]):order.append(cell);stack.pop();continue
                owner=graph[cell][index];stack[-1]=(cell,index+1)
                if owner not in seen:seen.add(owner);stack.append((owner,0))
        component=[-1]*self.n
        for root in reversed(order):
            if component[root]>=0:continue
            component[root]=root;stack=[root]
            while stack:
                cell=stack.pop();self.tick('matching_graph_visits')
                for other in reverse[cell]:
                    self.tick('matching_edge_visits')
                    if component[other]<0:component[other]=root;stack.append(other)
        explanations={};changed=False
        for cell,choices in enumerate(candidates):
            for piece in choices:
                self.tick('matching_edge_visits');owner=owners[piece]
                if component[cell]==component[owner]:continue
                key=component[owner]
                if key not in explanations:
                    cells={owner};stack=[owner]
                    while stack:
                        current=stack.pop();self.tick('matching_graph_visits')
                        for other in graph[current]:
                            self.tick('matching_edge_visits')
                            if other not in cells:cells.add(other);stack.append(other)
                    pool={matched[c] for c in cells}
                    # Closure guarantees all current choices of these cells
                    # lie in pool. They consume exactly that many pieces.
                    assert cell not in cells and piece in pool and len(cells)==len(pool)
                    explanations[key]=self.hall_reason(cells,pool,reasons)
                why=explanations[key]
                for state in tuple(_bits(domains[cell]&(15<<(4*piece)))):
                    if self.remove(cell,state,why,domains,reasons):
                        changed=True;self.counters['matching_removed_states']+=1
        return changed,None

    def propagate(self,domains,reasons):
        while True:
            self.tick('propagation_rounds');changed=False
            for c,mask in enumerate(domains):
                if not mask:return self.conflict(self.wipeout(c,reasons))
            # AC reasons contain every lost legal support, including the loss of
            # a support through a previous decision or another propagated rule.
            for a,neighbors in enumerate(self.adj):
                for b,sa,sb in neighbors:
                    for state in tuple(_bits(domains[a])):
                        self.tick('arc_state_checks')
                        supports=self.base[b]&self.colors[sb].get(self.faces[state][sa],0)&~(15<<(4*(state//4)))
                        if supports&domains[b]:continue
                        why=self.unavailable(b,supports,reasons)
                        changed|=self.remove(a,state,why,domains,reasons)
                        if not domains[a]:return self.conflict(self.wipeout(a,reasons))
            # If a cell can only use one physical piece, remove all four of that
            # piece's orientations elsewhere. Explain ALL non-piece alternatives.
            for a,mask in enumerate(domains):
                first=(mask&-mask).bit_length()-1;piece=first//4;piece_mask=15<<(4*piece)
                self.tick('piece_state_checks')
                if mask&~piece_mask:continue
                why=self.unavailable(a,self.base[a]&~piece_mask,reasons)
                for b in range(self.n):
                    if b==a:continue
                    self.tick('piece_state_checks')
                    for state in tuple(_bits(domains[b]&piece_mask)):
                        changed|=self.remove(b,state,why,domains,reasons)
                        if not domains[b]:return self.conflict(self.wipeout(b,reasons))
            # A physical piece must appear exactly once because there are n cells
            # and n pieces. Absence and unique-location deductions have complete
            # explanations over all initial locations of that physical piece.
            for piece in range(self.n):
                piece_mask=15<<(4*piece);locations=[];lost=[]
                for c,mask in enumerate(domains):
                    self.tick('piece_state_checks')
                    if mask&piece_mask:locations.append(c)
                    else:lost.extend(reasons[c][s] for s in _bits(self.base[c]&piece_mask))
                if not locations:return self.conflict(self.union(lost))
                if len(locations)==1:
                    c=locations[0];why=self.union(lost)
                    for state in tuple(_bits(domains[c]&~piece_mask)):
                        changed|=self.remove(c,state,why,domains,reasons)
                        if not domains[c]:return self.conflict(self.wipeout(c,reasons))
            if self.matching:
                matching_changed,failure=self.matching_filter(domains,reasons)
                if failure is not None:return self.conflict(failure)
                changed|=matching_changed
            if self.learning:
                for clause in tuple(self.nogoods):
                    open_literals=[];forced=[];satisfied=False
                    for c,state in clause:
                        self.tick('nogood_literal_checks')
                        if not domains[c]&(1<<state):satisfied=True;break
                        if domains[c]!=(1<<state):open_literals.append((c,state))
                        else:forced.append((c,state))
                    if satisfied or len(open_literals)>1:continue
                    why=self.union(self.force_reason(c,state,domains,reasons) for c,state in forced)
                    if not open_literals:
                        self.useful_clause(clause)
                        self.counters['nogood_rejections']+=1
                        return self.conflict(why)
                    c,state=open_literals[0]
                    if self.remove(c,state,why,domains,reasons):
                        self.useful_clause(clause)
                        changed=True;self.counters['nogood_unit_removals']+=1
                        if not domains[c]:return self.conflict(self.wipeout(c,reasons))
            if not changed:return None

    def copy(self,domains,reasons):
        self.tick('domain_copies',len(domains))
        copied_domains=domains.copy();copied_reasons=[]
        for item in reasons:
            self.tick('reason_copies',len(item));copied_reasons.append(item.copy())
        return copied_domains,copied_reasons

    def useful_clause(self, clause):
        pass

    def branch_options(self, domains, unresolved):
        cell = min(unresolved, key=lambda c: (domains[c].bit_count(), -len(self.adj[c]), self.rank[c]))
        return cell, tuple(_bits(domains[cell]))

    def visit(self,domains,reasons,depth):
        self.check();self.max_depth=max(self.max_depth,depth)
        failure=self.propagate(domains,reasons)
        if failure is not None:return None,failure
        unresolved=[c for c,d in enumerate(domains) if d&(d-1)]
        if not unresolved:
            board=[d.bit_length()-1 for d in domains]
            if not check_solution(self.problem,board):raise RuntimeError('Explained DFS returned invalid witness')
            return board,None
        c, choices = self.branch_options(domains, unresolved)
        combined=set(self.unavailable(c,self.base[c]&~domains[c],reasons))
        for state in choices:
            self.tick('decision_placements');self.branches+=1
            literal=(c,state);child_domains,child_reasons=self.copy(domains,reasons)
            for other in tuple(_bits(child_domains[c]&~(1<<state))):
                self.remove(c,other,frozenset((literal,)),child_domains,child_reasons)
            board,failure=self.visit(child_domains,child_reasons,depth+1)
            if board is not None:return board,None
            assert failure is not None
            if self.learning and literal not in failure:
                self.counters['backjumped_levels']+=1
                return None,failure
            self.tick('reason_literals',len(failure));combined.update(failure-{literal})
        failure=frozenset(combined);self.learn(failure)
        return None,failure

    def run(self):
        board=None;outcome='timeout';termination='timeout'
        try:
            domains,reasons=self.setup()
            board,failure=self.visit(domains,reasons,0)
            if board is not None:outcome=termination='solved'
            else:
                # Root exhaustion must resolve every temporary assumption.
                if failure:raise RuntimeError('Root failure still depends on a decision')
                outcome=termination='infeasible'
        except _Limit as limit:
            termination=limit.reason;outcome='stopped' if limit.reason=='stopped' else 'timeout'
        result=dict(self.report('finished'),outcome=outcome,termination=termination,board=board,
            workers=1,requested_workers=self.requested_workers,problem_hash=self.identity,
            proof_scope='supplied_problem_complete_search',coverage_verified=outcome in ('solved','infeasible'),
            resume_capability='fresh_search_no_checkpoint',work=self.work,work_limit=self.max_work,
            retained_nogoods=len(self.nogoods))
        if self.include_nogoods:result['learned_nogoods']=[[[c,s] for c,s in clause] for clause in self.nogoods]
        if self.on_progress is not None:self.on_progress(result)
        return result


def solve(problem,seconds=30.,seed=0,workers=1,should_stop=None,on_progress=None,*,
          learning=True,max_work=2000000,max_nogoods=256,max_nogood_literals=64,include_nogoods=False,
          matching=False):
    """Solve the exact supplied task, or return a bounded inconclusive outcome.

    `learning=False` disables both learned nogoods and backjumping while retaining
    the same propagators, reason accounting, branching order and work units.
    This is a controlled ablation, not a claim of parity with the optimized DFS.
    `matching=True` adds explained perfect-matching and all-different SCC filtering.
    Every graph visit and explanation scan is separately counted in work.
    """
    return _Search(problem,seconds,seed,workers,should_stop,on_progress,learning,
                   max_work,max_nogoods,max_nogood_literals,include_nogoods,matching).run()
