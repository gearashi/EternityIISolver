"""Impact ordering and bounded discrepancy passes over the established DFS.

History ranks alternatives only. Restricted passes cannot prove infeasibility;
the final unrestricted pass is complete. Every repeated placement is counted.
No checkpoint format or production DFS default is changed.
"""
from __future__ import annotations
import math
import time
from exact_dfs import _Search, _Interrupted, _bits


class _ImpactSearch(_Search):
    def __init__(self, problem, seconds, seed, should_stop, on_progress, hints,
                 options, ordering, discrepancy_limit, max_work):
        self.ordering = ordering
        self.discrepancy_limit = discrepancy_limit
        self.max_work = max_work
        self.limit_hit = False
        self.impact_work = 0
        self.observations = 0
        self.ranked_nodes = 0
        self.discrepancy_skips = 0
        self.passes = 0
        self.pass_budget = None
        self.history = {}
        super().__init__(problem, seconds, seed, should_stop, on_progress,
                         None, None, 5.0, hints, options)

    def check(self):
        if self.work_ticks + self.impact_work >= self.max_work:
            self.limit_hit = True
            raise _Interrupted('timeout')
        super().check()

    def charge(self, amount):
        # Reserve work before history/ordering calculations, including n-cell scans.
        if self.work_ticks + self.impact_work + amount > self.max_work:
            self.limit_hit = True
            raise _Interrupted('timeout')
        self.impact_work += amount
        self.check()

    def report(self):
        result = super().report()
        result.update(backend='impact-dfs-python', ordering=self.ordering,
            discrepancy_limit=self.discrepancy_limit, checkpoint_supported=False,
            work=self.work_ticks + self.impact_work,
            reasoning_counters_session=dict(work=self.work_ticks+self.impact_work, dfs_check_calls=self.work_ticks,
                impact_operations=self.impact_work, observed_branches=self.observations,
                ranked_nodes=self.ranked_nodes, discrepancy_skips=self.discrepancy_skips,
                passes_started=self.passes, active_discrepancy_budget=self.pass_budget),
            work_unit='DFS cancellation/check calls plus explicit impact-ordering operations')
        return result

    def prepare(self, domains, dirty, previous, depth):
        before_probes = self.probe_checks
        try:
            result = super().prepare(domains, dirty, previous, depth)
        finally:
            # Established optional singleton probing also makes real placements.
            self.branches += self.probe_checks-before_probes
            self.cell_branches += self.probe_checks-before_probes
        if self.ordering == 'impact' and len(self.stack) > 1:
            parent = self.stack[-2]
            if parent['kind'] == 'cell':
                self.charge(2 * self.n + 1)
                choice = parent['candidates'][parent['next'] - 1]
                key = (parent['cell'], choice)
                # A heuristic estimate of domain-volume contraction, not a
                # solution count or a reason for pruning a state.
                if result is None:
                    impact = 1.0
                else:
                    before = sum(math.log2(d.bit_count()) for d in parent['domains'])
                    after = sum(math.log2(d.bit_count()) for d in domains)
                    impact = max(0.0, min(1.0, -math.expm1((after-before)*math.log(2))))
                count, total = self.history.get(key, (0, 0.0))
                self.history[key] = (count + 1, total + impact)
                self.observations += 1
        return result

    def branch(self, domains):
        if self.ordering == 'baseline' or not self.history:
            return super().branch(domains)
        self.charge(self.n)
        undecided = [c for c, d in enumerate(domains) if d & (d-1)]
        estimates = {}
        for cell in undecided:
            self.charge(domains[cell].bit_count())
            # Unobserved choices use the cell's observed mean, or the neutral
            # contraction from fixing one of its currently available values.
            seen = [self.history[(cell,s)] for s in _bits(domains[cell]) if (cell,s) in self.history]
            mean = sum(total for count,total in seen)/sum(count for count,total in seen) if seen else 1-1/domains[cell].bit_count()
            estimates[cell] = mean
        def cell_order(cell):
            degree = sum(bool(domains[o] & (domains[o]-1)) for o,_,_ in self.neighbors[cell])
            return domains[cell].bit_count(), -estimates[cell], -degree, self.tie_order[cell]
        cell = min(undecided, key=cell_order)
        candidates = list(_bits(domains[cell])); self.rng.shuffle(candidates)
        def value_order(state):
            self.charge(len(self.neighbors[cell])+1)
            count,total = self.history.get((cell,state),(0,0.0))
            impact = total/count if count else estimates[cell]
            other_pieces = ~(15 << (4*(state//4)))
            support = sum((domains[o] & self.colors[os].get(self.faces[state][side],0) & other_pieces).bit_count()
                          for o,side,os in self.neighbors[cell])
            return impact, -support, -self.hint_votes[cell].get(state,0)
        candidates.sort(key=value_order)
        self.ranked_nodes += 1
        return cell,candidates

    def choose_partition(self, domains):
        kind,target,candidates = super().choose_partition(domains)
        if self.pass_budget is not None:
            self.charge(len(self.stack))
            spent = sum(f['next']-1 for f in self.stack[:-1] if f['phase']=='branch')
            remaining = self.pass_budget-spent
            assert remaining >= 0
            kept = candidates[:remaining+1]
            self.discrepancy_skips += len(candidates)-len(kept)
            candidates = kept
        return kind,target,candidates


def solve(problem, seconds=10.0, seed=0, workers=1, should_stop=None,
          on_progress=None, hints=None, options=None, ordering='impact',
          discrepancy_limit=2, max_work=2000000):
    """Fresh bounded exact search; optional restricted passes precede full DFS.

    ``branches`` includes every placement in repeated prefixes. Impact is
    learned from those placements; no unreported speculative probes run.
    Discrepancy is the sum of zero-based alternative ranks along a path.
    Ordering estimates cannot prune. A timeout/stop never proves infeasibility.
    The work cap covers DFS check calls and ordering operations; optional rule
    and component counters remain separate in the inherited report.
    """
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
        raise ValueError('seconds must be finite and nonnegative')
    if type(seed) is not int or not -(2**63)<=seed<2**64:raise ValueError('Invalid seed')
    if type(workers) is not int or workers!=1:raise ValueError('Exactly one worker is supported')
    if ordering not in ('baseline','impact'):raise ValueError('Invalid ordering')
    if discrepancy_limit is not None and (type(discrepancy_limit) is not int or not 0<=discrepancy_limit<=8):
        raise ValueError('discrepancy_limit must be None or 0..8')
    if type(max_work) is not int or not 0<=max_work<=100000000:raise ValueError('Invalid work cap')
    if any(callback is not None and not callable(callback) for callback in (should_stop,on_progress)):
        raise ValueError('Callbacks must be callable')
    search = _ImpactSearch(problem,seconds,seed,should_stop,on_progress,hints,options,
                           ordering,discrepancy_limit,max_work)
    board=None; termination=None; outcome='timeout'
    try:
        domains=search.build()
        budgets=[] if discrepancy_limit is None else list(range(discrepancy_limit+1))
        for budget in budgets+[None]:
            search.check(); search.pass_budget=budget; search.passes+=1
            board=search.visit(domains.copy())
            if board is not None:
                outcome='solved'; break
            if budget is None:
                outcome='infeasible'
        termination=outcome
    except _Interrupted as exc:
        outcome=exc.outcome
        termination='work_limit' if search.limit_hit else outcome
    result=dict(search.report(),outcome=outcome,termination=termination,board=board,
                workers=1,language='python',resumed=False,checkpoint=None,
                proof_scope='entire_supplied_problem',coverage_verified=outcome=='infeasible',
                solution_verified=board is not None)
    if result['build_seconds'] is None:result['build_seconds']=result['elapsed_seconds']
    if on_progress is not None:on_progress(result.copy())
    return result
