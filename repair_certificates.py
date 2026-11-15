"""Bounded, native-proved score failures reused across frozen-board contexts.

A certificate means: under the immutable puzzle rules and an absolute minimum
score, the listed frozen placements have no completion. Omitting other freezes
is justified only by an actual UNSAT core (or a completed deletion query).
Certificates are memory-only; there is no proof-import or serialization path.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import replace
import hashlib
import itertools
import json
import math
import threading
import time
from types import MappingProxyType

from exact_model import constraint_edges, grid_edges, problem_hash
from exact_optimize import evaluate_board
from exact_sat import AssumptionSession, _Cancelled
from exact_sat_optimize import ScoreSession

SCOPE = 'immutable-full-puzzle-score-target-and-certified-frozen-placement-subset'


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value


def _seconds(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
        raise ValueError('seconds must be finite and nonnegative')
    return float(value)


class FixedTargetScoreSession(ScoreSession):
    """ScoreSession with one hard threshold and temporary placement selectors.

    Hardwiring the threshold prevents assumption-core extraction or deletion
    minimization from accidentally dropping the score condition. A new target
    needs another native model; the controller keeps at most one alive.
    Changing the seed also replaces the session to preserve phase semantics.
    """
    def __init__(self, problem, target_score, *, seconds=3600., seed=0,
                 should_stop=None, conflict_limit=1000, propagation_limit=50000,
                 max_variables=500000, max_clauses=1500000):
        self.conflict_limit = _integer(conflict_limit,'conflict_limit',1,1000000)
        self.propagation_limit = _integer(propagation_limit,'propagation_limit',1,10000000)
        self.max_variables = _integer(max_variables,'max_variables',1,1000000)
        self.max_clauses = _integer(max_clauses,'max_clauses',1,3000000)
        self.model_limit_reached = False
        self.encoding_clauses = 0
        super().__init__(problem,seconds=seconds,seed=seed,should_stop=should_stop,min_score=target_score)
        self.target_score = target_score
        self._fixed_target = target_score

    def _new_var(self):
        if self.next_var >= self.max_variables:
            self.model_limit_reached = True
            raise _Cancelled('timeout')
        return super()._new_var()

    def _add(self, clause):
        if self.next_var > self.max_variables or self.clause_count >= self.max_clauses:
            self.model_limit_reached = True
            raise _Cancelled('timeout')
        answer = super()._add(clause)
        self.encoding_clauses += 1
        return answer

    def _build(self):
        super()._build()
        self._built = False
        allowed = self.maximum - self.target_score
        if allowed < self.maximum:
            self._add([-self.thresholds[allowed]])
        self.check()
        self._built = True

    def _search_literals(self, selectors):
        self.native.conf_budget(self.conflict_limit)
        self.native.prop_budget(self.propagation_limit)
        return list(selectors)

    def _validate_candidate(self, candidate, assumptions, selected):
        return (super()._validate_candidate(candidate,assumptions,selected)
                and all(candidate[c] == state for c,state in (assumptions or {}).items()))

    def solve_frozen(self, assumptions, *, seconds=None):
        if self.target_score != self._fixed_target:
            raise ValueError("A fixed score session cannot change its target")
        encoded_before = self.encoding_clauses
        before = {} if self.native is None else self.native.accum_stats() or {}
        result = AssumptionSession.solve(self,assumptions,seconds=seconds)
        after = {} if self.native is None else self.native.accum_stats() or {}
        result.update(target_score=self.target_score, max_score=self.maximum,
                      score=None if result['board'] is None else evaluate_board(self.full_problem,result['board'])['score'],
                      propagations=max(0,int(after.get('propagations',0))-int(before.get('propagations',0))),
                      model_limit_reached=self.model_limit_reached,
                      encoding_clauses=self.encoding_clauses-encoded_before)
        return result

    def solve(self, *args, **kwargs):
        raise TypeError('Use solve_frozen; this session has one immutable score target')


class GeneralizedFailureCache:
    """One native session plus bounded certificates for an immutable problem.

    `generalize=False` is the exact-context ablation. In generalized mode a
    certificate can be used for a stricter target and any context containing all
    its frozen placements. Other cells may change or be released safely.
    """
    def __init__(self, problem, capacity=256, *, generalize=True,
                 conflict_limit=1000, propagation_limit=50000,
                 max_variables=500000, max_clauses=1500000):
        if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
            raise ValueError('Generalized score certificates require the full grid')
        if type(generalize) is not bool:
            raise ValueError('generalize must be boolean')
        self._problem = replace(problem, faces=tuple(tuple(f) for f in problem.faces),
                                domains=tuple(tuple(d) for d in problem.domains),
                                fixed=MappingProxyType(dict(problem.fixed)),edges=None)
        self._fingerprint = problem_hash(self._problem)
        self._capacity = _integer(capacity,'capacity',1,4096)
        self._limits = dict(conflict_limit=_integer(conflict_limit,'conflict_limit',1,1000000),
                            propagation_limit=_integer(propagation_limit,'propagation_limit',1,10000000),
                            max_variables=_integer(max_variables,'max_variables',1,1000000),
                            max_clauses=_integer(max_clauses,'max_clauses',1,3000000))
        self.generalize = generalize
        self._proofs = OrderedDict()
        self._session = None
        self._session_key = None
        self._stop = None
        self._lock = threading.Lock()
        self.closed = False
        self.cache_hits = 0

    @property
    def cache_size(self): return len(self._proofs)

    @property
    def problem(self): return self._problem

    def __enter__(self):
        if self.closed: raise RuntimeError('Certificate cache is closed')
        return self

    def __exit__(self,*_): self.close()

    def close(self):
        self.closed = True
        if self._session is not None:
            self._session.close()

    def certificates(self):
        """Detached diagnostic descriptions, not accepted as external proofs."""
        return [dict(problem=self._fingerprint,target_score=target,placements=dict(core),certificate_id=key)
                for key,(target,core) in self._proofs.items()]

    def _save(self,target,core):
        core = tuple(sorted(core.items()))
        key = hashlib.sha256(json.dumps([self._fingerprint,target,core],separators=(',',':')).encode()).hexdigest()
        self._proofs[key]=(target,core);self._proofs.move_to_end(key)
        while len(self._proofs)>self._capacity:self._proofs.popitem(last=False)
        return key

    def solve(self, board, cells, *, target_score=None, seconds=.25, seed=0,
              should_stop=None, minimize_checks=0):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Certificate query is already active')
        try:
            if self.closed: raise RuntimeError('Certificate cache is closed')
            return self._solve(board,cells,target_score,seconds,seed,should_stop,minimize_checks)
        finally:
            self._stop=None
            self._lock.release()

    def _solve(self,board,cells,target_score,seconds,seed,should_stop,minimize_checks):
        started=time.monotonic();seconds=_seconds(seconds);deadline=started+seconds
        _integer(seed,'seed',0,2**32-1);_integer(minimize_checks,'minimize_checks',0,64)
        if should_stop is not None and not callable(should_stop):raise ValueError('should_stop must be callable')
        reference=list(board);checked=evaluate_board(self._problem,reference)
        if not checked['valid']:raise ValueError('Certificate reference violates pieces, frame, domains or clues')
        chosen=list(itertools.islice(iter(cells),65))
        if (len(chosen)>64 or any(type(c) is not int or not 0<=c<len(reference) for c in chosen)
                or len(set(chosen))!=len(chosen) or set(chosen)&self._problem.fixed.keys()):
            raise ValueError('Release at most64 distinct mutable cells')
        chosen.sort();selected=set(chosen);maximum=checked['max_score']
        target=min(maximum,checked['score']+1) if target_score is None else target_score
        _integer(target,'target_score',0,maximum)
        assumptions={c:state for c,state in enumerate(reference) if c not in selected and c not in self._problem.fixed}
        self._stop=should_stop
        totals=dict(branches=0,conflicts=0,propagations=0,native_queries=0,build_seconds=0.,
                    model_builds=0,lookup_literal_checks=0,minimization_queries=0,encoding_clauses=0)
        best=reference.copy();best_score=checked['score'];core=None;proof_id=None;cache_hit=False;native_status='NOT_RUN'
        model_limit=False;certificate_target=None

        def stopped(): return self.closed or bool(should_stop is not None and should_stop())
        def remaining(): return max(0.,deadline-time.monotonic())
        def failure_termination():
            # Failure at a higher target can coexist with a lower improvement.
            return ("no_improvement_in_region" if certificate_target <= checked["score"]+1
                    else "target_unreachable")
        def result(termination):
            return dict(board=best.copy(),score=best_score,reference_score=checked['score'],
                        target_score=target,max_score=maximum,improved=best_score>checked['score'],
                        outcome='solved' if best_score==maximum else 'improved' if best_score>checked['score'] else termination,
                        termination=termination,operator='generalized-region-certificate',backend='pysat-frozen-score',workers=1,
                        region_cells=chosen.copy(),neighborhood_size=len(chosen),proof_scope=SCOPE,
                        proven_unreachable=termination in ('no_improvement_in_region','target_unreachable'),
                        optimal=termination in ('no_improvement_in_region','perfect_board_validated'),
                        cache_hit=cache_hit,proof_reused=cache_hit,certificate_id=proof_id,
                        certificate_target=certificate_target,
                        core_placements=None if core is None else dict(core),
                        core_is_minimum=False,solver_status=native_status,model_limit_reached=model_limit,
                        elapsed_seconds=time.monotonic()-started,**totals)

        if stopped(): return result('stopped')
        if not remaining():return result('timeout')
        if best_score>=target:return result('perfect_board_validated' if best_score==maximum else 'target_already_reached')
        for key,(proved_target,conditions) in list(self._proofs.items()):
            if stopped():return result('stopped')
            if not remaining():return result('timeout')
            if target<proved_target or (not self.generalize and target!=proved_target):continue
            if not self.generalize and len(conditions)!=len(assumptions):continue
            matched=True
            for c,state in conditions:
                totals['lookup_literal_checks']+=1
                if assumptions.get(c)!=state:matched=False;break
            if matched:
                self._proofs.move_to_end(key);self.cache_hits+=1;cache_hit=True
                core=dict(conditions);proof_id=key;native_status='INFEASIBLE';certificate_target=proved_target
                return result(failure_termination())
        key=(target,seed)
        if (self._session is None or self._session_key!=key or self._session.closed
                or self._session.budget.reason() is not None):
            if self._session is not None:self._session.close()
            self._session=FixedTargetScoreSession(self._problem,target,seconds=3600.,seed=seed,
                              should_stop=lambda:self.closed or bool(self._stop is not None and self._stop()),**self._limits)
            self._session_key=key
        session=self._session

        def query(frozen,minimizing=False):
            before_builds=session.model_builds
            answer=session.solve_frozen(frozen,seconds=remaining())
            for k in ('branches','conflicts','propagations','build_seconds','encoding_clauses'):totals[k]+=answer[k]
            totals['native_queries']+=1;totals['model_builds']+=session.model_builds-before_builds
            totals['minimization_queries']+=int(minimizing)
            return answer

        answer=query(assumptions);model_limit=answer['model_limit_reached'];native_status=answer['outcome'].upper()
        if answer['outcome']=='solved':
            candidate=list(answer['board']);actual=evaluate_board(self._problem,candidate)
            if (not actual['valid'] or actual['score']<target or actual['score']!=answer['score']
                    or any(candidate[c]!=state for c,state in assumptions.items())
                    or {candidate[c]//4 for c in chosen}!={reference[c]//4 for c in chosen}):
                raise RuntimeError('Frozen score witness escaped its validated region')
            if actual['score']>best_score:best,best_score=candidate,actual['score']
            return result('perfect_board_validated' if best_score==maximum else 'target_reached')
        if answer['outcome']!='infeasible':
            return result('stopped' if answer['outcome']=='stopped' else 'model_limit' if model_limit else 'timeout')
        raw=answer['core_placements']
        if type(raw) is not dict or any(assumptions.get(c)!=state for c,state in raw.items()):
            raise RuntimeError('Native score failure returned an invalid assumption core')
        core=dict(raw) if self.generalize else assumptions.copy()
        # Each deletion is an actual query under the same HARD score threshold.
        # An unknown result cannot delete an assumption or create a new proof.
        if self.generalize:
            for cell in list(sorted(core))[:minimize_checks]:
                if stopped() or not remaining():break
                if cell not in core:continue
                trial={c:s for c,s in core.items() if c!=cell}
                minimized=query(trial,True)
                if minimized['outcome']=='infeasible':
                    new=minimized['core_placements']
                    if type(new) is not dict or any(trial.get(c)!=state for c,state in new.items()):
                        raise RuntimeError('Invalid minimization core')
                    core=dict(new)
                elif minimized['outcome'] in ('timeout','stopped'):break
        # The initial complete proof remains sound even when optional minimization
        # is interrupted; no interrupted query contributes a deletion.
        proof_id=self._save(target,core);certificate_target=target
        return result(failure_termination())
