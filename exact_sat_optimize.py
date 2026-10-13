"""Incremental full-board SAT score targets; offline and independently checked.

One structural model and one totalizer remain alive while temporary assumptions
reduce the allowed mismatches. UNSAT applies only to the supplied domains,
oriented clues and queried score. Unknown never creates an infeasibility claim.
"""
from __future__ import annotations

import time

from exact_model import ExactProblem, constraint_edges, grid_edges
from exact_optimize import _inputs, evaluate_board
from exact_sat import AssumptionSession


class ScoreSession(AssumptionSession):
    """Reusable SAT threshold oracle for min_score..maximum, with one deadline.

    solve(target_score) returns a structurally valid board whose independently
    counted score meets the target, or infeasible/timeout/stopped. Queries may
    tighten or loosen within the initially encoded range. All setup time counts.
    """
    def __init__(self, problem, seconds=30.0, seed=0, workers=1, should_stop=None,
                 on_progress=None, *, min_score=0, phase_board=None):
        if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
            raise ValueError('SAT score optimization requires all grid edges')
        maximum = len(grid_edges(problem.size))
        if type(min_score) is not int or not 0 <= min_score <= maximum:
            raise ValueError('min_score must be an integer in 0..maximum')
        structural = ExactProblem(problem.size, tuple(tuple(f) for f in problem.faces),
                                  dict(problem.fixed), tuple(tuple(d) for d in problem.domains), ())
        super().__init__(structural, seconds, seed, workers, should_stop, on_progress)
        self.full_problem = ExactProblem(self.problem.size, self.problem.faces,
                                         dict(self.problem.fixed), self.problem.domains)
        self.phase_board = None if phase_board is None else tuple(phase_board)
        if self.phase_board is not None and not evaluate_board(self.full_problem, self.phase_board)['valid']:
            raise ValueError('phase_board must be a structurally legal reference board')
        self.min_score, self.maximum = min_score, maximum
        self.target_score = min_score
        self.matches, self.thresholds = [], []

    def _build(self):
        self.matches, self.thresholds = [], []
        super()._build()
        self._built = False  # A cancelled score encoding must discard the CNF.
        faces = self.problem.faces
        for a, b, sa, sb in grid_edges(self.problem.size):
            self.check()
            channels = []
            for cell, side in ((a, sa), (b, sb)):
                support = {}
                for state, literal in self.cell_vars[cell]:
                    support.setdefault(faces[state][side], []).append(literal)
                endpoint = {}
                for color, literals in sorted(support.items()):
                    channel = self._new_var()
                    endpoint[color] = channel
                    for literal in literals:
                        self._add([-literal, channel])
                    self._add([-channel, *literals])
                # Cell exactly-one plus iff supports already imply exactly one
                # endpoint color, so an extra quadratic color AMO is redundant.
                channels.append(endpoint)
            left, right = channels
            match = self._new_var()
            self.matches.append(match)
            for color, literal in left.items():
                other = right.get(color)
                self._add([-match, -literal, other] if other is not None else [-match, -literal])
                if other is not None:
                    self._add([-literal, -other, match])
            for color, literal in right.items():
                other = left.get(color)
                self._add([-match, -literal, other] if other is not None else [-match, -literal])
        self.check()
        if self.matches:
            from pysat.card import ITotalizer
            self.check()
            limit = min(self.maximum - self.min_score, self.maximum - 1)
            # Native generation is bounded by <=480 literals and <=480 outputs.
            # Check immediately afterwards; it cannot create a second budget.
            with ITotalizer(lits=[-v for v in self.matches], ubound=limit,
                            top_id=self.next_var) as totalizer:
                self.check()
                self.next_var = max(self.next_var, totalizer.top_id)
                for clause in totalizer.cnf.clauses:
                    self._add(clause)
                self.thresholds = list(totalizer.rhs)
        self.check()
        if self.phase_board is not None:
            self.native.set_phases([literal if state == self.phase_board[cell] else -literal
                                    for cell, values in enumerate(self.cell_vars) for state, literal in values])
            self.check()
        self._built = True

    def _search_literals(self, selectors):
        allowed = self.maximum - self.target_score
        return [] if allowed == self.maximum else [-self.thresholds[allowed]]

    def _validate_candidate(self, candidate, assumptions, selected):
        checked = evaluate_board(self.full_problem, candidate)
        return (checked['valid'] and checked['score'] >= self.target_score and
                sum(literal in selected for literal in self.matches) == checked['score'])

    def solve(self, target_score, *, seconds=None):
        if type(target_score) is not int or not self.min_score <= target_score <= self.maximum:
            raise ValueError('target_score outside encoded score range')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('SAT session query is already active')
        try:
            # Do not overwrite the active threshold during callback reentry.
            with self._lifecycle_lock:
                if self.closed or self._closing.is_set():
                    raise RuntimeError('SAT session is closed')
                if self._active_query:
                    raise RuntimeError('SAT session query is already active')
                self.target_score = target_score
            answer = super().solve(None, seconds=seconds)
            answer.update(backend='pysat-score', target_score=target_score,
                          max_score=self.maximum,
                          score=None if answer['board'] is None else
                          evaluate_board(self.full_problem, answer['board'])['score'],
                          proof_scope='supplied-domains-fixed-placements-and-full-grid-score-target')
            return answer
        finally:
            self._lock.release()


def optimize(problem, reference, seconds=30.0, seed=0, workers=1, should_stop=None,
             on_improvement=None, on_progress=None, *, target_score=None, reference_phases=True):
    """Seek a strict score improvement, then tighten to best+1 without rebuilding.

    Every return retains the validated reference or an independently validated
    improvement. One Glucose worker is used even if more were requested.
    """
    started = time.monotonic()
    problem = ExactProblem(problem.size, tuple(tuple(f) for f in problem.faces),
                           dict(problem.fixed), tuple(tuple(d) for d in problem.domains),
                           None if problem.edges is None else tuple(tuple(e) for e in problem.edges))
    reference = list(reference)
    baseline, maximum = _inputs(problem, reference, seconds, seed, workers,
                                should_stop, on_improvement, on_progress)
    if type(reference_phases) is not bool:
        raise ValueError('reference_phases must be boolean')
    if target_score is None:
        target_score = min(maximum, baseline + 1)
    if type(target_score) is not int or not min(maximum, baseline + 1) <= target_score <= maximum:
        raise ValueError('target_score must be a strict improvement up to max_score')
    best, score = reference, baseline
    branches = conflicts = queries = model_builds = 0
    build_seconds = 0.0
    deadline = started + seconds
    current_target = target_score

    def result(outcome, termination):
        return {'outcome': outcome, 'termination': termination, 'board': list(best),
                'score': score, 'reference_score': baseline, 'target_score': current_target,
                'max_score': maximum, 'improved': score > baseline,
                'backend': 'sat-optimize', 'workers': 1, 'requested_workers': workers,
                'branches': branches, 'conflicts': conflicts, 'model_builds': model_builds,
                'target_queries': queries, 'build_seconds': build_seconds,
                'elapsed_seconds': time.monotonic() - started, 'incremental_sat': True,
                'reference_phases': reference_phases,
                'proof_scope': 'supplied-domains-fixed-placements-and-full-grid-score-target'}

    if should_stop is not None and should_stop():
        return result('stopped', 'stopped')
    if time.monotonic() >= deadline:
        return result('timeout', 'timeout')
    if baseline == maximum:
        return result('solved', 'reference_already_perfect')

    def progress(update):
        if on_progress is not None:
            on_progress({**update, 'backend': 'sat-optimize', 'score': score,
                         'target_score': current_target, 'target_queries': queries,
                         'elapsed_seconds': time.monotonic() - started})

    with ScoreSession(problem, seconds=max(0., deadline - time.monotonic()), seed=seed,
                      workers=workers, should_stop=should_stop, on_progress=progress,
                      min_score=target_score, phase_board=reference if reference_phases else None) as session:
        # Validate against the detached snapshot used by the encoded model.
        puzzle = session.full_problem
        while True:
            queries += 1
            answer = session.solve(current_target, seconds=max(0., deadline - time.monotonic()))
            branches += answer['branches']
            conflicts += answer['conflicts']
            build_seconds += answer['build_seconds']
            model_builds = answer['model_builds']
            if answer['outcome'] == 'solved':
                checked = evaluate_board(puzzle, answer['board'])
                if not checked['valid'] or checked['score'] < current_target:
                    raise RuntimeError('SAT score witness failed independent target validation')
                best, score = list(answer['board']), checked['score']
                if on_improvement is not None:
                    on_improvement({'board': list(best), 'score': score, 'backend': 'sat-optimize',
                                    'branches': branches, 'conflicts': conflicts,
                                    'elapsed_seconds': time.monotonic() - started})
                if score == maximum:
                    return result('solved', 'perfect_board_validated')
                current_target = score + 1
            elif answer['outcome'] == 'infeasible':
                return result('improved' if score > baseline else 'infeasible_for_target',
                              'optimal_for_supplied_problem' if score > baseline else 'infeasible_for_target')
            elif answer['outcome'] == 'stopped':
                return result('stopped', 'stopped')
            elif answer['outcome'] == 'timeout':
                return result('improved' if score > baseline else 'timeout', 'timeout')
            else:
                raise RuntimeError('Unexpected SAT threshold outcome')
