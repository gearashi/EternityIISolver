"""Optional bounded repair operators; scheduling and true incumbent stay outside.

Counters keep complete-candidate tests, path ranking and native CP-SAT decisions
separate. A changed placement includes a rotation. This module never enables an
operator automatically and never treats a restricted failure as global proof.
"""
from __future__ import annotations
from dataclasses import replace
import itertools
import math

from exact_model import constraint_edges, grid_edges, problem_hash
from exact_optimize import evaluate_board
from repair_guidance import Guidance
from repair_multisource import multi_parent_recombine


class QualityRepairController:
    SCHEMA = 'eternity-quality-repair/v1'
    EJECTION_CANDIDATES = 128
    EJECTION_PATHS = 512
    RECOMBINATION_WORK = .05
    MAX_COUNTER = 2**53 - 1
    INTEGER_COUNTERS = ('ejection_calls', 'recombination_calls', 'candidate_evaluations',
                        'heuristic_state_evaluations', 'path_expansions', 'native_branches',
                        'native_conflicts', 'table_rows', 'returned_piece_changes',
                        'returned_board_improvements', 'true_score_gain', 'stopped_calls',
                        'adoptions', 'accepted_piece_changes', 'accepted_board_improvements')

    LIFT_COUNTERS = ('grid_edges','domain_states','orientation_tests','color_tests','rows_before','rows_after')

    def __init__(self, problem, *, lift_rotations=False):
        if type(lift_rotations) is not bool:
            raise ValueError('lift_rotations must be boolean')
        self.lift_rotations = lift_rotations
        self._lift_work = dict.fromkeys(self.LIFT_COUNTERS, 0)
        if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
            raise ValueError('Quality repair requires the full grid')
        self.problem = replace(problem, fixed=dict(problem.fixed),
                               domains=tuple(tuple(d) for d in problem.domains),
                               faces=tuple(tuple(f) for f in problem.faces), edges=None)
        self.fingerprint = problem_hash(self.problem)
        self.guidance = Guidance(self.problem)
        self._counters = dict.fromkeys(self.INTEGER_COUNTERS, 0)
        self._counters['deterministic_work'] = 0.0

    def _board(self, board):
        checked = evaluate_board(self.problem, board)
        if not checked['valid']:
            raise ValueError('Quality repair board violates pieces, domains, frame or fixed oriented clues')
        return list(board), checked['score']

    @staticmethod
    def _controls(seed, should_stop):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('seed must be a uint32 integer')
        if should_stop is not None and not callable(should_stop):
            raise ValueError('should_stop must be callable')

    def _count(self, value, name, maximum=None):
        if type(value) is not int or value < 0 or value > (self.MAX_COUNTER if maximum is None else maximum):
            raise RuntimeError(f'Invalid operator {name} counter')
        return value

    def _commit(self, increments):
        pending = self._counters.copy()
        for key, value in increments.items():
            pending[key] += value
        # Validate before replacing state so callback/validation failures cannot
        # leave a half-accounted invocation in a resumable checkpoint.
        self._validate_counters(pending)
        self._counters = pending

    def _validate_counters(self, counters):
        if type(counters) is not dict or set(counters) != {*self.INTEGER_COUNTERS, 'deterministic_work'}:
            raise ValueError('Invalid quality counter fields')
        if any(type(counters[k]) is not int or not 0 <= counters[k] <= self.MAX_COUNTER for k in self.INTEGER_COUNTERS):
            raise ValueError('Invalid quality integer counters')
        native_work = counters['deterministic_work']
        if (type(native_work) not in (int, float) or not math.isfinite(native_work)
                or not 0 <= native_work <= self.MAX_COUNTER):
            raise ValueError('Invalid deterministic-work counter')
        ejections, recombinations = counters['ejection_calls'], counters['recombination_calls']
        calls = ejections + recombinations
        n = self.problem.size**2
        if (calls > self.MAX_COUNTER or counters['returned_board_improvements'] > calls
                or counters['stopped_calls'] > calls
                or counters['candidate_evaluations'] > ejections * self.EJECTION_CANDIDATES
                or counters['path_expansions'] > ejections * self.EJECTION_PATHS
                or counters['heuristic_state_evaluations'] > ejections * 4 * n * n
                or counters['table_rows'] > recombinations * 64 * 8
                or counters['returned_piece_changes'] > counters['returned_board_improvements'] * n
                or counters['true_score_gain'] > counters['returned_board_improvements'] * len(grid_edges(self.problem.size))
                or counters['accepted_piece_changes'] > counters['adoptions'] * n
                or counters['accepted_board_improvements'] > counters['adoptions']
                or (recombinations == 0 and (counters['native_branches'] or counters['native_conflicts'] or native_work))):
            raise ValueError('Inconsistent quality counters')

    def _acceptance_counts(self, reference, reference_score, board, score, termination):
        gain = score - reference_score
        if gain < 0:
            raise RuntimeError('Quality repair attempted to lower its true-score reference')
        return dict(returned_board_improvements=int(gain > 0), true_score_gain=gain,
                    returned_piece_changes=sum(a != b for a, b in zip(reference, board)) if gain > 0 else 0,
                    stopped_calls=int(termination == 'stopped'))

    def eject(self, board, *, seed, should_stop=None):
        self._controls(seed, should_stop)
        reference, reference_score = self._board(board)
        answer = self.guidance.ejection_repair(reference, seed=seed,
                    max_evaluations=self.EJECTION_CANDIDATES, max_expansions=self.EJECTION_PATHS,
                    should_stop=should_stop)
        candidate, actual_score = self._board(answer['best_true_board'])
        if actual_score != answer['best_true_score'] or reference_score != answer['reference_score']:
            raise RuntimeError('Ejection reported a false true score')
        increments = dict(ejection_calls=1,
                          candidate_evaluations=self._count(answer['candidate_evaluations'], 'candidates', self.EJECTION_CANDIDATES),
                          path_expansions=self._count(answer['path_expansions'], 'paths', self.EJECTION_PATHS),
                          heuristic_state_evaluations=self._count(answer['heuristic_state_evaluations'], 'probes', 4 * self.problem.size**4),
                          **self._acceptance_counts(reference, reference_score, candidate, actual_score, answer['termination']))
        self._commit(increments)
        return dict(answer, board=candidate.copy(), score=actual_score, best_true_board=candidate.copy(),
                    best_true_score=actual_score, reference_score=reference_score,
                    changed=actual_score > reference_score, delta=actual_score-reference_score,
                    quality_stats=self.stats())

    def recombine(self, parents, *, seed, seconds=.25, should_stop=None):
        self._controls(seed, should_stop)
        try:
            parents = list(itertools.islice(parents, 9))
        except TypeError as exc:
            raise ValueError('parents must contain 3..8 complete boards') from exc
        if not 3 <= len(parents) <= 8:
            raise ValueError('parents must contain 3..8 complete boards')
        checked = [self._board(board) for board in parents]
        parents = [board for board, _ in checked]
        reference_index = max(range(len(parents)), key=lambda i: checked[i][1])
        reference, reference_score = checked[reference_index]
        answer = multi_parent_recombine(self.problem, parents, seconds=seconds,
                    deterministic_budget=self.RECOMBINATION_WORK, seed=seed, should_stop=should_stop,
                    **({'lift_rotations': True} if self.lift_rotations else {}))
        candidate, actual_score = self._board(answer['board'])
        if actual_score != answer['score'] or reference_score != answer['reference_score']:
            raise RuntimeError('Recombination reported a false true score')
        cells = answer['region_cells']
        if (type(cells) is not list or len(cells) > 64
                or any(type(c) is not int or not 0 <= c < len(reference) for c in cells)
                or len(set(cells)) != len(cells) or set(cells) & self.problem.fixed.keys()):
            raise RuntimeError('Recombination reported an invalid region')
        selected = set(cells)
        def supported(c):
            observed = {parent[c] for parent in parents}
            if candidate[c] in observed:
                return True
            if not self.lift_rotations:
                return False
            sides = [side for a,b,sa,sb in grid_edges(self.problem.size)
                     for cell,other,side in ((a,b,sa),(b,a,sb)) if cell == c and other in selected]
            return any(s//4 == candidate[c]//4 and all(
                self.problem.faces[s][d] == self.problem.faces[candidate[c]][d] for d in sides) for s in observed)
        if (any(candidate[c] != reference[c] for c in range(len(reference)) if c not in selected)
                or {candidate[c] // 4 for c in cells} != {reference[c] // 4 for c in cells}
                or any(not supported(c) for c in cells)):
            raise RuntimeError('Recombination witness escaped its observed states or frozen complement')
        native_work = answer['deterministic_work']
        if type(native_work) not in (int, float) or not math.isfinite(native_work) or native_work < 0:
            raise RuntimeError('Invalid recombination deterministic work')
        increments = dict(recombination_calls=1,
                          native_branches=self._count(answer['branches'], 'branches'),
                          native_conflicts=self._count(answer['conflicts'], 'conflicts'),
                          table_rows=self._count(answer['table_rows'], 'table rows', 64 * 8),
                          deterministic_work=float(native_work),
                          **self._acceptance_counts(reference, reference_score, candidate, actual_score, answer['termination']))
        lift = answer.get('lift_work')
        pending = self._lift_work.copy()
        if self.lift_rotations and lift is not None:
            if not isinstance(lift,dict) or set(lift) != set(self.LIFT_COUNTERS):
                raise RuntimeError('Invalid rotation-lift work fields')
            for key in self.LIFT_COUNTERS:
                inc = self._count(lift[key],key)
                pending[key] = self._count(pending[key]+inc,key)
        self._commit(increments)
        self._lift_work = pending
        return dict(answer, board=candidate.copy(), score=actual_score, best_true_board=candidate.copy(),
                    best_true_score=actual_score, changed=actual_score > reference_score,
                    delta=actual_score-reference_score, quality_stats=self.stats())

    def record_adoption(self, before, after):
        """Count a portfolio replacement only after the caller actually adopts it.

        Returned proposals can be rejected as tabu. A proposal may also replace
        a different population parent, so changed cells are measured against
        this explicit before-board, never against the operator reference.
        true_score_gain remains the separately counted returned-proposal gain.
        """
        previous, previous_score = self._board(before)
        current, current_score = self._board(after)
        changed = sum(a != b for a, b in zip(previous, current))
        if changed:
            self._commit(dict(adoptions=1, accepted_piece_changes=changed,
                              accepted_board_improvements=int(current_score > previous_score)))
        return changed

    def stats(self):
        result = {'quality_' + key: value for key, value in self._counters.items()}
        if self.lift_rotations:
            result.update({'quality_lift_' + k: v for k,v in self._lift_work.items()})
        return result

    def export_state(self):
        result = {'schema': self.SCHEMA, 'problem': self.fingerprint, 'counters': self._counters.copy()}
        if self.lift_rotations:
            result['lift_work'] = self._lift_work.copy()
        return result

    def restore_state(self, state):
        fields = {'schema', 'problem', 'counters'} | ({'lift_work'} if self.lift_rotations else set())
        if type(state) is not dict or set(state) != fields:
            raise ValueError('Invalid quality state fields')
        if state['schema'] != self.SCHEMA or state['problem'] != self.fingerprint:
            raise ValueError('Quality state belongs to another schema or puzzle')
        self._validate_counters(state['counters'])
        if self.lift_rotations:
            lift = state['lift_work']
            if type(lift) is not dict or set(lift) != set(self.LIFT_COUNTERS) or any(
                type(n) is not int or not 0 <= n <= self.MAX_COUNTER for n in lift.values()):
                raise ValueError('Invalid saved lift work')
            self._lift_work = lift.copy()
        self._counters = state['counters'].copy()
