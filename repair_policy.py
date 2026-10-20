"""Bounded heuristic choices for structured repair; no branch is pruned here.

Pair quality measures available permutation-cycle choices, not solution quality.
Adaptive region rewards are independently validated score gains supplied by the
caller. No result, including a timeout, supplies a full-puzzle exhaustion claim.
"""
from __future__ import annotations

import itertools
import math
import random
from exact_model import constraint_edges, frame_ok, grid_edges


def _number(name, value, minimum=0., maximum=1e12):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(name + ' must be a finite number in range')
    return float(value)


def _integer(name, value, minimum=0, maximum=2**53-1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(name + ' must be an integer in range')
    return value


class PairSelector:
    """Select from <=64 legal parents using <=32 sampled pair evaluations.

    Immutable structural data is detached at construction. Every sampled parent
    is checked for physical uniqueness, domains, frame and oriented clues. The
    full-grid check and constraints themselves are not rebuilt per parent. The
    downstream recombination still validates its own inputs and complete result.
    """
    def __init__(self, problem):
        if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
            raise ValueError('Pair selection requires the full-grid problem')
        self.size = problem.size
        self.cells = self.size**2
        self.faces = tuple(tuple(row) for row in problem.faces)
        self.domains = tuple(frozenset(domain) for domain in problem.domains)
        self.fixed = tuple(sorted(problem.fixed.items()))
        self.edges = grid_edges(self.size)

    def _parent(self, board):
        values = tuple(itertools.islice(iter(board), self.cells + 1))
        if (len(values) != self.cells or
                any(type(s) is not int or not 0 <= s < 4*self.cells for s in values) or
                len({s//4 for s in values}) != self.cells or
                any(s not in self.domains[c] or not frame_ok(self.size, c, self.faces[s])
                    for c, s in enumerate(values)) or
                any(values[c] != s for c, s in self.fixed)):
            raise ValueError('Pair parent violates physical pieces, domains, frame or clues')
        locations = [0] * self.cells
        for cell, state in enumerate(values):
            locations[state//4] = cell
        bad = set()
        score = 0
        for a, b, sa, sb in self.edges:
            if self.faces[values[a]][sa] == self.faces[values[b]][sb]:
                score += 1
            else:
                bad.add(a); bad.add(b)
        return values, locations, bad, score

    @staticmethod
    def _features(left, right):
        a, locations, bad_a, score_a = left
        b, _, bad_b, score_b = right
        mapping = [locations[s//4] for s in b]
        visited = set(); active = []; nontrivial = 0
        for start in range(len(mapping)):
            if start in visited:
                continue
            cycle = []; cell = start
            while cell not in visited:
                visited.add(cell); cycle.append(cell); cell = mapping[cell]
            if cell != start:
                raise RuntimeError('Invalid physical-piece permutation')
            if any(a[c] != b[c] for c in cycle):
                active.append(cycle)
                nontrivial += len(cycle) > 1
        bad = bad_a | bad_b
        mismatch_cycles = sum(any(c in bad for c in cycle) for cycle in active)
        largest = max(map(len, active), default=0)
        # At least two binary components allow outcomes other than the parents.
        # Cap the count term so many orientation ties cannot dominate indefinitely.
        rank = (len(active) >= 2, min(len(active), 16), min(mismatch_cycles, 16),
                max(score_a, score_b), -largest)
        return dict(active_cycles=len(active), nontrivial_cycles=nontrivial,
                    mismatch_cycles=mismatch_cycles, largest_cycle=largest,
                    parent_scores=[score_a, score_b]), rank

    def choose(self, bank, *, seed=0, samples=8, should_stop=None):
        _integer('seed', seed, 0, 2**32-1)
        _integer('samples', samples, 1, 32)
        if should_stop is not None and not callable(should_stop):
            raise ValueError('should_stop must be callable')
        parents = list(itertools.islice(iter(bank), 65))
        if len(parents) > 64:
            raise ValueError('At most 64 parents are permitted')
        if len(parents) < 2 or (should_stop and should_stop()):
            return None
        rng = random.Random(seed)
        pairs = list(itertools.combinations(range(len(parents)), 2))
        selected = rng.sample(pairs, min(samples, len(pairs)))
        checked = {}; best = None; best_rank = None; examined = 0
        for i, j in selected:
            if should_stop and should_stop():
                return None
            for index in (i, j):
                if index not in checked:
                    checked[index] = self._parent(parents[index])
            features, rank = self._features(checked[i], checked[j])
            examined += 1
            rank = (*rank, rng.getrandbits(64))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = dict(left=i, right=j, **features)
        if should_stop and should_stop():
            return None
        return dict(best, examined_pairs=examined)


class AdaptiveRegions:
    """Deterministic bounded scheduling for four size/donor/time profiles.

    The caller supplies total search elapsed time (including resumed elapsed),
    remaining wall time, and actual elapsed time plus validated gain per attempt.
    Native overrun is charged in full, so it creates debt before another plan.
    Exploration picks the least-tried arm on at least one in five observations;
    with four arms, no arm is permanently disabled for lacking an early success.
    If all rewards are zero, all four arms continue in round-robin order.
    """
    SCHEMA = 'eternity-adaptive-regions/v1'
    ARMS = ((8, 2, .125), (12, 4, .25), (16, 4, .5), (20, 6, 1.))
    def __init__(self, *, max_seconds=.25, share=.2):
        self.max_seconds = _number('max_seconds', max_seconds, .02, 5.)
        self.share = _number('share', share, .01, .5)
        self.attempts = [0] * len(self.ARMS)
        self.seconds = [0.] * len(self.ARMS)
        self.gains = [0] * len(self.ARMS)

    def plan(self, *, elapsed_seconds, remaining_seconds, should_stop=None):
        elapsed = _number('elapsed_seconds', elapsed_seconds)
        remaining = _number('remaining_seconds', remaining_seconds)
        if should_stop is not None and not callable(should_stop):
            raise ValueError('should_stop must be callable')
        if should_stop and should_stop():
            return None
        observations = sum(self.attempts)
        if 0 in self.attempts or observations % 5 == 0 or not any(self.gains):
            arm = min(range(len(self.ARMS)), key=lambda i: (self.attempts[i], i))
        else:
            arm = max(range(len(self.ARMS)),
                      key=lambda i: (self.gains[i] / max(.001, self.seconds[i]),
                                     -self.attempts[i], -i))
        size, donors, fraction = self.ARMS[arm]
        allowance = min(max(.02, self.max_seconds * fraction), remaining,
                        max(0., self.share * elapsed - sum(self.seconds)))
        if allowance < .02:
            return None
        return dict(arm=arm, region_size=size, donor_count=donors, seconds=allowance)

    def observe(self, arm, *, elapsed_seconds, score_gain):
        _integer('arm', arm, 0, len(self.ARMS)-1)
        elapsed = _number('elapsed_seconds', elapsed_seconds)
        gain = _integer('score_gain', score_gain, 0, 480)
        if self.attempts[arm] >= 2**53-1 or self.gains[arm] + gain > 2**53-1 or self.seconds[arm] + elapsed > 1e12:
            raise ValueError('Adaptive history limit exceeded')
        self.attempts[arm] += 1
        self.seconds[arm] += elapsed
        self.gains[arm] += gain

    def export_state(self):
        return dict(schema=self.SCHEMA, max_seconds=self.max_seconds, share=self.share,
                    attempts=list(self.attempts), seconds=list(self.seconds), gains=list(self.gains))

    def restore_state(self, state):
        required = {'schema', 'max_seconds', 'share', 'attempts', 'seconds', 'gains'}
        if type(state) is not dict or set(state) != required or state['schema'] != self.SCHEMA:
            raise ValueError('Invalid adaptive-region checkpoint schema')
        max_seconds = _number('max_seconds', state['max_seconds'], .02, 5.)
        share = _number('share', state['share'], .01, .5)
        if max_seconds != self.max_seconds or share != self.share:
            raise ValueError('Adaptive-region checkpoint configuration mismatch')
        for key in ('attempts', 'seconds', 'gains'):
            if type(state[key]) is not list or len(state[key]) != len(self.ARMS):
                raise ValueError('Invalid adaptive-region checkpoint vectors')
        attempts = [_integer('attempts', x) for x in state['attempts']]
        seconds = [_number('seconds', x) for x in state['seconds']]
        gains = [_integer('gains', x) for x in state['gains']]
        if any((count == 0 and (spent != 0 or gain != 0)) or gain > 480*count
               for count, spent, gain in zip(attempts, seconds, gains)):
            raise ValueError('Inconsistent adaptive-region checkpoint history')
        self.attempts, self.seconds, self.gains = attempts, seconds, gains

