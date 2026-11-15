"""Bounded joint checkerboard color inventories for exact full-grid search.

Every matched nonzero color edge has one endpoint on each checkerboard half.
Thus the same physical-piece subset placed on black cells must contain exactly
half of every color inventory. Separate one-color tests can miss joint failure.
This relaxation ignores exact piece locations and adjacency, so feasible never
means solved. Incomplete enumeration never rejects or mutates any domain.
"""
from __future__ import annotations
from collections import Counter
from types import MappingProxyType
from exact_model import constraint_edges, grid_edges

MAX_PIECES = 256
MAX_WORK = 1000000
MAX_STATES = 100000


def _int(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value


class _Incomplete(Exception):
    pass


class _Budget:
    def __init__(self, max_work, should_stop):
        self.limit = _int('max_work', max_work, 0, MAX_WORK)
        if should_stop is not None and not callable(should_stop):
            raise ValueError('should_stop must be callable')
        self.stop = should_stop
        self.work = self.preprocessing = self.transitions = self.copies = 0
        self.peak = 0

    def spend(self, amount=1, kind='preprocessing'):
        if self.stop is not None and self.stop():
            raise _Incomplete('stopped')
        if self.work + amount > self.limit:
            raise _Incomplete('work_limit')
        self.work += amount
        setattr(self, kind, getattr(self, kind) + amount)

    def result(self, consistent, complete, termination, **extra):
        return dict(consistent=consistent, complete=complete, termination=termination,
                    work=self.work, preprocessing_work=self.preprocessing,
                    dp_transitions=self.transitions, state_copies=self.copies,
                    peak_states=self.peak, **extra)


def _vector_dp(weights, count, target, budget, max_states):
    """Exact nonnegative cardinality knapsack; group equal vectors into chunks."""
    width = len(target)
    if count < 0 or count > len(weights) or any(t < 0 for t in target):
        return False, 'impossible_count_or_target'
    # Necessary coordinate ranges are cheap proofs; they do not replace the
    # joint DP. Each inspected scalar is included in preprocessing work.
    groups = Counter()
    totals = [0] * width
    for vector in weights:
        budget.spend(width)
        groups[vector] += 1
        for i, value in enumerate(vector): totals[i] += value
    if any(t > maximum for t, maximum in zip(target, totals)):
        return False, 'inventory_shortage'
    zero = (0,) * (width + 1)
    goal = (count, *target)
    states = {zero}
    budget.peak = max(budget.peak, 1)
    if goal == zero:
        return True, 'feasible_subset'
    chunks = []
    for vector, quantity in sorted(groups.items()):
        step = 1
        while quantity:
            budget.spend()
            take = min(step, quantity)
            chunks.append((take, tuple(take * value for value in vector)))
            quantity -= take; step *= 2
    for take, shifts in chunks:
        budget.spend(len(states), 'copies')
        after = states.copy()
        for state in states:
            budget.spend(1, 'transitions')
            used = state[0] + take
            sums = tuple(value + shift for value, shift in zip(state[1:], shifts))
            if used <= count and all(value <= bound for value, bound in zip(sums, target)):
                after.add((used, *sums))
                budget.peak = max(budget.peak, len(after))
                if len(after) > max_states:
                    raise _Incomplete('state_limit')
        states = after
        if goal in states:
            return True, 'feasible_subset'
    return False, 'joint_inventory_impossible'


def cardinality_vectors(weights, count, target, *, max_work=50000, max_states=20000,
                        should_stop=None):
    """Low-level bounded 1..3-color oracle, independent of a puzzle representation.

    This public helper is useful for exhaustive finite tests. Weights are small
    nonnegative per-piece counts, not interchangeable physical piece identities.
    Grouping equal vectors is valid only for this eligibility-free relaxation.
    """
    _int('max_states', max_states, 1, MAX_STATES)
    _int('count', count, 0, MAX_PIECES)
    if type(target) not in (tuple, list) or not 1 <= len(target) <= 3:
        raise ValueError('target needs 1..3 integer coordinates')
    target = tuple(target)
    if any(type(t) is not int or not 0 <= t <= 4 * MAX_PIECES for t in target):
        raise ValueError('Invalid vector target')
    if type(weights) not in (tuple, list) or len(weights) > MAX_PIECES:
        raise ValueError('weights needs at most256 piece vectors')
    if any(type(w) not in (tuple, list) or len(w) != len(target)
           or any(type(x) is not int or not 0 <= x <= 4 for x in w) for w in weights):
        raise ValueError('Invalid piece color vector')
    weights = tuple(tuple(w) for w in weights)
    budget = _Budget(max_work, should_stop)
    try:
        budget.spend()
        feasible, why = _vector_dp(weights, count, target, budget, max_states)
        return budget.result(feasible, True, why)
    except _Incomplete as exc:
        return budget.result(True, False, str(exc))


class JointInventory:
    """Reusable immutable inventory tables; at most256 pieces/1024 rotations.

    Static setup is bounded by the input limits and reported as setup_work.
    check() includes all runtime domain/eligibility scans and DP work in its cap.
    Neither instance setup nor a check reads files or modifies its input.
    """
    def __init__(self, problem):
        self.size = _int('size', problem.size, 1, 16)
        self.n = self.size**2
        self.faces = tuple(tuple(face) for face in problem.faces)
        if len(self.faces) != 4 * self.n or any(len(f) != 4 for f in self.faces):
            raise ValueError('Expected four oriented states per piece')
        self.full_grid = set(constraint_edges(problem)) == set(grid_edges(self.size))
        self.invariant = all(len({tuple(sorted(f)) for f in self.faces[4*p:4*p+4]}) == 1
                             for p in range(self.n))
        self.setup_work = 8 * len(self.faces)
        frame = {}
        for state, face in enumerate(self.faces):
            key = tuple(color == 0 for color in face)
            frame[key] = frame.get(key, 0) | (1 << state)
        self._black = tuple((c//self.size + c%self.size)%2 == 0 for c in range(self.n))
        self.black_count = sum(self._black)
        self._piece_mask = sum(1 << (4*p) for p in range(self.n))
        allowed = []; frames = []
        for cell, domain in enumerate(problem.domains):
            if len(domain) > len(self.faces):
                raise ValueError('Cell domain exceeds the number of oriented states')
            self.setup_work += len(domain) + 1
            allowed.append(sum(1 << state for state in set(domain)))
            outside = (cell < self.size, cell%self.size == self.size-1,
                       cell >= self.n-self.size, cell%self.size == 0)
            frames.append(frame.get(outside, 0))
        self._allowed = tuple(allowed); self._frames = tuple(frames)
        piece_counts = [Counter(self.faces[4*p]) for p in range(self.n)]
        self.setup_work += 4 * self.n
        palette = sorted({color for counter in piece_counts for color in counter if color != 0})
        self.setup_work += 2 * self.n * len(palette)
        self._weights = MappingProxyType({color: tuple(counter[color] for counter in piece_counts) for color in palette})
        self._totals = MappingProxyType({color: sum(weights) for color, weights in self._weights.items()})
        self.palette = tuple(sorted(palette, key=lambda color: (self._totals[color], color)))

    def check(self, domains, *, colors=None, width=2, max_work=50000, max_states=20000,
              should_stop=None):
        _int('width', width, 1, 3)
        _int('max_states', max_states, 1, MAX_STATES)
        if type(domains) not in (tuple, list) or len(domains) != self.n:
            raise ValueError('Expected one bounded state-bitset domain per cell')
        if any(type(d) is not int or d < 0 or d >> len(self.faces) for d in domains):
            raise ValueError('Invalid state-bitset domain')
        domains = tuple(domains)
        if colors is None:
            chosen = self.palette[:width]
        else:
            if type(colors) not in (tuple, list) or not 1 <= len(colors) <= 3:
                raise ValueError('Choose1..3 distinct nonzero puzzle colors')
            chosen = tuple(colors)
            if any(type(c) is not int or c not in self._weights for c in chosen) or len(set(chosen)) != len(chosen):
                raise ValueError('Invalid or repeated inventory color')
        budget = _Budget(max_work, should_stop)
        extra = dict(colors=list(chosen), setup_work=self.setup_work,
                     proof_scope='necessary-checkerboard-piece-subset-inventory',
                     fixed_black_pieces=0, flexible_pieces=0, target=[], required_black=0)
        def result(consistent, complete, why):
            return budget.result(consistent, complete, why, **extra)
        try:
            budget.spend()
            if not self.full_grid or not self.invariant:
                return result(True, False, 'unsupported_grid_or_rotations')
            if not chosen:
                return result(True, True, 'no_nonzero_colors')
            eligible = [0] * self.n
            for cell, domain in enumerate(domains):
                budget.spend()
                if domain & ~self._allowed[cell]:
                    raise ValueError('Domain is not a subset of this problem')
                if domain & ~self._frames[cell]:
                    return result(True, False, 'unframed_domains')
                if domain == 0:
                    return result(False, True, 'empty_cell_domain')
                pieces = (domain | domain >> 1 | domain >> 2 | domain >> 3) & self._piece_mask
                while pieces:
                    budget.spend()
                    bit = pieces & -pieces; pieces ^= bit
                    eligible[(bit.bit_length()-1)//4] |= 1 if self._black[cell] else 2
            fixed = []; flexible = []
            for p, mask in enumerate(eligible):
                budget.spend()
                if mask == 0: return result(False, True, 'piece_has_no_cell')
                if mask == 1: fixed.append(p)
                elif mask == 3: flexible.append(p)
            extra.update(fixed_black_pieces=len(fixed), flexible_pieces=len(flexible),
                         required_black=self.black_count-len(fixed))
            target = []
            for color in chosen:
                budget.spend(len(fixed)+1)
                if self._totals[color]%2:
                    return result(False, True, 'odd_color_inventory')
                target.append(self._totals[color]//2-sum(self._weights[color][p] for p in fixed))
            extra['target'] = target.copy()
            vectors = []
            for piece in flexible:
                budget.spend(len(chosen))
                vectors.append(tuple(self._weights[color][piece] for color in chosen))
            feasible, why = _vector_dp(vectors, extra['required_black'], tuple(target), budget, max_states)
            return result(feasible, True, why)
        except _Incomplete as exc:
            return result(True, False, str(exc))
