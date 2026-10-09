"""Portable exact DFS baseline; Python timings are not native-DFS benchmarks.

Every pruning rule preserves solutions: frame/fixed/domain intersections,
adjacency arc consistency, forced-piece exclusion, unique piece locations, and
bipartite perfect-matching feasibility. Explicitly supplied constraint edges
may describe a regional repair rather than the complete puzzle.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import random
import time

from exact_model import check_solution, constraint_edges, problem_hash

CHECKPOINT_VERSION = 2
ORDERING_VERSION = 'mrv-degree-lcv-hint-v1'
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024
_COUNTERS = ('branches', 'conflicts', 'propagations', 'max_depth', 'matching_checks', 'matching_failures')


def _seal(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise ValueError('DFS checkpoint exceeds the size limit')
    return dict(value, checksum=hashlib.sha256(payload).hexdigest())


def _hint_data(hints, n):
    """Validate bounded ordering data, without imposing puzzle constraints."""
    if hints is None:
        hints = ()
    if not isinstance(hints, (list, tuple)) or len(hints) > 64:
        raise ValueError('hints must contain at most 64 complete state boards')
    boards = []
    for board in hints:
        if not isinstance(board, (list, tuple)) or len(board) != n:
            raise ValueError('Each hint board must contain exactly size squared states')
        if any(type(state) is not int or not 0 <= state < 4 * n for state in board):
            raise ValueError('Hint states must be plain integers in the puzzle state range')
        boards.append(tuple(board))
    raw = json.dumps(boards, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return tuple(boards), hashlib.sha256(raw).hexdigest()


def _checked_checkpoint(value, identity, seed, n, hint_identity):
    """Bounded data validation; the checksum detects corruption, not forgery."""
    keys = {'version', 'ordering_version', 'problem_hash', 'hint_hash', 'seed', 'stack', 'rng_state', 'counters', 'checksum'}
    if not isinstance(value, dict):
        raise ValueError('Invalid DFS checkpoint fields')
    if type(value.get('version')) is not int or value['version'] != CHECKPOINT_VERSION:
        raise ValueError('Unsupported DFS checkpoint version')
    if set(value) != keys:
        raise ValueError('Invalid DFS checkpoint fields')
    if value['ordering_version'] != ORDERING_VERSION:
        raise ValueError('DFS checkpoint ordering version mismatch')
    if value['hint_hash'] != hint_identity:
        raise ValueError('DFS checkpoint hint hash mismatch')
    if value['problem_hash'] != identity or type(value['seed']) is not int or value['seed'] != seed:
        raise ValueError('DFS checkpoint problem or seed mismatch')
    counters = value['counters']
    if not isinstance(counters, dict) or set(counters) != set(_COUNTERS):
        raise ValueError('Invalid DFS checkpoint counters')
    if any(type(v) is not int or not 0 <= v < 2**63 for v in counters.values()) or counters['max_depth'] > n:
        raise ValueError('DFS checkpoint counters out of range')
    stack = value['stack']
    if not isinstance(stack, list) or not 1 <= len(stack) <= n + 1:
        raise ValueError('Invalid DFS checkpoint depth')
    candidate_count = 0
    for index, frame in enumerate(stack):
        if not isinstance(frame, dict):
            raise ValueError('Invalid DFS checkpoint frame')
        if frame.get('phase') == 'pending':
            if set(frame) != {'phase'} or index != len(stack) - 1:
                raise ValueError('Only the final DFS frame may be pending')
            continue
        if set(frame) != {'phase', 'cell', 'candidates', 'next'} or frame['phase'] != 'branch':
            raise ValueError('Invalid DFS branch frame')
        choices = frame['candidates']
        if type(frame['cell']) is not int or not 0 <= frame['cell'] < n:
            raise ValueError('Invalid DFS branch cell')
        if not isinstance(choices, list) or not 2 <= len(choices) <= 4 * n:
            raise ValueError('Invalid DFS branch candidates')
        if any(type(s) is not int or not 0 <= s < 4 * n for s in choices) or len(set(choices)) != len(choices):
            raise ValueError('Invalid or repeated DFS branch state')
        if type(frame['next']) is not int or not 0 <= frame['next'] <= len(choices):
            raise ValueError('Invalid DFS branch cursor')
        if index < len(stack) - 1 and frame['next'] == 0:
            raise ValueError('DFS ancestor has no active child')
        candidate_count += len(choices)
    if candidate_count > 2 * n * (n + 1):
        raise ValueError('DFS checkpoint candidate count exceeds its bound')
    rng = value['rng_state']
    if not isinstance(rng, list) or len(rng) != 3 or type(rng[0]) is not int or rng[0] != 3 or rng[2] is not None:
        raise ValueError('Invalid DFS random state')
    if not isinstance(rng[1], list) or len(rng[1]) != 625:
        raise ValueError('Invalid DFS random state size')
    if any(type(v) is not int or not 0 <= v < 2**32 for v in rng[1][:-1]) or type(rng[1][-1]) is not int or not 0 <= rng[1][-1] <= 624:
        raise ValueError('Invalid DFS random state values')
    if not isinstance(value['checksum'], str) or len(value['checksum']) != 64:
        raise ValueError('Invalid DFS checkpoint checksum')
    body = {k: value[k] for k in keys if k != 'checksum'}
    if _seal(body)['checksum'] != value['checksum']:
        raise ValueError('DFS checkpoint checksum mismatch')
    # Detach the caller's mutable data after the bounded structural checks.
    return json.loads(json.dumps(value, separators=(',', ':')))


class _Interrupted(Exception):
    def __init__(self, outcome):
        self.outcome = outcome


class _Search:
    def __init__(self, problem, seconds, seed, should_stop, on_progress, checkpoint, on_checkpoint, checkpoint_interval, hints=None):
        self.problem = problem
        self.started = time.perf_counter()
        self.deadline = self.started + seconds
        self.should_stop = should_stop
        self.on_progress = on_progress
        self.on_checkpoint = on_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.last_progress = self.started
        self.last_checkpoint = self.started
        self.build_seconds = None
        self.branches = self.conflicts = self.propagations = self.max_depth = 0
        self.matching_checks = self.matching_failures = 0
        self.rng = random.Random(seed)
        self.seed = seed
        self.size = problem.size
        self.n = self.size * self.size
        self.identity = problem_hash(problem)
        hint_boards, self.hint_hash = _hint_data(hints, self.n)
        self.hint_count = len(hint_boards)
        self.hint_votes = [{} for _ in range(self.n)]
        for board in hint_boards:
            for cell, state in enumerate(board):
                votes = self.hint_votes[cell]
                votes[state] = votes.get(state, 0) + 1
        self.tie_order = list(range(self.n))
        self.rng.shuffle(self.tie_order)
        self.committed_rng = self.rng.getstate()
        self.stack = []
        self.restoring = False
        self.restore_complete = checkpoint is None
        self.input_checkpoint = None if checkpoint is None else _checked_checkpoint(checkpoint, self.identity, seed, self.n, self.hint_hash)
        if self.input_checkpoint is not None:
            for key, value in self.input_checkpoint['counters'].items():
                setattr(self, key, value)

    def report(self):
        return {
            'backend': 'dfs-python', 'branches': self.branches,
            'conflicts': self.conflicts, 'propagations': self.propagations,
            'matching_checks': self.matching_checks, 'matching_failures': self.matching_failures,
            'max_depth': self.max_depth,
            'elapsed_seconds': time.perf_counter() - self.started,
            'build_seconds': self.build_seconds,
            'hint_count': self.hint_count, 'hint_hash': self.hint_hash,
            'ordering_version': ORDERING_VERSION,
        }

    def check(self):
        if self.should_stop is not None and self.should_stop():
            raise _Interrupted('stopped')
        now = time.perf_counter()
        if now >= self.deadline:
            raise _Interrupted('timeout')
        if self.on_checkpoint is not None and not self.restoring and now - self.last_checkpoint >= self.checkpoint_interval:
            self.last_checkpoint = now
            self.on_checkpoint(self.checkpoint())
        if self.on_progress is not None and not self.restoring and now - self.last_progress >= .25:
            self.last_progress = now
            self.on_progress(dict(self.report(), outcome='running'))

    def build(self):
        self.check()
        self.size = int(self.problem.size)
        self.n = self.size * self.size
        self.faces = tuple(tuple(face) for face in self.problem.faces)
        if self.size < 1 or len(self.faces) != 4 * self.n:
            raise ValueError('Expected four oriented states for each board piece')
        self.piece_mask = sum(1 << (4 * p) for p in range(self.n))
        self.all_states = (1 << (4 * self.n)) - 1
        self.colors = [{} for _ in range(4)]
        for state, face in enumerate(self.faces):
            if state % 64 == 0:
                self.check()
            if len(face) != 4:
                raise ValueError('Each state requires four NESW colors')
            for side, color in enumerate(face):
                self.colors[side][color] = self.colors[side].get(color, 0) | (1 << state)
        self.neighbors = [[] for _ in range(self.n)]
        for a, b, side_a, side_b in constraint_edges(self.problem):
            self.neighbors[a].append((b, side_a, side_b))
            self.neighbors[b].append((a, side_b, side_a))
        supplied = self.problem.domains
        if supplied is not None and len(supplied) != self.n:
            raise ValueError('Expected one domain per cell')
        domains = []
        for cell in range(self.n):
            self.check()
            domain = self.all_states
            if supplied is not None:
                domain = 0
                for state in supplied[cell]:
                    if isinstance(state, bool) or not isinstance(state, int) or not 0 <= state < 4 * self.n:
                        raise ValueError('Invalid state in cell domain')
                    domain |= 1 << state
            row, col = divmod(cell, self.size)
            outside = (row == 0, col == self.size - 1, row == self.size - 1, col == 0)
            for side, border in enumerate(outside):
                gray = self.colors[side].get(0, 0)
                domain &= gray if border else self.all_states ^ gray
            if cell in self.problem.fixed:
                state = self.problem.fixed[cell]
                if isinstance(state, bool) or not isinstance(state, int) or not 0 <= state < 4 * self.n:
                    raise ValueError('Invalid fixed state')
                domain &= 1 << state
            domains.append(domain)
        self.build_seconds = time.perf_counter() - self.started
        return domains

    def checkpoint(self):
        if self.input_checkpoint is not None and not self.restore_complete:
            return json.loads(json.dumps(self.input_checkpoint))
        frames = []
        for frame in self.stack:
            if frame['phase'] == 'pending':
                frames.append({'phase': 'pending'})
            else:
                frames.append({key: frame[key] if key != 'candidates' else frame[key].copy()
                               for key in ('phase', 'cell', 'candidates', 'next')})
        if not frames:
            frames = [{'phase': 'pending'}]
        rng = self.committed_rng
        return _seal({'version': CHECKPOINT_VERSION, 'ordering_version': ORDERING_VERSION,
                      'problem_hash': self.identity, 'hint_hash': self.hint_hash,
                      'seed': self.seed, 'stack': frames,
                      'rng_state': [rng[0], list(rng[1]), rng[2]],
                      'counters': {key: getattr(self, key) for key in _COUNTERS}})

    @staticmethod
    def pending(domains, dirty, previous):
        return {'phase': 'pending', 'domains': domains, 'dirty': dirty, 'matching': previous}

    def restore(self, domains):
        """Rebuild every live frame from original domains and its decisions.

        Stored domains are deliberately absent, so a corrupt checkpoint cannot
        inject arbitrary domain deletions. Cursor claims about prior siblings
        remain unauthenticated; resumed exhaustion has a narrower proof scope.
        """
        self.restoring = True
        self.build_seconds = None
        saved_counters = {key: getattr(self, key) for key in _COUNTERS}
        frames = []
        dirty, previous = range(self.n), None
        try:
            for index, description in enumerate(self.input_checkpoint['stack']):
                self.check()
                if description['phase'] == 'pending':
                    frames.append(self.pending(domains.copy(), dirty, previous))
                    break
                current = domains.copy()
                matching = self.propagate(current, dirty, previous)
                if matching is None:
                    raise ValueError('DFS checkpoint contains an impossible branch')
                cell = description['cell']
                mask = 0
                for state in description['candidates']:
                    mask |= 1 << state
                if mask != current[cell] or current[cell] & (current[cell] - 1) == 0:
                    raise ValueError('DFS checkpoint branch does not partition its domain')
                frame = dict(description, candidates=description['candidates'].copy(), domains=current, matching=matching)
                frames.append(frame)
                if index < len(self.input_checkpoint['stack']) - 1:
                    state = description['candidates'][description['next'] - 1]
                    domains = current.copy()
                    domains[cell] = 1 << state
                    dirty, previous = (cell,), matching
            rng = self.input_checkpoint['rng_state']
            self.committed_rng = (rng[0], tuple(rng[1]), rng[2])
            self.rng.setstate(self.committed_rng)
            self.stack = frames
            self.restore_complete = True
        finally:
            for key, value in saved_counters.items():
                setattr(self, key, value)
            self.restoring = False
            self.build_seconds = time.perf_counter() - self.started

    def pieces(self, domain):
        """One bit at position 4*p for every piece present in a state bitset."""
        return (domain | (domain >> 1) | (domain >> 2) | (domain >> 3)) & self.piece_mask

    def supported(self, source, source_side, target_side):
        allowed = 0
        for color, states in self.colors[source_side].items():
            support = source & states
            if support:
                candidates = self.colors[target_side].get(color, 0)
                pieces = self.pieces(support)
                if pieces & (pieces - 1) == 0:
                    # A neighbor cannot reuse the only supporting piece.
                    candidates &= ~(15 * pieces)
                allowed |= candidates
        return allowed

    def matching(self, domains, previous):
        """Return a perfect cell/piece matching, or None (Hall contradiction).

        Retain valid parent matches and repair with augmenting paths. This is
        only feasibility pruning; it does not claim full all-different GAC.
        """
        self.matching_checks += 1
        candidates = [self.pieces(domain) for domain in domains]
        matched = [-1] * self.n
        owners = [-1] * self.n
        if previous is not None:
            for cell, piece in enumerate(previous):
                if piece >= 0 and candidates[cell] & (1 << (4 * piece)) and owners[piece] == -1:
                    matched[cell] = piece
                    owners[piece] = cell
        unmatched = sorted((i for i in range(self.n) if matched[i] < 0), key=lambda i: candidates[i].bit_count())
        for root in unmatched:
            self.check()
            queue = deque([root])
            parents = {root: -1}
            visited_pieces = 0
            found = False
            while queue and not found:
                self.check()
                cell = queue.popleft()
                choices = candidates[cell] & ~visited_pieces
                while choices:
                    flag = choices & -choices
                    choices ^= flag
                    visited_pieces |= flag
                    piece = (flag.bit_length() - 1) // 4
                    owner = owners[piece]
                    if owner < 0:
                        # Each displaced old match is the edge by which its
                        # cell was reached, so walking parents flips the path.
                        while cell >= 0:
                            old_piece = matched[cell]
                            matched[cell] = piece
                            owners[piece] = cell
                            cell = parents[cell]
                            piece = old_piece
                        found = True
                        break
                    if owner not in parents:
                        parents[owner] = cell
                        queue.append(owner)
            if not found:
                self.matching_failures += 1
                return None
        return matched

    def propagate(self, domains, dirty, previous):
        queue = deque(dirty)
        queued = set(dirty)

        def reduce(cell, domain):
            if not domain:
                return False
            if domain != domains[cell]:
                domains[cell] = domain
                self.propagations += 1
                if cell not in queued:
                    queue.append(cell)
                    queued.add(cell)
            return True

        if any(not domain for domain in domains):
            return None
        while True:
            while queue:
                self.check()
                cell = queue.popleft()
                queued.remove(cell)
                domain = domains[cell]
                pieces = self.pieces(domain)
                if pieces & (pieces - 1) == 0:
                    forbidden = pieces * 15
                    for other in range(self.n):
                        if other != cell and not reduce(other, domains[other] & ~forbidden):
                            return None
                for other, side, other_side in self.neighbors[cell]:
                    if not reduce(other, domains[other] & self.supported(domain, side, other_side)):
                        return None
            self.check()
            seen = multiple = 0
            pieces_by_cell = []
            for domain in domains:
                pieces = self.pieces(domain)
                pieces_by_cell.append(pieces)
                multiple |= seen & pieces
                seen |= pieces
            if seen != self.piece_mask:
                return None
            unique = seen & ~multiple
            for cell, pieces in enumerate(pieces_by_cell):
                required = pieces & unique
                if required:
                    # Two pieces with only this one possible cell cannot fit.
                    if required & (required - 1):
                        return None
                    if not reduce(cell, domains[cell] & (required * 15)):
                        return None
            if not queue:
                return self.matching(domains, previous)

    def branch(self, domains):
        undecided = [cell for cell, domain in enumerate(domains) if domain & (domain - 1)]
        def order(cell):
            degree = sum(bool(domains[other] & (domains[other] - 1)) for other, _, _ in self.neighbors[cell])
            return domains[cell].bit_count(), -degree, self.tie_order[cell]
        cell = min(undecided, key=order)
        candidates = []
        remaining = domains[cell]
        while remaining:
            flag = remaining & -remaining
            remaining ^= flag
            candidates.append(flag.bit_length() - 1)
        self.rng.shuffle(candidates)
        def value_support(state):
            self.check()
            other_pieces = ~(15 << (4 * (state // 4)))
            return sum((domains[other] & self.colors[other_side].get(self.faces[state][side], 0) & other_pieces).bit_count()
                       for other, side, other_side in self.neighbors[cell])
        if self.hint_count:
            candidates.sort(key=lambda state: (value_support(state), self.hint_votes[cell].get(state, 0)), reverse=True)
        else:
            candidates.sort(key=value_support, reverse=True)
        return cell, candidates

    def visit(self, domains):
        if self.input_checkpoint is not None:
            self.restore(domains)
        else:
            self.stack = [self.pending(domains, range(self.n), None)]
        while self.stack:
            self.check()
            depth = len(self.stack) - 1
            self.max_depth = max(self.max_depth, depth)
            frame = self.stack[-1]
            if frame['phase'] == 'pending':
                # Propagation and ordering work on a copy. Any interruption
                # retains the whole pending node and the pre-order RNG state.
                current = frame['domains'].copy()
                self.rng.setstate(self.committed_rng)
                matching = self.propagate(current, frame['dirty'], frame['matching'])
                if matching is None:
                    self.conflicts += 1
                    self.stack.pop()
                    continue
                if all(domain & (domain - 1) == 0 for domain in current):
                    board = [domain.bit_length() - 1 for domain in current]
                    if not check_solution(self.problem, board):
                        raise AssertionError('DFS candidate failed independent exact-model validation')
                    return board
                cell, candidates = self.branch(current)
                self.committed_rng = self.rng.getstate()
                frame = {'phase': 'branch', 'domains': current, 'matching': matching,
                         'cell': cell, 'candidates': candidates, 'next': 0}
                self.stack[-1] = frame
            if frame['next'] == len(frame['candidates']):
                self.stack.pop()
                continue
            state = frame['candidates'][frame['next']]
            frame['next'] += 1
            self.branches += 1
            child = frame['domains'].copy()
            child[frame['cell']] = 1 << state
            self.stack.append(self.pending(child, (frame['cell'],), frame['matching']))
        return None


def solve(problem, seconds=10.0, seed=0, workers=1, should_stop=None, on_progress=None,
          checkpoint=None, on_checkpoint=None, checkpoint_interval=5.0, hints=None):
    """Solve exactly within a total build+search budget, without any I/O.

    A fresh ``infeasible`` result covers the entire supplied problem. After
    resume it covers only the recorded remaining frontier: checkpoint cursors
    are not authenticated evidence of previously exhausted siblings. A
    timeout/cancellation is not a proof. Required edges are controlled by
    ExactProblem.edges. Checkpoint validation/reconstruction and periodic
    callbacks count against the time budget; very short budgets may repeatedly
    interrupt reconstruction without advancing the saved frontier.

    Optional hints are at most 64 complete list/tuple boards of state integers.
    They need not be legal puzzle boards. Their votes break ties only after
    existing value-support ordering; no hint freezes or removes a candidate.
    Checkpoint v2 binds the exact hint sequence and ordering version. A v1
    checkpoint or a changed hint sequence is rejected rather than restarted.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if isinstance(workers, bool) or workers != 1:
        raise ValueError('The portable DFS baseline supports exactly one worker')
    if type(seed) is not int or not -(2**63) <= seed < 2**64:
        raise ValueError('seed must be a bounded integer')
    if isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, (int, float)) or not math.isfinite(checkpoint_interval) or checkpoint_interval < 0:
        raise ValueError('checkpoint_interval must be finite and nonnegative')
    search = _Search(problem, seconds, seed, should_stop, on_progress, checkpoint, on_checkpoint, checkpoint_interval, hints)
    board = None
    try:
        domains = search.build()
        board = search.visit(domains)
        outcome = 'solved' if board is not None else 'infeasible'
    except _Interrupted as exc:
        outcome = exc.outcome
    result = dict(search.report(), outcome=outcome, board=board, workers=1, language='python')
    if result['build_seconds'] is None:
        result['build_seconds'] = result['elapsed_seconds']
    result['resumed'] = checkpoint is not None
    result['proof_scope'] = 'remaining_checkpoint_frontier' if checkpoint is not None else 'entire_supplied_problem'
    result['coverage_verified'] = checkpoint is None and outcome == 'infeasible'
    result['solution_verified'] = board is not None
    result['checkpoint'] = search.checkpoint() if outcome in ('timeout', 'stopped') else None
    if on_progress is not None:
        on_progress(result.copy())
    return result
