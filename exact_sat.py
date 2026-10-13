"""Bounded, offline SAT with a reusable assumption session.

The base encoding uses cell/piece one-hot constraints and shared edge-color
channels. A session retains one Glucose instance, its learned clauses, and
selector implications across assumption queries. No assumption is a hard
placement: removing its selector restores the corresponding alternatives.
"""
from __future__ import annotations

import math
import random
import threading
import time


class _Cancelled(Exception):
    def __init__(self, outcome):
        self.outcome = outcome


class _Infeasible(Exception):
    pass


class _Budget:
    def __init__(self, seconds, should_stop):
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.should_stop = should_stop
        self.outcome = None
        self.callback_error = None

    def reason(self):
        if self.outcome is None:
            try:
                stopped = self.should_stop is not None and self.should_stop()
            except BaseException as exc:
                self.callback_error = exc
                stopped = True
            if stopped:
                self.outcome = 'stopped'
            elif time.monotonic() >= self.deadline:
                self.outcome = 'timeout'
        return self.outcome

    def check(self):
        reason = self.reason()
        if reason:
            raise _Cancelled(reason)


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('seconds must be finite and nonnegative')
    return float(value)


class AssumptionSession:
    """One native model, repeated temporary placement queries, one total budget.

    Construction starts the total deadline; the first solve lazily encodes the
    base problem. ``solve(..., seconds=...)`` can impose a shorter query budget.
    A native unknown/short query timeout never creates a core or poisons a later
    query. An interrupted *encoding* is discarded and may be rebuilt if the
    total budget remains. Completed base models are never rebuilt.

    Use a context manager or call close(). Queries are serialized. During a
    query, close() requests cancellation and returns immediately, even inside
    a callback; query cleanup then deletes the solver after its monitor exits.
    Callback time and selector preparation count against budgets.
    Glucose uses one worker. Seed sets initial placement phases only.
    """
    def __init__(self, problem, seconds=30.0, seed=0, workers=1, should_stop=None,
                 on_progress=None, *, reverse_support=True):
        seconds = _seconds(seconds)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError('workers must be a positive integer')
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError('seed must be an integer')
        if should_stop is not None and not callable(should_stop):
            raise ValueError('should_stop must be callable')
        if on_progress is not None and not callable(on_progress):
            raise ValueError('on_progress must be callable')
        if type(reverse_support) is not bool:
            raise ValueError('reverse_support must be a boolean')
        self.budget = _Budget(seconds, should_stop)
        from exact_model import ExactProblem
        # ExactProblem is frozen, but its fixed mapping and caller-supplied
        # containers need not be. Encoding and validation must share this
        # detached snapshot for the entire lifetime of the native CNF.
        self.problem = ExactProblem(problem.size, tuple(tuple(face) for face in problem.faces),
                                    dict(problem.fixed), tuple(tuple(domain) for domain in problem.domains),
                                    None if problem.edges is None else tuple(tuple(edge) for edge in problem.edges))
        self.seed, self.requested_workers = seed, workers
        self.reverse_support, self.on_progress = reverse_support, on_progress
        self.native = None
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._closing = threading.Event()
        self.closed = False
        self._active_query = False
        self._built = self._base_unsat = False
        self._query_deadline = self.budget.deadline
        self._query_outcome = None
        self.next_var = self.clause_count = 0
        self.cell_vars = []
        self.placements = []
        self.selectors = {}
        self.model_builds = self.queries = 0
        self.total_build_seconds = 0.0

    def __enter__(self):
        if self.closed:
            raise RuntimeError('SAT session is closed')
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._closing.set()
        # Never wait for the query lock: should_stop may call this method from
        # the very monitor which that query must join. The separate lifecycle
        # lock is held only while entering/leaving a query, never during a
        # callback, native search or monitor join. It closes the race between
        # a final cancellation check and releasing the query lock.
        with self._lifecycle_lock:
            if not self._active_query:
                self._dispose()

    def _dispose(self):
        if not self.closed:
            if self.native is not None:
                self.native.delete()
                self.native = None
            self.closed = True

    def _reason(self):
        if self._query_outcome is None:
            self._query_outcome = self.budget.reason()
            if self._query_outcome is None and self._closing.is_set():
                self._query_outcome = 'stopped'
            if self._query_outcome is None and time.monotonic() >= self._query_deadline:
                self._query_outcome = 'timeout'
        return self._query_outcome

    def check(self):
        reason = self._reason()
        if reason:
            raise _Cancelled(reason)

    def _new_var(self):
        self.next_var += 1
        if self.next_var % 512 == 0:
            self.check()
        return self.next_var

    def _add(self, clause):
        if self.clause_count % 512 == 0:
            self.check()
        self.native.add_clause(clause)
        self.clause_count += 1

    def _at_most_one(self, literals):
        if len(literals) < 2:
            return
        if len(literals) == 2:
            self._add([-literals[0], -literals[1]])
            return
        previous = self._new_var()
        self._add([-literals[0], previous])
        for literal in literals[1:-1]:
            current = self._new_var()
            self._add([-literal, current])
            self._add([-previous, current])
            self._add([-literal, -previous])
            previous = current
        self._add([-literals[-1], -previous])

    def _exactly_one(self, literals):
        if not literals:
            raise _Infeasible()
        self._add(literals)
        self._at_most_one(literals)

    def _progress(self, phase):
        if self.on_progress is not None:
            self.on_progress({'backend': 'pysat-glucose4', 'phase': phase,
                              'elapsed_seconds': time.monotonic() - self.budget.started,
                              'variables': self.next_var, 'clauses': self.clause_count,
                              'model_builds': self.model_builds, 'assumption_queries': self.queries})
        self.check()

    def _build(self):
        self.check()
        try:
            from pysat.solvers import Glucose4
        except ImportError as exc:
            raise RuntimeError('The SAT backend requires the python-sat package') from exc
        from exact_model import constraint_edges
        self.check()
        problem = self.problem
        size, faces, fixed = problem.size, problem.faces, problem.fixed
        if type(size) is not int or size < 1:
            raise ValueError('Problem size must be a positive integer')
        count = size * size
        if len(faces) != 4 * count or any(len(face) != 4 for face in faces):
            raise ValueError('Expected four oriented NESW states per piece')
        if len(problem.domains) != count:
            raise ValueError('Expected one domain per board cell')
        if any(type(c) is not int or not 0 <= c < count or type(s) is not int or not 0 <= s < len(faces)
               for c, s in fixed.items()):
            raise ValueError('Invalid fixed cell or state')
        occupied = {state // 4 for state in fixed.values()}
        self.native = Glucose4(incr=True)
        self.model_builds += 1
        self._progress('encoding')
        piece_vars = [[] for _ in range(count)]
        for cell, domain in enumerate(problem.domains):
            self.check()
            exterior = (cell < size, cell % size == size - 1,
                        cell >= count - size, cell % size == 0)
            placements, seen = [], set()
            for index, state in enumerate(domain):
                if index % 256 == 0:
                    self.check()
                if type(state) is not int or not 0 <= state < len(faces):
                    raise ValueError('Invalid state in cell domain')
                if state in seen:
                    continue
                seen.add(state)
                if cell in fixed and state != fixed[cell]:
                    continue
                if cell not in fixed and state // 4 in occupied:
                    continue
                if any((color == 0) != boundary for color, boundary in zip(faces[state], exterior)):
                    continue
                literal = self._new_var()
                placements.append((state, literal))
                piece_vars[state // 4].append(literal)
            self.cell_vars.append(placements)
            self._exactly_one([literal for _, literal in placements])
        for literals in piece_vars:
            self.check()
            self._exactly_one(literals)
        for a, b, sa, sb in constraint_edges(problem):
            self.check()
            if not (0 <= a < count and 0 <= b < count and 0 <= sa < 4 and 0 <= sb < 4):
                raise ValueError('Invalid constrained edge')
            common = sorted({faces[s][sa] for s, _ in self.cell_vars[a]} &
                            {faces[s][sb] for s, _ in self.cell_vars[b]})
            if not common:
                raise _Infeasible()
            channels = {color: self._new_var() for color in common}
            self._at_most_one(list(channels.values()))
            for cell, side in ((a, sa), (b, sb)):
                support = {color: [] for color in common}
                for state, literal in self.cell_vars[cell]:
                    color = faces[state][side]
                    channel = channels.get(color)
                    self._add([-literal, channel] if channel is not None else [-literal])
                    if channel is not None:
                        support[color].append(literal)
                if self.reverse_support:
                    for color, literals in support.items():
                        self._add([-channels[color], *literals])
        rng = random.Random(self.seed)
        phases = []
        for placements in self.cell_vars:
            self.check()
            phases.extend(literal if rng.getrandbits(1) else -literal for _, literal in placements)
        self.native.set_phases(phases)
        self.placements = [dict(values) for values in self.cell_vars]
        self.check()
        self._built = True

    def _assumptions(self, assumptions):
        if assumptions is None:
            return None
        if not isinstance(assumptions, dict):
            raise ValueError('assumptions must map cells to oriented states')
        if any(type(c) is not int or not 0 <= c < self.problem.size ** 2 or
               type(s) is not int or not 0 <= s < len(self.problem.faces) for c, s in assumptions.items()):
            raise ValueError('Invalid assumed placement')
        return dict(assumptions)

    def solve(self, assumptions=None, *, seconds=None):
        assumptions = self._assumptions(assumptions)
        if seconds is not None:
            seconds = _seconds(seconds)
        with self._lock:
            with self._lifecycle_lock:
                if self.closed or self._closing.is_set():
                    raise RuntimeError('SAT session is closed')
                if self._active_query:
                    raise RuntimeError('SAT session query is already active')
                self._active_query = True
            try:
                return self._solve_locked(assumptions, seconds)
            finally:
                with self._lifecycle_lock:
                    self._active_query = False
                    if self._closing.is_set():
                        self._dispose()

    def _solve_locked(self, assumptions, seconds):
        self._query_outcome = None
        started = time.monotonic()
        self._query_deadline = self.budget.deadline if seconds is None else min(self.budget.deadline, started + seconds)
        self.queries += 1
        before = {} if self.native is None else self.native.accum_stats() or {}
        stats = dict(before)
        outcome, board, core = 'timeout', None, None
        done, monitor = threading.Event(), None
        selectors = {}
        build_started = time.monotonic()
        build_seconds = None
        try:
            self.check()
            if not self._built:
                self._build()
            if self._base_unsat:
                raise _Infeasible()
            if assumptions is not None:
                for cell, state in sorted(assumptions.items()):
                    self.check()
                    key = (cell, state)
                    selector = self.selectors.get(key)
                    if selector is None:
                        selector = self._new_var()
                        literal = self.placements[cell].get(state)
                        self._add([-selector, literal] if literal is not None else [-selector])
                        self.selectors[key] = selector
                    selectors[selector] = key
            self.check()
            build_seconds = time.monotonic() - build_started
            self._progress('searching')
            # Real PySAT supports this. getattr keeps existing injected
            # one-query test solvers compatible with the public wrapper.
            clear = getattr(self.native, 'clear_interrupt', None)
            if clear is not None:
                clear()
            self.check()
            def interrupt_if_needed():
                while not done.wait(.01):
                    if self._reason():
                        self.native.interrupt()
            monitor = threading.Thread(target=interrupt_if_needed, name='eternity-sat-interrupt', daemon=True)
            monitor.start()
            if assumptions is None:
                satisfiable = self.native.solve_limited(expect_interrupt=True)
            else:
                satisfiable = self.native.solve_limited(assumptions=list(selectors), expect_interrupt=True)
            done.set()
            monitor.join()
            self.check()
            if satisfiable is False:
                outcome = 'infeasible'
                if assumptions is not None:
                    raw_core = self.native.get_core() or []
                    if any(literal not in selectors for literal in raw_core):
                        raise RuntimeError('SAT returned an unknown assumption in its core')
                    core = sorted(selectors[literal] for literal in raw_core)
            elif satisfiable is True:
                from exact_model import check_solution
                selected = {literal for literal in self.native.get_model() if literal > 0}
                candidate = []
                for placements in self.cell_vars:
                    self.check()
                    states = [state for state, literal in placements if literal in selected]
                    if len(states) != 1:
                        raise RuntimeError('SAT model did not select exactly one state per cell')
                    candidate.append(states[0])
                if not check_solution(self.problem, candidate) or any(candidate[c] != s for c, s in (assumptions or {}).items()):
                    raise RuntimeError('SAT result failed independent exact-board/assumption validation')
                self.check()
                board, outcome = candidate, 'solved'
            # None means unknown; never inspect a stale model/core.
        except _Cancelled as exc:
            outcome = exc.outcome
        except _Infeasible:
            self._base_unsat = self._built = True
            outcome = self._reason() or 'infeasible'
            if outcome == 'infeasible' and assumptions is not None:
                core = []
        finally:
            done.set()
            if monitor is not None:
                monitor.join()
            if build_seconds is None:
                build_seconds = time.monotonic() - build_started
            self.total_build_seconds += build_seconds
            if self.native is not None:
                stats = self.native.accum_stats() or {}
                if not self._built:
                    # An interrupted partial CNF is not a reusable model.
                    self.native.delete()
                    self.native = None
                    self.next_var = self.clause_count = 0
                    self.cell_vars, self.placements, self.selectors = [], [], {}
        if self.budget.callback_error is not None:
            raise RuntimeError('SAT cancellation callback failed') from self.budget.callback_error
        return {'outcome': outcome, 'board': board, 'backend': 'pysat-glucose4',
                'branches': max(0, int(stats.get('decisions', 0)) - int(before.get('decisions', 0))),
                'conflicts': max(0, int(stats.get('conflicts', 0)) - int(before.get('conflicts', 0))),
                'cumulative_branches': int(stats.get('decisions', 0)),
                'cumulative_conflicts': int(stats.get('conflicts', 0)),
                'elapsed_seconds': time.monotonic() - started, 'build_seconds': build_seconds,
                'session_elapsed_seconds': time.monotonic() - self.budget.started,
                'session_build_seconds': self.total_build_seconds,
                'model_builds': self.model_builds, 'assumption_queries': self.queries,
                'variables': self.next_var, 'clauses': self.clause_count, 'workers': 1,
                'requested_workers': self.requested_workers, 'seed': self.seed,
                'seed_policy': 'initial-placement-phases', 'reverse_support': self.reverse_support,
                'core_cells': None if core is None else [cell for cell, _ in core],
                'core_placements': None if core is None else dict(core),
                'proof_scope': 'supplied-constraints-with-assumptions' if assumptions is not None else 'supplied-constraints'}


def solve(problem, seconds=30.0, seed=0, workers=1, should_stop=None, on_progress=None,
          *, reverse_support=True, assumptions=None):
    """Single-query compatibility wrapper; use AssumptionSession to retain CNF."""
    with AssumptionSession(problem, seconds, seed, workers, should_stop, on_progress,
                           reverse_support=reverse_support) as session:
        return session.solve(assumptions)


def solve_with_assumptions(problem, assumptions, **kwargs):
    """Temporary freezes; core is None unless this query proves infeasibility.

    An empty UNSAT core means the base supplied constraints are inconsistent.
    Explicit regional edges retain their narrower scope.
    """
    return solve(problem, assumptions=assumptions, **kwargs)
