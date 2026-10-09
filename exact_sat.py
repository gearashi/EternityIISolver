"""Bounded, local-only SAT backend for exact edge-matching problems.

Cell/piece one-hot constraints and shared edge-color variables use linear-size
sequential at-most-one clauses. No pairwise placement incompatibility table is
built. The deadline covers encoding as well as native search; cancellation is
cooperative between small encoding operations and interrupts native search.
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


def solve(problem, seconds=30.0, seed=0, workers=1, should_stop=None, on_progress=None):
    """Return solved/infeasible/timeout/stopped; never treat interrupts as UNSAT.

    Glucose4 uses one native worker. ``seed`` selects initial variable phases;
    it is not a claim to configure Glucose's internal random-seed parameters.
    Callbacks must be short and thread-safe; should_stop is also polled by the
    search-interrupt monitor. No project service or GPU is used.
    """
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    budget = _Budget(float(seconds), should_stop)
    backend = 'pysat-glucose4'
    native = None
    monitor = None
    done = threading.Event()
    next_var = 0
    clause_count = 0
    build_seconds = None
    outcome = 'timeout'
    board = None
    stats = {}

    def new_var():
        nonlocal next_var
        next_var += 1
        if next_var % 512 == 0:
            budget.check()
        return next_var

    def add(clause):
        nonlocal clause_count
        if clause_count % 512 == 0:
            budget.check()
        native.add_clause(clause)
        clause_count += 1

    def at_most_one(literals):
        # Sinz sequential counter: one clause at each endpoint and three
        # per interior item. Two-item groups need only one binary clause.
        if len(literals) < 2:
            return
        if len(literals) == 2:
            add([-literals[0], -literals[1]])
            return
        previous = new_var()
        add([-literals[0], previous])
        for literal in literals[1:-1]:
            current = new_var()
            add([-literal, current])
            add([-previous, current])
            add([-literal, -previous])
            previous = current
        add([-literals[-1], -previous])

    def exactly_one(literals):
        if not literals:
            raise _Infeasible()
        add(literals)
        at_most_one(literals)

    def progress(phase):
        if on_progress is not None:
            on_progress({'backend': backend, 'phase': phase,
                         'elapsed_seconds': time.monotonic() - budget.started,
                         'variables': next_var, 'clauses': clause_count})
        budget.check()

    try:
        budget.check()
        try:
            from pysat.solvers import Glucose4
        except ImportError as exc:
            raise RuntimeError('The SAT backend requires the python-sat package') from exc
        from exact_model import check_solution, constraint_edges
        budget.check()
        size = problem.size
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError('Problem size must be a positive integer')
        count = size * size
        faces = problem.faces
        if len(faces) != 4 * count or any(len(face) != 4 for face in faces):
            raise ValueError('Expected four oriented NESW states per piece')
        if len(problem.domains) != count:
            raise ValueError('Expected one domain per board cell')
        fixed = problem.fixed
        if any(type(cell) is not int or not 0 <= cell < count or
               type(state) is not int or not 0 <= state < len(faces)
               for cell, state in fixed.items()):
            raise ValueError('Invalid fixed cell or state')
        occupied = {state // 4 for state in fixed.values()}
        native = Glucose4()
        progress('encoding')
        cell_vars = []
        piece_vars = [[] for _ in range(count)]
        for cell, domain in enumerate(problem.domains):
            budget.check()
            exterior = (cell < size, cell % size == size - 1,
                        cell >= count - size, cell % size == 0)
            placements = []
            seen = set()
            for index, state in enumerate(domain):
                if index % 256 == 0:
                    budget.check()
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
                literal = new_var()
                placements.append((state, literal))
                piece_vars[state // 4].append(literal)
            cell_vars.append(placements)
            exactly_one([literal for _, literal in placements])
        for literals in piece_vars:
            budget.check()
            exactly_one(literals)
        for a, b, sa, sb in constraint_edges(problem):
            budget.check()
            if not (0 <= a < count and 0 <= b < count and 0 <= sa < 4 and 0 <= sb < 4):
                raise ValueError('Invalid constrained edge')
            common = sorted({faces[s][sa] for s, _ in cell_vars[a]} &
                            {faces[s][sb] for s, _ in cell_vars[b]})
            if not common:
                raise _Infeasible()
            channels = {color: new_var() for color in common}
            at_most_one(list(channels.values()))
            # A selected endpoint supplies the at-least-one side implicitly.
            for cell, side in ((a, sa), (b, sb)):
                for state, literal in cell_vars[cell]:
                    channel = channels.get(faces[state][side])
                    add([-literal, channel] if channel is not None else [-literal])
        budget.check()
        rng = random.Random(seed)
        phases = []
        for placements in cell_vars:
            budget.check()
            phases.extend(literal if rng.getrandbits(1) else -literal for _, literal in placements)
        native.set_phases(phases)
        budget.check()
        build_seconds = time.monotonic() - budget.started
        progress('searching')

        def interrupt_if_needed():
            while not done.wait(0.01):
                if budget.reason():
                    native.interrupt()
                    return

        monitor = threading.Thread(target=interrupt_if_needed, name='eternity-sat-interrupt', daemon=True)
        monitor.start()
        satisfiable = native.solve_limited(expect_interrupt=True)
        done.set()
        monitor.join()
        budget.check()
        if satisfiable is None:
            outcome = 'timeout'
        elif satisfiable is False:
            outcome = 'infeasible'
        else:
            model = native.get_model()
            budget.check()
            selected = {literal for literal in model if literal > 0}
            candidate = []
            for placements in cell_vars:
                budget.check()
                states = [state for state, literal in placements if literal in selected]
                if len(states) != 1:
                    raise RuntimeError('SAT model did not select exactly one state per cell')
                candidate.append(states[0])
            if not check_solution(problem, candidate):
                raise RuntimeError('SAT result failed independent exact-board validation')
            budget.check()
            board = candidate
            outcome = 'solved'
    except _Cancelled as exc:
        outcome = exc.outcome
    except _Infeasible:
        # A late stop/timeout takes precedence over a preprocessing contradiction.
        outcome = budget.reason() or 'infeasible'
    finally:
        done.set()
        if monitor is not None:
            monitor.join()
        if build_seconds is None:
            build_seconds = time.monotonic() - budget.started
        if native is not None:
            stats = native.accum_stats() or {}
            native.delete()
    if budget.callback_error is not None:
        raise RuntimeError('SAT cancellation callback failed') from budget.callback_error
    return {'outcome': outcome, 'board': board, 'backend': backend,
            'branches': int(stats.get('decisions', 0)), 'conflicts': int(stats.get('conflicts', 0)),
            'elapsed_seconds': time.monotonic() - budget.started, 'build_seconds': build_seconds,
            'variables': next_var, 'clauses': clause_count, 'workers': 1,
            'requested_workers': workers, 'seed': seed, 'seed_policy': 'initial-placement-phases'}
