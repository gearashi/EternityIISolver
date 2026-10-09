"""Exact CP-SAT edge matching, with a wall budget shared by build and search.

OR-Tools is imported only when a nonzero-budget solve needs it. There is no
networking or GPU dependency. Cancellation is cooperative: callbacks should
return promptly, and an in-progress import/native model call cannot be
preempted by Python. Deadline checks immediately follow those operations.
"""
from __future__ import annotations

import math
import threading
import time


class BackendUnavailable(RuntimeError):
    pass


class _Aborted(Exception):
    def __init__(self, outcome, phase):
        self.outcome = outcome
        self.phase = phase


def solve(problem, seconds=60.0, seed=20261009, workers=1, should_stop=None, on_progress=None):
    """Find a witness or prove infeasibility for exactly the supplied problem.

    The selected constraint edges can be a subset of the full puzzle; solving
    such a problem does not establish a perfect full-board solution. Progress
    callbacks receive dictionaries and can run on the cancellation watcher
    thread during search. Native UNKNOWN is a timeout, never infeasibility.
    """
    started = time.monotonic()
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('workers must be an integer in 1..64')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    if on_progress is not None and not callable(on_progress):
        raise ValueError('on_progress must be callable')
    deadline = started + seconds
    build_seconds = 0.0
    table_rows = 0
    last_progress = float('-inf')
    phase = 'building'

    def result(outcome, board=None, branches=0, conflicts=0, solver_status='NOT_RUN', reason=None):
        elapsed = time.monotonic() - started
        return {'outcome': outcome, 'board': board, 'branches': int(branches), 'conflicts': int(conflicts),
                'elapsed_seconds': elapsed, 'build_seconds': build_seconds or elapsed,
                'backend': 'cp-sat', 'solver_status': solver_status, 'reason': reason,
                'table_rows': table_rows, 'workers': workers}

    def check_budget():
        if should_stop is not None and should_stop():
            raise _Aborted('stopped', phase)
        if time.monotonic() >= deadline:
            raise _Aborted('timeout', phase)

    def progress(force=False):
        nonlocal last_progress
        now = time.monotonic()
        if on_progress is not None and (force or now - last_progress >= .25):
            last_progress = now
            on_progress({'phase': phase, 'elapsed_seconds': now - started,
                         'build_seconds': build_seconds or now - started,
                         'branches': None, 'conflicts': None, 'backend': 'cp-sat'})

    try:
        check_budget()
        progress(force=True)
        check_budget()
        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:
            raise BackendUnavailable('Exact CP-SAT search requires the optional ortools dependency') from exc
        check_budget()
        from exact_model import check_solution, constraint_edges
        n = problem.size
        if type(n) is not int or not 1 <= n <= 16:
            raise ValueError('Exact puzzle size must be an integer in 1..16')
        cells = n * n
        if len(problem.faces) != 4 * cells or len(problem.domains) != cells:
            raise ValueError('Incorrect number of oriented faces or cell domains')
        for index, faces in enumerate(problem.faces):
            if len(faces) != 4 or any(type(color) is not int or not 0 <= color <= 2**31 - 1 for color in faces):
                raise ValueError('Faces must contain four nonnegative int32 colors per state')
            if index % 64 == 0:
                check_budget()
        for cell, state in problem.fixed.items():
            if type(cell) is not int or not 0 <= cell < cells or type(state) is not int or not 0 <= state < 4 * cells:
                raise ValueError('Invalid fixed placement')

        model = cp_model.CpModel()
        variables = []
        state_maps = []
        for cell, domain in enumerate(problem.domains):
            check_budget()
            row, column = divmod(cell, n)
            outside = (row == 0, column == n - 1, row == n - 1, column == 0)
            entries = {}
            for offset, state in enumerate(domain):
                if type(state) is not int or not 0 <= state < 4 * cells:
                    raise ValueError('Invalid state in a cell domain')
                if offset % 64 == 0:
                    check_budget()
                if cell in problem.fixed and state != problem.fixed[cell]:
                    continue
                faces = problem.faces[state]
                if any((color == 0) != exterior for color, exterior in zip(faces, outside)):
                    continue
                # A table tuple represents an allowed physical piece and its
                # four oriented colors. Symmetric rotations may share a tuple;
                # picking any matching allowed state preserves all constraints.
                entries.setdefault((state // 4, *faces), state)
            if not entries:
                return result('infeasible', reason=f'Cell {cell} has no legal placement')
            rows = list(entries)
            table_rows += len(rows)
            cols = [model.new_int_var_from_domain(cp_model.Domain.from_values(sorted({entry[k] for entry in rows})),
                                                  f'c{cell}_{k}') for k in range(5)]
            model.add_allowed_assignments(cols, rows)
            variables.append(cols)
            state_maps.append(entries)
            check_budget()
            progress()
        model.add_all_different([cols[0] for cols in variables])
        check_budget()
        for a, b, side_a, side_b in constraint_edges(problem):
            if (any(type(value) is not int for value in (a, b, side_a, side_b)) or
                    not 0 <= a < cells or not 0 <= b < cells or not 0 <= side_a < 4 or not 0 <= side_b < 4):
                raise ValueError('Invalid constrained edge')
            model.add(variables[a][side_a + 1] == variables[b][side_b + 1])
            check_budget()
        progress()
        check_budget()
        build_seconds = time.monotonic() - started
        phase = 'solving'
        progress(force=True)
        check_budget()
    except _Aborted as aborted:
        return result(aborted.outcome, reason=f'{aborted.outcome} during {aborted.phase}')

    solver = cp_model.CpSolver()
    # Subtract all construction/import time; never give search a second full
    # budget. uint32 input seeds are mapped into CP-SAT's nonnegative int32.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return result('timeout', reason='Time budget consumed by model construction')
    solver.parameters.max_time_in_seconds = remaining
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed & 0x7FFFFFFF
    finished = threading.Event()
    interrupted = threading.Event()
    timed_out = threading.Event()
    callback_errors = []

    def watch():
        while not finished.wait(.02):
            try:
                if should_stop is not None and should_stop():
                    interrupted.set()
                if time.monotonic() >= deadline:
                    timed_out.set()
                if interrupted.is_set() or timed_out.is_set():
                    # Repeat until solve exits: a request racing native solve
                    # startup must not be lost by a single early stop_search.
                    solver.stop_search()
                else:
                    progress()
            except Exception as exc:
                callback_errors.append(exc)
                interrupted.set()
                solver.stop_search()

    watcher = threading.Thread(target=watch, name='eternity-cp-cancel', daemon=True)
    watcher.start()
    try:
        status = solver.solve(model)
    finally:
        finished.set()
        watcher.join()
    if callback_errors:
        raise callback_errors[0]
    branches, conflicts = solver.num_branches, solver.num_conflicts
    status_name = solver.status_name(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        board = [state_maps[cell][tuple(solver.value(var) for var in cols)] for cell, cols in enumerate(variables)]
        if not check_solution(problem, board):
            raise RuntimeError('CP-SAT witness failed independent solution validation')
        answer = result('solved', board, branches, conflicts, status_name)
    elif status == cp_model.INFEASIBLE:
        answer = result('infeasible', branches=branches, conflicts=conflicts, solver_status=status_name)
    elif status == cp_model.MODEL_INVALID:
        raise RuntimeError('CP-SAT rejected the constructed model: ' + solver.response_stats())
    elif status == cp_model.UNKNOWN:
        answer = result('stopped' if interrupted.is_set() else 'timeout', branches=branches,
                        conflicts=conflicts, solver_status=status_name,
                        reason='Cancellation requested' if interrupted.is_set() else 'No proof or solution before the search ended')
    else:
        raise RuntimeError(f'Unexpected CP-SAT status: {status_name}')
    if on_progress is not None:
        on_progress({'phase': 'finished', **answer})
    return answer
