"""Bounded, offline score improvement of adjacent regions and donor pieces.

Only selected cells may change. The physical pieces currently occupying those
cells form the entire replacement pool; the complement and oriented clues stay
fixed. CP-SAT scores every grid edge exactly once, including region boundaries.
Failure of this restricted neighborhood says nothing about global feasibility.
"""
from __future__ import annotations

import itertools
import math
import random
import threading
import time

from exact_model import constraint_edges, frame_ok, grid_edges
from exact_optimize import evaluate_board

MAX_REGION_CELLS = 64


def _validate(problem, board, seed, should_stop):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
        raise ValueError('Connected-region repair requires the full grid')
    checked = evaluate_board(problem, board)
    if not checked['valid']:
        raise ValueError('Region parent violates pieces, domains, frame or fixed oriented clues')
    return checked


def select_region(problem, board, seed=0, region_size=12, donor_count=4, *, should_stop=None):
    """Choose a connected mutable core near mismatches and compatible donors.

    The core stays connected through mutable cells. Donors can be anywhere and
    must have a legal potential exchange with a core piece. Local scores rank
    choices only; they never remove legal alternatives from the repair model.
    Different seeds diversify ties and mismatch anchors. Cancellation returns
    no cells, allowing the caller to retain its current board.
    """
    _validate(problem, board, seed, should_stop)
    if type(region_size) is not int or not 1 <= region_size <= MAX_REGION_CELLS:
        raise ValueError('region_size must be an integer in 1..64')
    if type(donor_count) is not int or not 0 <= donor_count <= MAX_REGION_CELLS - region_size:
        raise ValueError('donor_count must be nonnegative and total cells at most 64')
    rng = random.Random(seed)
    mutable = set(range(problem.size**2)) - problem.fixed.keys()
    neighbors = {cell: set() for cell in mutable}
    incident = {cell: [] for cell in range(problem.size**2)}
    violations = dict.fromkeys(mutable, 0)
    for a, b, sa, sb in grid_edges(problem.size):
        incident[a].append((b, sa, sb))
        incident[b].append((a, sb, sa))
        if a in mutable and b in mutable:
            neighbors[a].add(b)
            neighbors[b].add(a)
        if problem.faces[board[a]][sa] != problem.faces[board[b]][sb]:
            if a in mutable:
                violations[a] += 1
            if b in mutable:
                violations[b] += 1
    def stopped():
        return bool(should_stop is not None and should_stop())
    anchors = sorted(cell for cell in mutable if violations[cell])
    if not anchors or stopped():
        return []
    # Weighted selection avoids repeatedly locking onto one worst edge.
    start = rng.choices(anchors, weights=[violations[c] for c in anchors], k=1)[0]
    core = {start}
    frontier = neighbors[start].copy()
    while frontier and len(core) < region_size:
        if stopped():
            return []
        tie = {cell: rng.random() for cell in sorted(frontier)}
        chosen = max(frontier, key=lambda cell: (violations[cell], len(neighbors[cell] & core), tie[cell]))
        core.add(chosen)
        frontier = (frontier | neighbors[chosen]) - core
    if not donor_count:
        return sorted(core)
    core_pieces = {board[c] // 4 for c in core}
    domains = {cell: set(problem.domains[cell]) for cell in mutable}
    def local_score(cell, state):
        return sum(problem.faces[state][side] == problem.faces[board[other]][other_side]
                   for other, side, other_side in incident[cell])
    current_scores = {cell: local_score(cell, board[cell]) for cell in mutable}
    candidates = []
    for donor in sorted(mutable - core):
        if stopped():
            return []
        piece = board[donor] // 4
        incoming = [local_score(cell, state) - current_scores[cell]
                    for cell in sorted(core) for state in range(piece * 4, piece * 4 + 4)
                    if state in domains[cell] and frame_ok(problem.size, cell, problem.faces[state])]
        if not incoming:
            continue
        outgoing = [local_score(donor, state) - current_scores[donor]
                    for physical in sorted(core_pieces) for state in range(physical * 4, physical * 4 + 4)
                    if state in domains[donor] and frame_ok(problem.size, donor, problem.faces[state])]
        if outgoing:
            candidates.append((max(incoming) + max(outgoing), violations[donor], rng.random(), donor))
    donors = [item[-1] for item in sorted(candidates, reverse=True)[:donor_count]]
    return sorted(core | set(donors))


class _Abort(Exception):
    pass


def region_repair(problem, board, cells, seconds=.5, seed=0, should_stop=None):
    """Seek parent score + 1 in a bounded compact CP-SAT neighborhood.

    The single CPU worker shares one deadline with model preparation. Every
    accepted witness is checked against the original full problem, frozen
    complement, physical piece pool and native objective. An empty, exhausted,
    cancelled or infeasible neighborhood returns the validated parent unless
    an improvement has already been independently checked.
    """
    started = time.monotonic()
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    checked = _validate(problem, board, seed, should_stop)
    reference = list(board)
    try:
        cells = list(itertools.islice(cells, MAX_REGION_CELLS + 1))
    except TypeError as exc:
        raise ValueError('cells must be an iterable of distinct cell indices') from exc
    if (len(cells) > MAX_REGION_CELLS or any(type(c) is not int or not 0 <= c < problem.size**2 for c in cells)
            or len(set(cells)) != len(cells)):
        raise ValueError('cells must contain at most 64 distinct valid indices')
    if set(cells) & problem.fixed.keys():
        raise ValueError('Fixed oriented clue cells cannot be released')
    cells = sorted(cells)
    selected = set(cells)
    pool = {reference[cell] // 4 for cell in cells}
    best_board, best_score = reference.copy(), checked['score']
    maximum = checked['max_score']
    target = min(maximum, best_score + 1)
    deadline = started + seconds
    phase = 'building'
    build_seconds = 0.0
    branches = conflicts = table_rows = changing_edges = fixed_score = 0
    solver_status = 'NOT_RUN'
    interrupted = threading.Event()
    expired = threading.Event()
    errors = []

    def abort_reason():
        if should_stop is not None and should_stop():
            interrupted.set()
        if interrupted.is_set():
            return 'stopped'
        if time.monotonic() >= deadline:
            expired.set()
        return 'timeout' if expired.is_set() else None

    def check():
        why = abort_reason()
        if why:
            raise _Abort(why)

    def result(termination):
        return {'board': best_board.copy(), 'score': best_score, 'reference_score': checked['score'],
                'target_score': target, 'max_score': maximum, 'improved': best_score > checked['score'],
                'outcome': 'solved' if best_score == maximum else 'improved' if best_score > checked['score'] else termination,
                'termination': termination, 'operator': 'connected-region', 'backend': 'cp-sat-region',
                'workers': 1, 'region_cells': cells.copy(), 'neighborhood_size': len(cells),
                'proof_scope': 'selected-cells-and-piece-pool-with-frozen-complement-only',
                'optimal': termination in ('optimal_in_region', 'no_improvement_in_region', 'already_perfect', 'perfect_board_validated'),
                'solver_status': solver_status, 'branches': int(branches), 'conflicts': int(conflicts),
                'table_rows': table_rows, 'changing_edges': changing_edges, 'fixed_score': fixed_score,
                'phase': phase, 'elapsed_seconds': time.monotonic() - started,
                'build_seconds': build_seconds or time.monotonic() - started}

    def accept(candidate, native_score):
        nonlocal best_board, best_score
        actual = evaluate_board(problem, candidate)
        if (not actual['valid'] or actual['score'] < target or actual['score'] != native_score
                or any(candidate[c] != reference[c] for c in range(len(reference)) if c not in selected)
                or {candidate[c] // 4 for c in cells} != pool):
            raise RuntimeError('Connected-region witness failed independent full-board validation')
        if actual['score'] > best_score:
            best_board, best_score = list(candidate), actual['score']

    try:
        check()
        if best_score == maximum:
            return result('already_perfect')
        if not cells:
            return result('empty_region')
        from ortools.sat.python import cp_model
        check()
        model = cp_model.CpModel()
        variables = {}
        for cell in cells:
            check()
            # At most four states for each of the selected physical pieces.
            allowed = set(problem.domains[cell])
            rows = [(state, state // 4, *problem.faces[state])
                    for piece in sorted(pool) for state in range(piece * 4, piece * 4 + 4)
                    if state in allowed and frame_ok(problem.size, cell, problem.faces[state])]
            if not rows or reference[cell] not in {row[0] for row in rows}:
                raise RuntimeError('Validated region parent was removed from the model')
            columns = [model.new_int_var_from_domain(cp_model.Domain.from_values(sorted({row[i] for row in rows})),
                                                    f'c{cell}_{i}') for i in range(6)]
            model.add_allowed_assignments(columns, rows)
            check()
            table_rows += len(rows)
            variables[cell] = columns
            for variable, value in zip(columns, (reference[cell], reference[cell] // 4, *problem.faces[reference[cell]])):
                model.add_hint(variable, value)
        model.add_all_different([variables[cell][1] for cell in cells])
        matches = []
        for index, (a, b, sa, sb) in enumerate(grid_edges(problem.size)):
            check()
            matched = int(problem.faces[reference[a]][sa] == problem.faces[reference[b]][sb])
            if a not in selected and b not in selected:
                fixed_score += matched
                continue
            left = variables[a][sa + 2] if a in selected else problem.faces[reference[a]][sa]
            right = variables[b][sb + 2] if b in selected else problem.faces[reference[b]][sb]
            match = model.new_bool_var(f'edge_{index}')
            model.add(left == right).only_enforce_if(match)
            model.add(left != right).only_enforce_if(match.Not())
            model.add_hint(match, matched)
            matches.append(match)
        changing_edges = len(matches)
        objective = fixed_score + sum(matches)
        model.add(objective >= target)
        model.maximize(objective)
        check()
        build_seconds = time.monotonic() - started
        phase = 'solving'
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed & 0x7fffffff
        solver.parameters.max_time_in_seconds = max(0., deadline - time.monotonic())
        check()
    except _Abort as exc:
        return result(str(exc))

    class Improvements(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            try:
                candidate = reference.copy()
                for cell in cells:
                    candidate[cell] = self.value(variables[cell][0])
                accept(candidate, int(round(self.objective_value)))
                if best_score == maximum or abort_reason():
                    self.stop_search()
            except BaseException as exc:
                errors.append(exc)
                interrupted.set()
                self.stop_search()

    finished = threading.Event()
    def watch():
        while not finished.wait(.01):
            try:
                if abort_reason():
                    # Repeat until native solve starts and observes cancellation.
                    solver.stop_search()
            except BaseException as exc:
                errors.append(exc)
                interrupted.set()
                solver.stop_search()
    watcher = threading.Thread(target=watch, name='eternity-region-cancel', daemon=True)
    watcher.start()
    try:
        status = solver.solve(model, Improvements())
    finally:
        finished.set()
        watcher.join()
    if errors:
        raise RuntimeError('Connected-region callback failed') from errors[0]
    branches, conflicts = solver.num_branches, solver.num_conflicts
    solver_status = solver.status_name(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        candidate = reference.copy()
        for cell in cells:
            candidate[cell] = solver.value(variables[cell][0])
        accept(candidate, int(round(solver.objective_value)))
    elif status == cp_model.MODEL_INVALID:
        raise RuntimeError('CP-SAT rejected the region model: ' + solver.response_stats())
    elif status not in (cp_model.UNKNOWN, cp_model.INFEASIBLE):
        raise RuntimeError('Unexpected connected-region status: ' + solver_status)
    reason = abort_reason()
    if best_score == maximum:
        return result('perfect_board_validated')
    if reason == 'stopped':
        return result('stopped')
    if status == cp_model.OPTIMAL and not reason:
        return result('optimal_in_region')
    if status == cp_model.INFEASIBLE and not reason:
        return result('no_improvement_in_region')
    return result('timeout')
