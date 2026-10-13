"""Bounded offline full-board score optimization and assumption-core repair.

All frame rules, physical-piece identities, domains and fixed oriented clues
remain hard. Optimization reifies *every* grid edge in both directions. The
reference supplies hints and a validated fallback, never frozen placements.
Assumption repair freezes only explicit temporary assumptions and expands
its neighborhood using genuine SAT UNSAT cores, never a timeout.
"""
from __future__ import annotations

import math
import threading
import time

from exact_model import constraint_edges, frame_ok, grid_edges


class _Aborted(Exception):
    def __init__(self, outcome):
        self.outcome = outcome


def evaluate_board(problem, board):
    """Independently check structural legality and count all grid-edge matches.

    Unlike check_solution, a legal mismatched board is accepted here. Domain,
    exact fixed-state, frame and physical-piece checks remain mandatory.
    """
    try:
        count = problem.size ** 2
        if len(board) != count or any(type(state) is not int or not 0 <= state < len(problem.faces) for state in board):
            return {'valid': False, 'score': None}
        if len({state // 4 for state in board}) != count:
            return {'valid': False, 'score': None}
        if any(state not in problem.domains[cell] or not frame_ok(problem.size, cell, problem.faces[state])
               for cell, state in enumerate(board)):
            return {'valid': False, 'score': None}
        if any(board[cell] != state for cell, state in problem.fixed.items()):
            return {'valid': False, 'score': None}
        edges = grid_edges(problem.size)
        score = sum(problem.faces[board[a]][sa] == problem.faces[board[b]][sb] for a, b, sa, sb in edges)
        return {'valid': True, 'score': score, 'max_score': len(edges)}
    except (TypeError, KeyError, IndexError, AttributeError):
        return {'valid': False, 'score': None}


def _inputs(problem, reference, seconds, seed, workers, should_stop, on_improvement, on_progress):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('workers must be an integer in 1..64')
    for name, callback in (('should_stop', should_stop), ('on_improvement', on_improvement), ('on_progress', on_progress)):
        if callback is not None and not callable(callback):
            raise ValueError(name + ' must be callable')
    if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
        raise ValueError('Full-board optimization requires all grid edges; regional tasks are not accepted')
    checked = evaluate_board(problem, reference)
    if not checked['valid']:
        raise ValueError('Reference board violates the frame, domains, unique pieces or fixed oriented clues')
    return checked['score'], checked['max_score']


def optimize(problem, reference, seconds=30.0, seed=0, workers=1, should_stop=None,
             on_improvement=None, on_progress=None, *, target_score=None, regional_bounds=True):
    """Maximize the full-board score within one build-and-search time budget.

    By default require at least reference_score+1 matches. A failed target
    proves only that score target infeasible for the supplied domains/fixed
    placements. Every return retains an independently checked reference or
    better board. on_improvement receives a dictionary with board and score.
    Callbacks must be short and thread-safe. Stop requests take precedence
    except that an already validated perfect board can be returned as solved.
    """
    started = time.monotonic()
    reference_score, maximum = _inputs(problem, reference, seconds, seed, workers,
                                       should_stop, on_improvement, on_progress)
    if target_score is None:
        target_score = min(maximum, reference_score + 1)
    if type(target_score) is not int or not reference_score <= target_score <= maximum:
        raise ValueError('target_score must be an integer from reference_score to max_score')
    if type(regional_bounds) is not bool:
        raise ValueError("regional_bounds must be boolean")
    mismatch_bound = None
    deadline = started + seconds
    best_board, best_score = list(reference), reference_score
    backend = 'cp-sat-optimize'
    phase = 'building'
    build_seconds = 0.0
    table_rows = 0
    branches = conflicts = 0
    solver_status = 'NOT_RUN'
    last_progress = float('-inf')
    reason = None
    interrupted = threading.Event()
    timed_out = threading.Event()
    callback_errors = []
    best_lock = threading.Lock()

    def abort_reason():
        if should_stop is not None and should_stop():
            interrupted.set()
        if interrupted.is_set():
            return 'stopped'
        if time.monotonic() >= deadline:
            timed_out.set()
        return 'timeout' if timed_out.is_set() else None

    def check():
        current = abort_reason()
        if current:
            raise _Aborted(current)

    def result(outcome, termination, why=None):
        return {'outcome': outcome, 'termination': termination, 'board': list(best_board),
                'score': best_score, 'reference_score': reference_score, 'target_score': target_score,
                'max_score': maximum, 'improved': best_score > reference_score,
                'backend': backend, 'solver_status': solver_status, 'workers': workers,
                'branches': int(branches), 'conflicts': int(conflicts), 'table_rows': table_rows,
                'elapsed_seconds': time.monotonic() - started,
                'build_seconds': build_seconds or time.monotonic() - started, 'reason': why,
                'mismatch_lower_bound': None if mismatch_bound is None else mismatch_bound['lower_bound'],
                'proof_scope': 'supplied-domains-fixed-placements-and-full-grid-score-target'}

    def progress(force=False):
        nonlocal last_progress
        now = time.monotonic()
        if on_progress is not None and (force or now - last_progress >= .25):
            last_progress = now
            with best_lock:
                score = best_score
            on_progress({'phase': phase, 'backend': backend, 'elapsed_seconds': now - started,
                         'build_seconds': build_seconds or now - started,
                         'score': score, 'target_score': target_score, 'branches': None, 'conflicts': None})

    def accept(candidate, candidate_branches=0, candidate_conflicts=0, native_score=None):
        nonlocal best_board, best_score
        checked = evaluate_board(problem, candidate)
        if not checked['valid'] or checked['score'] < target_score:
            raise RuntimeError('CP-SAT optimization witness failed independent structural/target validation')
        score = checked['score']
        if native_score is not None and score != native_score:
            raise RuntimeError('CP-SAT objective disagrees with independent full-board score')
        with best_lock:
            if score <= best_score:
                return
            best_board, best_score = list(candidate), score
        if on_improvement is not None:
            on_improvement({'board': list(candidate), 'score': score, 'backend': backend,
                            'branches': int(candidate_branches), 'conflicts': int(candidate_conflicts),
                            'elapsed_seconds': time.monotonic() - started})

    try:
        check()
        if best_score == maximum:
            return result('solved', 'reference_already_perfect')
        progress(True)
        check()
        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:
            raise RuntimeError('Score optimization requires the optional ortools dependency') from exc
        check()
        # Broad full-puzzle domains make this relaxation weak and expensive.
        # Native CP-SAT retains its own objective bounds on those tasks.
        if regional_bounds and sum(map(len, problem.domains)) <= 4096:
            from mismatch_bounds import mismatch_lower_bound
            mismatch_bound = mismatch_lower_bound(problem, check=check)
            if mismatch_bound['structurally_infeasible']:
                raise RuntimeError('Validated reference lost by mismatch relaxation')
            if maximum - mismatch_bound['lower_bound'] < target_score:
                return result('infeasible_for_target', 'admissible_mismatch_bound')
        model = cp_model.CpModel()
        variables = []
        fixed_pieces = {state // 4 for state in problem.fixed.values()}
        for cell, domain in enumerate(problem.domains):
            check()
            rows = []
            seen = set()
            for index, state in enumerate(domain):
                if index % 64 == 0:
                    check()
                if state in seen:
                    continue
                seen.add(state)
                if cell in problem.fixed:
                    if state != problem.fixed[cell]:
                        continue
                elif state // 4 in fixed_pieces:
                    continue
                if not frame_ok(problem.size, cell, problem.faces[state]):
                    continue
                rows.append((state, state // 4, *problem.faces[state]))
            # Reference validation guarantees at least one structurally legal
            # candidate in each domain; still fail explicitly if that invariant
            # changes in a future domain preparer.
            if not rows:
                raise RuntimeError('Validated reference was removed from an optimizer cell domain')
            columns = []
            for index in range(6):
                values = sorted({row[index] for row in rows})
                columns.append(model.new_int_var_from_domain(cp_model.Domain.from_values(values), f'c{cell}_{index}'))
                check()
            model.add_allowed_assignments(columns, rows)
            check()
            table_rows += len(rows)
            variables.append(columns)
            for variable, value in zip(columns, (reference[cell], reference[cell] // 4, *problem.faces[reference[cell]])):
                model.add_hint(variable, value)
            progress()
        model.add_all_different([columns[1] for columns in variables])
        check()
        matches = []
        for index, (a, b, side_a, side_b) in enumerate(grid_edges(problem.size)):
            check()
            match = model.new_bool_var(f'match_{index}')
            left, right = variables[a][side_a + 2], variables[b][side_b + 2]
            model.add(left == right).only_enforce_if(match)
            model.add(left != right).only_enforce_if(match.Not())
            model.add_hint(match, int(problem.faces[reference[a]][side_a] == problem.faces[reference[b]][side_b]))
            matches.append(match)
        if mismatch_bound is not None:
            for group in mismatch_bound['groups']:
                check()
                if group['minimum_mismatches']:
                    model.add(sum(matches[i] for i in group['edges']) <=
                              len(group['edges']) - group['minimum_mismatches'])
        model.add(sum(matches) >= target_score)
        model.maximize(sum(matches))
        check()
        build_seconds = time.monotonic() - started
        phase = 'solving'
        progress(True)
        check()
    except _Aborted as aborted:
        return result(aborted.outcome, aborted.outcome, f'{aborted.outcome} during {phase}')

    try:
        check()
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(0.0, deadline - time.monotonic())
        solver.parameters.num_search_workers = workers
        solver.parameters.random_seed = seed & 0x7FFFFFFF
        check()
    except _Aborted as aborted:
        return result(aborted.outcome, aborted.outcome, 'Budget exhausted before native search')
    finished = threading.Event()

    class Improvements(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            try:
                candidate = [self.value(columns[0]) for columns in variables]
                accept(candidate, self.num_branches, self.num_conflicts, int(round(self.objective_value)))
                if best_score == maximum or abort_reason():
                    self.stop_search()
            except BaseException as exc:
                callback_errors.append(exc)
                interrupted.set()
                self.stop_search()

    def watch():
        while not finished.wait(.02):
            try:
                if abort_reason():
                    # Repeat through the native-start race, not just once.
                    solver.stop_search()
                else:
                    progress()
            except BaseException as exc:
                callback_errors.append(exc)
                interrupted.set()
                solver.stop_search()

    watcher = threading.Thread(target=watch, name='eternity-optimize-cancel', daemon=True)
    watcher.start()
    try:
        status = solver.solve(model, Improvements())
    finally:
        finished.set()
        watcher.join()
    if callback_errors:
        raise RuntimeError('Score optimization callback failed') from callback_errors[0]
    branches, conflicts = solver.num_branches, solver.num_conflicts
    solver_status = solver.status_name(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        candidate = [solver.value(columns[0]) for columns in variables]
        accept(candidate, branches, conflicts, int(round(solver.objective_value)))
    elif status == cp_model.MODEL_INVALID:
        raise RuntimeError('CP-SAT rejected the score model: ' + solver.response_stats())
    elif status not in (cp_model.UNKNOWN, cp_model.INFEASIBLE):
        raise RuntimeError('Unexpected CP-SAT optimization status: ' + solver_status)
    stopped = abort_reason()
    if best_score == maximum:
        answer = result('solved', 'perfect_board_validated')
    elif stopped == 'stopped':
        answer = result('stopped', 'stopped', 'Cancellation requested; best validated board retained')
    elif best_score > reference_score:
        termination = 'optimal_for_supplied_problem' if status == cp_model.OPTIMAL else 'timeout'
        answer = result('improved', termination)
    elif status == cp_model.INFEASIBLE and not stopped:
        answer = result('infeasible_for_target', 'infeasible_for_target',
                        'No board reaches the requested score within the supplied domains and fixed placements')
    elif status == cp_model.OPTIMAL and not stopped:
        answer = result('unchanged', 'optimal_for_supplied_problem')
    else:
        answer = result('timeout', 'timeout', 'No improvement before the bounded search ended')
    if on_progress is not None:
        on_progress({'phase': 'finished', **answer})
    return answer


def core_guided_repair(problem, reference, seconds=30.0, seed=0, workers=1, should_stop=None,
                       on_improvement=None, on_progress=None, *, free_cells=None, max_rounds=None,
                       reverse_support=True):
    """Expand an exact repair neighborhood using SAT assumption UNSAT cores.

    All grid edges are always hard, even between frozen cells. Fixed clues are
    never assumptions and are never released. A core only says at least one
    of those *temporary* freezes must be released; it does not blame every
    member or prove the full puzzle impossible. Native timeout never creates
    a core. One incremental SAT model and its learned clauses are reused
    across rounds under a single build-and-search deadline.
    """
    started = time.monotonic()
    baseline, maximum = _inputs(problem, reference, seconds, seed, workers,
                                should_stop, on_improvement, on_progress)
    deadline = started + seconds
    count = problem.size ** 2
    if max_rounds is None:
        max_rounds = count + 1
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError('max_rounds must be a positive integer')
    if type(reverse_support) is not bool:
        raise ValueError('reverse_support must be a boolean')
    if free_cells is None:
        free = {cell for a, b, sa, sb in grid_edges(problem.size)
                if problem.faces[reference[a]][sa] != problem.faces[reference[b]][sb]
                for cell in (a, b) if cell not in problem.fixed}
    else:
        free = set(free_cells)
        if any(type(cell) is not int or not 0 <= cell < count for cell in free):
            raise ValueError('Invalid free cell')
        free.difference_update(problem.fixed)
    best = list(reference)
    score = baseline
    cores = []
    rounds = branches = conflicts = model_builds = 0
    build_seconds = 0.0
    final_native_status = None
    edges = grid_edges(problem.size)

    def result(outcome, termination, why=None):
        return {'outcome': outcome, 'termination': termination, 'board': list(best), 'score': score,
                'reference_score': baseline, 'target_score': maximum, 'max_score': maximum,
                'improved': score > baseline, 'backend': 'sat-core-repair', 'workers': 1,
                'requested_workers': workers, 'rounds': rounds, 'branches': branches, 'conflicts': conflicts,
                'incremental_sat': True, 'model_builds': model_builds, 'assumption_queries': rounds,
                'elapsed_seconds': time.monotonic() - started, 'build_seconds': build_seconds,
                'free_cells': sorted(free), 'cores': [list(core) for core in cores],
                'solver_status': final_native_status, 'reason': why,
                'proof_scope': 'supplied-domains-fixed-placements-and-full-grid-exact-target'}

    def stopped():
        return should_stop is not None and should_stop()

    if stopped():
        return result('stopped', 'stopped')
    if time.monotonic() >= deadline:
        return result('timeout', 'timeout')
    if baseline == maximum:
        return result('solved', 'reference_already_perfect')
    import exact_sat

    def progress(update):
        if on_progress is not None:
            on_progress({**update, 'backend': 'sat-core-repair', 'rounds': rounds,
                         'free_cells': len(free), 'cores': len(cores),
                         'elapsed_seconds': time.monotonic() - started})

    # One CNF and one native learned-clause database for the whole repair.
    # Per-query assumption selectors are temporary; hard clues never change.
    with exact_sat.AssumptionSession(problem, seconds=max(0.0, deadline - time.monotonic()),
                                     seed=seed, workers=workers, should_stop=should_stop,
                                     on_progress=progress, reverse_support=reverse_support) as sat:
        for round_index in range(max_rounds):
            if stopped():
                return result('stopped', 'stopped')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return result('timeout', 'timeout')
            assumptions = {cell: state for cell, state in enumerate(reference)
                           if cell not in free and cell not in problem.fixed}
            rounds += 1
            answer = sat.solve(assumptions, seconds=remaining)
            branches += answer['branches']
            conflicts += answer['conflicts']
            build_seconds += answer['build_seconds']
            model_builds = answer['model_builds']
            final_native_status = answer['outcome']
            if answer['outcome'] == 'solved':
                checked = evaluate_board(problem, answer['board'])
                if not checked['valid'] or checked['score'] != maximum:
                    raise RuntimeError('Core-guided repair failed independent full-board validation')
                best, score = list(answer['board']), checked['score']
                if score > baseline and on_improvement is not None:
                    on_improvement({'board': list(best), 'score': score, 'backend': 'sat-core-repair',
                                    'branches': branches, 'conflicts': conflicts,
                                    'elapsed_seconds': time.monotonic() - started})
                return result('solved', 'perfect_board_validated')
            if answer['outcome'] in ('timeout', 'stopped'):
                return result(answer['outcome'], answer['outcome'])
            if answer['outcome'] != 'infeasible' or answer['core_cells'] is None:
                raise RuntimeError('SAT assumption repair returned neither a witness nor a checked UNSAT core')
            core = tuple(answer['core_cells'])
            if any(cell not in assumptions for cell in core):
                raise RuntimeError('SAT core contains a placement that was not temporarily frozen')
            if not core:
                return result('infeasible_for_target', 'infeasible_for_target',
                              'The supplied full-grid exact model is infeasible independently of temporary freezes')
            cores.append(core)
            # A core says at least one freeze must change, not that every core
            # member is wrong. Release one; all alternatives remain in the CNF.
            def priority(cell):
                recurrence = sum(cell in previous for previous in cores)
                adjacent = sum((a == cell and b in free) or (b == cell and a in free) for a, b, _, _ in edges)
                return recurrence, adjacent, -cell
            free.add(max(core, key=priority))
    return result('timeout', 'round_limit', 'Repair round budget reached; no additional infeasibility claim')
