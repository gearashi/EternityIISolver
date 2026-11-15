"""Offline multi-parent repair and assumption-core neighborhood guidance.

The recombination model explores only states observed in its parents. Its
infeasibility is a neighborhood fact, not a claim about the full puzzle.
Square support clauses are redundant exact constraints: an incomplete bounded
enumeration is discarded instead of being used to prune any placement.
"""
from __future__ import annotations

from collections import Counter
import itertools
import math
import random
import threading
import time

from exact_model import check_solution, constraint_edges, frame_ok, grid_edges
from exact_optimize import evaluate_board
from exact_sat import AssumptionSession


def _duration(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def _cells(problem, cells, maximum=64):
    try:
        cells = list(itertools.islice(cells, maximum + 1))
    except TypeError as exc:
        raise ValueError('cells must be an iterable of distinct valid indices') from exc
    if (len(cells) > maximum or len(set(cells)) != len(cells)
            or any(type(c) is not int or not 0 <= c < problem.size**2 for c in cells)):
        raise ValueError(f'cells must contain at most {maximum} distinct valid indices')
    if set(cells) & problem.fixed.keys():
        raise ValueError('Fixed oriented clues cannot be released')
    return sorted(cells)


def _full_grid(problem):
    if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
        raise ValueError('This operation requires the full grid')


def multi_parent_recombine(problem, parents, *, cells=None, seconds=.5,
                           deterministic_budget=.05, seed=0, should_stop=None,
                           first_improvement=True, max_cells=32, lift_rotations=False):
    """Seek a strict true-score improvement over the best of 3..8 parents.

    By default, each selected cell uses states seen there in an input parent and
    physical pieces belonging to the selected part of the best parent. The
    complement stays frozen. CP-SAT enforces all-different pieces and scores
    every shared edge exactly once. The deterministic limit is native CP-SAT
    work, not a promise about a number of puzzle moves. Branches and conflicts
    are reported separately. The default stops at the first verified gain.
    """
    started = time.monotonic()
    seconds = _duration(seconds, 'seconds')
    deterministic_budget = _duration(deterministic_budget, 'deterministic_budget')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    if type(lift_rotations) is not bool:
        raise ValueError('lift_rotations must be a boolean')
    if type(first_improvement) is not bool:
        raise ValueError('first_improvement must be a boolean')
    if type(max_cells) is not int or not 1 <= max_cells <= 64:
        raise ValueError('max_cells must be an integer in 1..64')
    _full_grid(problem)
    try:
        parents = [list(board) for board in itertools.islice(parents, 9)]
    except TypeError as exc:
        raise ValueError('parents must contain 3..8 complete boards') from exc
    if not 3 <= len(parents) <= 8:
        raise ValueError('parents must contain 3..8 complete boards')
    checked = [evaluate_board(problem, board) for board in parents]
    if not all(item['valid'] for item in checked):
        raise ValueError('A parent violates pieces, domains, frame or fixed oriented clues')
    reference_index = max(range(len(parents)), key=lambda i: checked[i]['score'])
    reference = parents[reference_index].copy()
    reference_score = checked[reference_index]['score']
    maximum = checked[reference_index]['max_score']
    if cells is None:
        candidates = [c for c in range(len(reference)) if c not in problem.fixed
                      and len({parent[c] for parent in parents}) > 1]
        violations = Counter()
        for parent in parents:
            for a, b, sa, sb in grid_edges(problem.size):
                if problem.faces[parent[a]][sa] != problem.faces[parent[b]][sb]:
                    violations[a] += 1
                    violations[b] += 1
        rng = random.Random(seed)
        ties = {c: rng.random() for c in candidates}
        cells = sorted(sorted(candidates, key=lambda c: (-violations[c], ties[c]))[:max_cells])
    cells = _cells(problem, cells)
    selected = set(cells)
    pool = {reference[c] // 4 for c in cells}
    allowed_states = {c: sorted({parent[c] for parent in parents if parent[c] // 4 in pool}) for c in cells}
    model_parent = reference
    lift_work = None
    best_board, best_score = reference.copy(), reference_score
    target = min(maximum, reference_score + 1)
    deadline = started + seconds
    branches = conflicts = table_rows = changing_edges = fixed_score = 0
    deterministic_work = 0.0
    build_seconds = 0.0
    solver_status = 'NOT_RUN'
    interrupted = threading.Event()
    errors = []

    def reason():
        if should_stop is not None and should_stop():
            interrupted.set()
        if interrupted.is_set():
            return 'stopped'
        return 'timeout' if time.monotonic() >= deadline else None

    def result(termination):
        return {'board': best_board.copy(), 'score': best_score, 'reference_score': reference_score,
                'target_score': target, 'max_score': maximum, 'improved': best_score > reference_score,
                'outcome': 'solved' if best_score == maximum else 'improved' if best_score > reference_score else termination,
                'termination': termination, 'operator': 'multi-parent', 'backend': 'cp-sat-parent-union',
                'parent_count': len(parents), 'distinct_parents': len({tuple(p) for p in parents}),
                'reference_parent': reference_index, 'region_cells': cells.copy(), 'workers': 1,
                'branches': int(branches), 'conflicts': int(conflicts), 'table_rows': table_rows,
                'changing_edges': changing_edges, 'deterministic_work': deterministic_work,
                'deterministic_budget': deterministic_budget, 'solver_status': solver_status,
                'optimal': termination in ('already_perfect', 'perfect_board_validated', 'optimal_in_union', 'no_improvement_in_union'),
                'lift_work': lift_work,
                'proof_scope': ('boundary-dominating-rotations-of-parent-states-with-frozen-complement' if lift_rotations
                                else 'observed-parent-states-in-selected-piece-pool-with-frozen-complement-only'),
                'elapsed_seconds': time.monotonic() - started, 'build_seconds': build_seconds}

    def accept(candidate, native_score):
        nonlocal best_board, best_score
        actual = evaluate_board(problem, candidate)
        if (not actual['valid'] or actual['score'] < target or actual['score'] != native_score
                or any(candidate[c] != reference[c] for c in range(len(reference)) if c not in selected)
                or any(candidate[c] not in allowed_states[c] for c in cells)
                or {candidate[c] // 4 for c in cells} != pool):
            raise RuntimeError('Parent-union witness failed independent full-board validation')
        if actual['score'] > best_score:
            best_board, best_score = list(candidate), actual['score']

    why = reason()
    if why:
        return result(why)
    if best_score == maximum:
        return result('already_perfect')
    if not cells:
        return result('empty_region')
    if deterministic_budget == 0:
        return result('work_limit')
    if lift_rotations:
        from repair_rotations import lift_rows
        revised, model_parent, lift_work = lift_rows(problem, reference, cells, allowed_states, stop=reason)
        if revised is None:
            return result(reason() or 'stopped')
        allowed_states = revised
        warm = evaluate_board(problem, model_parent)
        if not warm['valid'] or warm['score'] < reference_score:
            raise RuntimeError('Lifted reference violates score dominance')
        if warm['score'] >= target:
            accept(model_parent, warm['score'])
            if best_score == maximum or first_improvement:
                return result('perfect_board_validated' if best_score == maximum else 'first_improvement')
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    variables = {}
    for cell in cells:
        why = reason()
        if why:
            return result(why)
        rows = [(state, state // 4, *problem.faces[state]) for state in allowed_states[cell]
                if state in problem.domains[cell] and frame_ok(problem.size, cell, problem.faces[state])]
        if model_parent[cell] not in {row[0] for row in rows}:
            raise RuntimeError('Validated reference disappeared from parent state union')
        columns = [model.new_int_var_from_domain(cp_model.Domain.from_values(sorted({row[i] for row in rows})),
                                                f'c{cell}_{i}') for i in range(6)]
        model.add_allowed_assignments(columns, rows)
        table_rows += len(rows)
        variables[cell] = columns
        for variable, value in zip(columns, (model_parent[cell], model_parent[cell] // 4, *problem.faces[model_parent[cell]])):
            model.add_hint(variable, value)
    model.add_all_different([variables[c][1] for c in cells])
    matches = []
    for index, (a, b, sa, sb) in enumerate(grid_edges(problem.size)):
        why = reason()
        if why:
            return result(why)
        matched = int(problem.faces[model_parent[a]][sa] == problem.faces[model_parent[b]][sb])
        if a not in selected and b not in selected:
            fixed_score += matched
            continue
        left = variables[a][sa + 2] if a in selected else problem.faces[reference[a]][sa]
        right = variables[b][sb + 2] if b in selected else problem.faces[reference[b]][sb]
        match = model.new_bool_var(f'e{index}')
        model.add(left == right).only_enforce_if(match)
        model.add(left != right).only_enforce_if(match.Not())
        model.add_hint(match, matched)
        matches.append(match)
    changing_edges = len(matches)
    objective = fixed_score + sum(matches)
    model.add(objective >= target)
    model.maximize(objective)
    build_seconds = time.monotonic() - started
    why = reason()
    if why:
        return result(why)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed & 0x7fffffff
    solver.parameters.max_time_in_seconds = max(0., deadline - time.monotonic())
    solver.parameters.max_deterministic_time = deterministic_budget

    class Improvements(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            try:
                candidate = reference.copy()
                for cell in cells:
                    candidate[cell] = self.value(variables[cell][0])
                accept(candidate, int(round(self.objective_value)))
                if first_improvement or best_score == maximum or reason():
                    self.stop_search()
            except BaseException as exc:
                errors.append(exc)
                interrupted.set()
                self.stop_search()

    finished = threading.Event()
    def watch():
        while not finished.wait(.01):
            try:
                if reason():
                    solver.stop_search()
            except BaseException as exc:
                errors.append(exc)
                interrupted.set()
                solver.stop_search()
    watcher = threading.Thread(target=watch, name='eternity-parent-union-cancel', daemon=True)
    watcher.start()
    try:
        status = solver.solve(model, Improvements())
    finally:
        finished.set()
        watcher.join()
    if errors:
        raise RuntimeError('Parent-union callback failed') from errors[0]
    branches, conflicts = solver.num_branches, solver.num_conflicts
    deterministic_work = float(solver.response_proto.deterministic_time)
    solver_status = solver.status_name(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        candidate = reference.copy()
        for cell in cells:
            candidate[cell] = solver.value(variables[cell][0])
        accept(candidate, int(round(solver.objective_value)))
    elif status == cp_model.MODEL_INVALID:
        raise RuntimeError('CP-SAT rejected parent-union model: ' + solver.response_stats())
    elif status not in (cp_model.UNKNOWN, cp_model.INFEASIBLE):
        raise RuntimeError('Unexpected parent-union status: ' + solver_status)
    why = reason()
    if best_score == maximum:
        return result('perfect_board_validated')
    if why:
        return result(why)
    if status == cp_model.OPTIMAL:
        return result('optimal_in_union')
    if status == cp_model.INFEASIBLE:
        return result('no_improvement_in_union')
    return result('first_improvement' if best_score > reference_score and first_improvement else 'work_limit')


class SquareSupportSession(AssumptionSession):
    """Reusable SAT with complete 2x2 tuple supports, when cheap to enumerate.

    Shared edge channels alone do not express four-cell consistency in a single
    support clause. These redundant tuple selectors do. A patch is encoded only
    after ALL its compatible distinct-piece tuples have been enumerated; a cap
    hit skips the patch completely. Budgets measure attempted state extensions.
    """
    def __init__(self, *args, square_work_budget=50000, square_tuple_cap=2048, **kwargs):
        for name, value in (('square_work_budget', square_work_budget), ('square_tuple_cap', square_tuple_cap)):
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        self.square_work_budget = square_work_budget
        self.square_tuple_cap = square_tuple_cap
        self.square_work = self.square_encoded = self.square_skipped = self.square_tuples = 0
        super().__init__(*args, **kwargs)

    def _build(self):
        self.square_work = self.square_encoded = self.square_skipped = self.square_tuples = 0
        super()._build()
        # A cancellation in this extension must discard the partial model too.
        self._built = False
        size, faces = self.problem.size, self.problem.faces
        edges = set(constraint_edges(self.problem))
        for row in range(size - 1):
            for col in range(size - 1):
                self.check()
                a = row * size + col
                cells = (a, a + 1, a + size, a + size + 1)
                internal = ((cells[0], cells[1], 1, 3), (cells[0], cells[2], 2, 0),
                            (cells[1], cells[3], 2, 0), (cells[2], cells[3], 1, 3))
                if not set(internal) <= edges:
                    continue
                if self.square_work >= self.square_work_budget or self.square_tuple_cap == 0:
                    self.square_skipped += 1
                    continue
                tuples = []
                exhausted = False
                def visit(chosen, pieces):
                    nonlocal exhausted
                    if len(chosen) == 4:
                        if len(tuples) >= self.square_tuple_cap:
                            exhausted = True
                            return
                        tuples.append(tuple(chosen))
                        return
                    index = len(chosen)
                    for state in self.placements[cells[index]]:
                        if self.square_work >= self.square_work_budget:
                            exhausted = True
                            return
                        self.square_work += 1
                        if self.square_work % 128 == 0:
                            self.check()
                        if state // 4 in pieces:
                            continue
                        if index == 1 and faces[chosen[0]][1] != faces[state][3]:
                            continue
                        if index == 2 and faces[chosen[0]][2] != faces[state][0]:
                            continue
                        if index == 3 and (faces[chosen[1]][2] != faces[state][0] or faces[chosen[2]][1] != faces[state][3]):
                            continue
                        visit(chosen + [state], pieces | {state // 4})
                        if exhausted:
                            return
                visit([], set())
                if exhausted:
                    self.square_skipped += 1
                    continue
                self.check()
                supports = {c: {state: [] for state in self.placements[c]} for c in cells}
                selectors = []
                for states in tuples:
                    selector = self._new_var()
                    selectors.append(selector)
                    for cell, state in zip(cells, states):
                        self._add([-selector, self.placements[cell][state]])
                        supports[cell][state].append(selector)
                self._add(selectors)
                for cell in cells:
                    for state, selectors_for_state in supports[cell].items():
                        self._add([-self.placements[cell][state], *selectors_for_state])
                self.square_encoded += 1
                self.square_tuples += len(tuples)
        self.check()
        self._built = True

    def solve(self, assumptions=None, *, seconds=None):
        result = super().solve(assumptions, seconds=seconds)
        result.update(square_support_work=self.square_work, square_support_encoded=self.square_encoded,
                      square_support_skipped=self.square_skipped, square_support_tuples=self.square_tuples,
                      square_support_work_budget=self.square_work_budget)
        return result


class ExplainedRegionExpansion:
    """Suggest extra mutable cells from actual full-target SAT assumption cores.

    A failed frozen-placement query proves only that at least one core placement
    must change to achieve a perfect board. It does NOT prove a local score-gain
    neighborhood impossible. The suggested release is a heuristic, using core
    recurrence and contact with the current region; the native model and learned
    clauses are reused. No frozen placement is changed by this helper.
    """
    def __init__(self, problem, *, seconds=10., seed=0, should_stop=None,
                 square_support=False, max_queries=64):
        _full_grid(problem)
        if type(square_support) is not bool:
            raise ValueError('square_support must be a boolean')
        if type(max_queries) is not int or max_queries < 1:
            raise ValueError('max_queries must be a positive integer')
        session_class = SquareSupportSession if square_support else AssumptionSession
        self.session = session_class(problem, seconds=seconds, seed=seed, should_stop=should_stop)
        self.problem = self.session.problem
        self.max_queries = max_queries
        self.core_frequency = Counter()
        self.queries = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.session.close()

    def expand(self, reference, free_cells, *, seconds=.1, max_cells=64):
        seconds = _duration(seconds, 'seconds')
        if type(max_cells) is not int or not 1 <= max_cells <= 64:
            raise ValueError('max_cells must be an integer in 1..64')
        free_cells = _cells(self.problem, free_cells, max_cells)
        checked = evaluate_board(self.problem, reference)
        if not checked['valid']:
            raise ValueError('Reference violates pieces, domains, frame or fixed oriented clues')
        result = {'region_cells': free_cells.copy(), 'released_cells': [], 'core_cells': None,
                  'core_placements': None, 'board': None, 'branches': 0, 'conflicts': 0,
                  'queries': self.queries, 'proof_target': 'all-shared-edges-match',
                  'proof_scope': 'full-target-with-temporary-frozen-placements-only'}
        if self.queries >= self.max_queries:
            return dict(result, outcome='query_limit')
        assumptions = {c: state for c, state in enumerate(reference)
                       if c not in free_cells and c not in self.problem.fixed}
        answer = self.session.solve(assumptions, seconds=seconds)
        self.queries += 1
        result.update(outcome=answer['outcome'], branches=answer['branches'], conflicts=answer['conflicts'],
                      queries=self.queries, sat_result=answer)
        if answer['outcome'] == 'solved':
            if not check_solution(self.problem, answer['board']):
                raise RuntimeError('SAT expansion witness failed exact validation')
            return dict(result, board=list(answer['board']))
        if answer['outcome'] != 'infeasible':
            return result
        core = answer['core_placements']
        if core is None or any(cell not in assumptions or assumptions[cell] != state for cell, state in core.items()):
            raise RuntimeError('SAT expansion returned a core outside active frozen placements')
        result.update(core_cells=sorted(core), core_placements=dict(core))
        if not core:
            return dict(result, outcome='base_infeasible')
        self.core_frequency.update(core.keys())
        if len(free_cells) >= max_cells:
            return dict(result, outcome='region_limit')
        adjacent = Counter()
        mismatches = Counter()
        for a, b, sa, sb in grid_edges(self.problem.size):
            if a in free_cells:
                adjacent[b] += 1
            if b in free_cells:
                adjacent[a] += 1
            if self.problem.faces[reference[a]][sa] != self.problem.faces[reference[b]][sb]:
                mismatches[a] += 1
                mismatches[b] += 1
        chosen = max(core, key=lambda c: (self.core_frequency[c], adjacent[c], mismatches[c], -c))
        return dict(result, outcome='expanded', region_cells=sorted([*free_cells, chosen]), released_cells=[chosen])
