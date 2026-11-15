"""Share the existing DFS root fixpoint with exact, perfect-match backends."""
from dataclasses import replace
import math
import time

from exact_dfs import _Interrupted, _Search
from exact_model import check_solution


def solve(problem, *, backend, seconds=30.0, seed=0, workers=1, should_stop=None,
          on_progress=None, profile='matching', **backend_options):
    """Run a fresh SAT, CP-SAT or regional-master query under one wall budget.

    The prepass only enforces the original supplied constraints. In particular it
    must never be used to optimize a soft score on a partially matching board.
    The root makes no guesses. Every native witness is checked on the original
    domains, fixed states and complete supplied edge set.
    """
    from exact_sat import solve as sat
    from exact_cp import solve as cp
    from regional_master import solve as master
    engines = {'sat': sat, 'cp-sat': cp, 'regional-master': master}
    if backend not in engines or profile not in ('matching', 'rules'):
        raise ValueError('Unknown exact preprocessing backend or profile')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('workers must be an integer in 1..64')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('Invalid should_stop')
    if on_progress is not None and not callable(on_progress):
        raise ValueError('Invalid on_progress')
    if 'assumptions' in backend_options:
        raise ValueError('This entry point accepts standalone exact queries')
    start = time.perf_counter()
    # ExactProblem is frozen but callers may retain its fixed dictionary or
    # domain containers. All phases must see the same original constraints.
    problem = replace(problem, fixed=dict(problem.fixed), faces=tuple(tuple(f) for f in problem.faces),
                      domains=tuple(tuple(d) for d in problem.domains),
                      edges=None if problem.edges is None else tuple(tuple(e) for e in problem.edges))
    options = {'gac': True}
    if profile == 'rules':
        options['extra_rules'] = {'relational': True}
    if on_progress:
        on_progress({'phase': 'preprocessing', 'backend': 'dfs-root-fixpoint',
                     'branches': 0, 'conflicts': 0, 'elapsed_seconds': 0.0})
    search = _Search(problem, max(0.0, seconds - (time.perf_counter() - start)), seed,
                     should_stop, None, None, None, 1.0, options=options)
    initial_states = sum(map(len, problem.domains))
    remaining_states = initial_states
    try:
        search.check()
        domains = search.build()
        matched = search.propagate(domains, range(search.n), None)
        if matched is None:
            reduced = None
            outcome = 'infeasible'
        else:
            exported = []
            for original, mask in zip(problem.domains, domains):
                search.check()
                exported.append(tuple(s for s in dict.fromkeys(original) if mask & (1 << s)))
            reduced = replace(problem, domains=tuple(exported))
            remaining_states = sum(map(len, exported))
        search.check()
    except _Interrupted as interrupted:
        reduced = None
        outcome = interrupted.outcome
    pre_seconds = time.perf_counter() - start
    stats = {'profile': profile, 'seconds': pre_seconds, 'work_ticks': search.work_ticks,
             'matching_checks': search.matching_checks, 'propagations': search.propagations,
             'rules': {} if search.rules is None else dict(search.rules.counters),
             'initial_states': initial_states,
             'remaining_states': remaining_states if reduced is not None else None}
    if reduced is None:
        result = dict(outcome=outcome, board=None, branches=0, conflicts=int(outcome == 'infeasible'),
                      backend=backend, workers=workers if backend=='cp-sat' else 1, native_started=False, build_seconds=pre_seconds)
    else:
        result = engines[backend](reduced, seconds=max(0.0, seconds-pre_seconds),
                                  seed=seed, workers=workers, should_stop=should_stop,
                                  on_progress=on_progress, **backend_options)
        if result.get('board') is not None and not check_solution(problem, result['board']):
            raise RuntimeError('Preprocessed witness failed original puzzle validation')
        result['native_started'] = True
        result['build_seconds'] = pre_seconds + (result.get('build_seconds') or 0.0)
    result['preprocessing'] = stats
    result['elapsed_seconds'] = time.perf_counter() - start
    return result
