"""Offline Hall-pruning ablation: identical DFS, one rule disabled.

Each measurement runs sequentially in an isolated, single-worker process.
Counts are observed operations, never estimates of eliminated search subtrees.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exact_dfs
from exact_model import check_solution, from_bundle, generated_problem, make_problem, problem_hash
from search_resources import peak_memory_mb
from validator import load_bundle, validate_board

VARIANTS = ('hall', 'no-hall')
CASES = ('generated-10', 'generated-12', 'hall-8', 'region-6', 'region-12', 'whole-five-clue')


def hall_problem(size=8):
    """Artificial Hall contradiction with no missing/single-location pieces.

    More than half of the edge cells share only half of the edge pieces.
    Other cell domains remain broad; adjacency edges are intentionally omitted
    to isolate the distinct-piece feasibility check.
    """
    if not 4 <= size <= 16:
        raise ValueError('Hall fixture size must be 4..16')
    source, _ = generated_problem(size, 1, 311 + size)
    n = size * size
    cells = [c for c in range(n) if sum((c < size, c % size == size - 1, c >= n - size, c % size == 0)) == 1]
    pieces = sorted({s // 4 for s in source.domains[cells[0]]})
    count = len(pieces) // 2
    allowed = set(pieces[:count])
    domains = list(source.domains)
    for cell in cells[:count + 1]:
        domains[cell] = tuple(s for s in domains[cell] if s // 4 in allowed)
    return make_problem(size, source.faces, domains=domains, edges=())


def case(name):
    if name.startswith('generated-'):
        size = int(name.split('-')[1])
        colors = 5 if size == 10 else 7
        problem, answer = generated_problem(size, colors, 1900 + size, clue_count=1)
        if not check_solution(problem, answer):
            raise AssertionError('Generated reference failed validation')
        return problem, 'synthetic known-solvable puzzle; low color diversity; not Eternity II'
    if name == 'hall-8':
        return hall_problem(8), 'artificial 13-edge-cell/12-piece Hall contradiction; adjacency omitted'
    bundle = load_bundle()
    if name == 'whole-five-clue':
        return from_bundle(bundle), 'complete original puzzle; all 480 edges and all five clues'
    width = int(name.split('-')[1])
    cells = [(3 + row) * 16 + 16 - width + col for row in range(width) for col in range(width)]
    return from_bundle(bundle, board=list(bundle.record_board), free_cells=cells), 'actual 466 seed region; outside frozen; only edges touching the region required'


def solve_variant(problem, variant='hall', *, seconds=3.0, seed=17, should_stop=None):
    """Process-local instrumentation; never used by the runtime solver.

    Disabling matching returns a feasible placeholder, not a contradiction.
    All adjacency and uniqueness propagation plus independent validation stay
    unchanged. No checkpoint is resumed or written by this ablation.
    """
    if variant not in VARIANTS:
        raise ValueError('Unknown pruning variant')
    original = exact_dfs._Search
    instance = []

    class Observed(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.matching_seconds = 0.0
            self.matching_skips = 0
            self.root_before = self.root_after = None
            self.root_propagation_outcome = 'not_started'
            self.root_propagation_seconds = 0.0
            instance.append(self)

        def domain_counts(self, domains):
            counts = [d.bit_count() for d in domains]
            return {'state_candidates': sum(counts),
                    'piece_candidates': sum(self.pieces(d).bit_count() for d in domains),
                    'undecided_cells': sum(c > 1 for c in counts),
                    'empty_cells': sum(c == 0 for c in counts)}

        def build(self):
            domains = super().build()
            self.root_before = self.domain_counts(domains)
            return domains

        def propagate(self, domains, dirty, previous):
            first = self.root_propagation_outcome == 'not_started'
            begin = time.perf_counter()
            if first:
                self.root_propagation_outcome = 'in_progress'
            try:
                result = super().propagate(domains, dirty, previous)
                if first:
                    self.root_propagation_outcome = 'contradiction' if result is None else 'consistent'
                return result
            except exact_dfs._Interrupted:
                if first:
                    self.root_propagation_outcome = 'interrupted'
                raise
            finally:
                if first:
                    self.root_after = self.domain_counts(domains)
                    self.root_propagation_seconds = time.perf_counter() - begin

        def matching(self, domains, previous):
            if variant == 'no-hall':
                self.matching_skips += 1
                return [-1] * self.n
            begin = time.perf_counter()
            try:
                return super().matching(domains, previous)
            finally:
                self.matching_seconds += time.perf_counter() - begin

    exact_dfs._Search = Observed
    try:
        result = exact_dfs.solve(problem, seconds=seconds, seed=seed, workers=1, should_stop=should_stop)
    finally:
        exact_dfs._Search = original
    observed = instance[0]
    result.pop('checkpoint', None)
    result.update(variant=variant, matching_seconds=observed.matching_seconds,
                  matching_skips=observed.matching_skips,
                  root_before=observed.root_before, root_after=observed.root_after,
                  root_propagation_outcome=observed.root_propagation_outcome,
                  root_propagation_seconds=observed.root_propagation_seconds)
    if result['outcome'] == 'solved':
        if result['board'] is None or not check_solution(problem, result['board']):
            raise AssertionError('Independent exact-task validation failed')
        result['independently_validated'] = True
    return result


def child(args):
    problem, scope = case(args.case)
    interrupted = []
    sampled = [0.0]
    def stop_for_memory():
        now = time.monotonic()
        if now - sampled[0] >= .1:
            sampled[0] = now
            if peak_memory_mb() > args.memory_mb:
                interrupted.append('memory_limit')
        return bool(interrupted)
    begin = time.monotonic()
    result = solve_variant(problem, args.variant, seconds=args.seconds, seed=args.seed, should_stop=stop_for_memory)
    result.update(case=args.case, scope=scope, problem_sha256=problem_hash(problem),
                  budget_seconds=args.seconds, process_task_seconds=time.monotonic() - begin,
                  peak_memory_mb=peak_memory_mb(), workers_requested=1)
    if result['outcome'] == 'solved' and problem.size == 16:
        full = validate_board(result['board'])
        if not full['valid']:
            raise AssertionError('Independent official-puzzle legality check failed')
        result.update(full_board_score=full['score'], full_board_complete=full['complete'])
    if interrupted:
        result['resource_interruption'] = 'memory_limit'
        if result['outcome'] == 'infeasible':
            raise AssertionError('Resource interruption must not claim infeasibility')
    print(json.dumps(result, allow_nan=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--seconds', type=float, default=3)
    parser.add_argument('--whole-seconds', type=float, default=10)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--memory-mb', type=int, default=2048)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not all(math.isfinite(s) and 0 < s <= 60 for s in (args.seconds, args.whole_seconds)) or not 128 <= args.memory_mb <= 4096:
        parser.error('Invalid resource caps')
    if args.child:
        if args.case is None or args.variant is None:
            parser.error('--child requires --case and --variant')
        return child(args)
    if args.output is None:
        parser.error('--output required')
    report = {'schema_version': 1, 'experiment': 'DFS Hall-feasibility ablation',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'platform': platform.platform(), 'seed': args.seed,
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ('exact_model.py', 'exact_dfs.py', 'search_resources.py', 'scripts/benchmark_pruning.py')},
              'limits': {'workers_per_case': 1, 'ordinary_seconds': args.seconds,
                         'whole_seconds': args.whole_seconds, 'soft_peak_memory_mb': args.memory_mb,
                         'hard_process_timeout_extra_seconds': 15},
              'results': [],
              'limitations': ['Only Hall matching is disabled; all other pruning remains active.',
                              'Matching checks are observed feasibility tests, not a count of eliminated subtrees.',
                              'Root candidate counts after contradiction may reflect a partially processed failed domain.',
                              'Timeouts are inconclusive; candidate counts and branches are not solved-puzzle progress.',
                              'Synthetic and regional results do not establish whole-puzzle performance.',
                              'Desktop and BOINC load are uncontrolled; single-run timings are exploratory.']}
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, args.output)
    for index, name in enumerate(args.cases):
        order = VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))
        for variant in order:
            budget = args.whole_seconds if name == 'whole-five-clue' else args.seconds
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--child', '--case', name,
                       '--variant', variant, '--seconds', str(budget), '--seed', str(args.seed),
                       '--memory-mb', str(args.memory_mb)]
            begin = time.monotonic()
            try:
                run = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=budget + 15)
                if run.returncode:
                    result = {'case': name, 'variant': variant, 'outcome': 'error', 'returncode': run.returncode, 'stderr': run.stderr[-4000:]}
                else:
                    result = json.loads(run.stdout)
            except subprocess.TimeoutExpired:
                result = {'case': name, 'variant': variant, 'outcome': 'external_timeout'}
            result['total_process_seconds'] = time.monotonic() - begin
            report['results'].append(result)
            save()
            fields = ('case', 'variant', 'outcome', 'elapsed_seconds', 'branches', 'conflicts', 'matching_checks', 'matching_failures', 'matching_seconds')
            print(json.dumps({key: result[key] for key in fields if key in result}), flush=True)
    contradictions = []
    for name in args.cases:
        outcomes = {r['outcome'] for r in report['results'] if r['case'] == name}
        if {'solved', 'infeasible'} <= outcomes:
            contradictions.append(name)
    report['contradictory_completed_results'] = contradictions
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    return int(bool(contradictions) or any(r['outcome'] == 'error' for r in report['results']))


if __name__ == '__main__':
    raise SystemExit(main())
