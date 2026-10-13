"""Bounded, offline, sequential ablation of optional DFS improvements.

One isolated CPU worker per case; no application/runtime files, GPU, or network.
Timings are exploratory. A timeout proves nothing about puzzle feasibility.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
from exact_model import check_solution, from_bundle, generated_problem, problem_hash
from exact_rules import DEFAULT_OPTIONS as RULE_DEFAULTS
from search_resources import peak_memory_mb
from validator import load_bundle, validate_board


def one_rule(name):
    return {'extra_rules': {key: key == name for key, value in RULE_DEFAULTS.items() if type(value) is bool}}


CHEAP = dict(gac=True, independent_completion=True, dual_branch=True, edge_branch=True)
VARIANTS = {
    'baseline': {},
    'gac': {'gac': True},
    'independent': {'independent_completion': True},
    'dual': {'dual_branch': True},
    'edge': {'edge_branch': True},
    'probe': {'probe': True},
    'cheap-combined': CHEAP,
    'identical-hall': one_rule('hall'),
    'checkerboard': one_rule('color_balance'),
    'directional': one_rule('directional_balance'),
    'blocks-2x2': one_rule('blocks'),
    'color-flow': one_rule('color_flow'),
    'all-additional-rules': {'extra_rules': {}},
    'cheap-plus-additional': dict(CHEAP, extra_rules={}),
    'full': dict(CHEAP, extra_rules={}, probe=True),
    'adaptive': dict(CHEAP, extra_rules={}, probe=True, adaptive=True),
}
CASES = ('generated-8', 'region-12', 'whole-five-clue')
SOURCE_FILES = ('exact_dfs.py', 'exact_rules.py', 'exact_model.py', 'validator.py',
                'search_resources.py', 'scripts/benchmark_smart_search.py')


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def case(name):
    if name == 'generated-8':
        problem, answer = generated_problem(8, 11, 909, clue_count=1)
        if not check_solution(problem, answer):
            raise AssertionError('Generated reference failed validation')
        return problem, 'synthetic known-solvable 8x8 puzzle; not Eternity II'
    bundle = load_bundle()
    if name == 'whole-five-clue':
        return from_bundle(bundle), 'whole original puzzle with all 480 edges and all five official clues'
    cells = [(3 + row) * 16 + 4 + col for row in range(12) for col in range(12)]
    return from_bundle(bundle, board=list(bundle.record_board), free_cells=cells), 'actual 466 seed; 12x12 region; outside frozen; only edges touching region constrained'


def child(args):
    problem, scope = case(args.case)
    interruptions = []
    sampled = [0.0]
    def should_stop():
        now = time.monotonic()
        if now - sampled[0] >= .1:
            sampled[0] = now
            if peak_memory_mb() > 2048:
                interruptions.append('memory_limit')
        return bool(interruptions)
    started = time.monotonic()
    result = exact_dfs.solve(problem, seconds=args.seconds, seed=args.seed, workers=1,
                             options=VARIANTS[args.variant], should_stop=should_stop)
    result.pop('checkpoint', None)
    result.update(case=args.case, variant=args.variant, scope=scope,
                  problem_sha256=problem_hash(problem), source_sha256=source_hashes(),
                  requested_seconds=args.seconds, worker_count=1,
                  task_wall_seconds=time.monotonic() - started, peak_memory_mb=peak_memory_mb())
    board = result.get('board')
    if result['outcome'] == 'solved':
        if board is None or not check_solution(problem, board):
            raise AssertionError('Independent task validation failed')
        result['independently_validated'] = True
        if problem.size == 16:
            full = validate_board(board)
            if not full['valid']:
                raise AssertionError('Independent official piece/frame/clue validation failed')
            result.update(full_board_score=full['score'], full_board_complete=full['complete'])
    if interruptions:
        result['resource_interruption'] = 'memory_limit'
        if result['outcome'] == 'infeasible':
            raise AssertionError('Resource interruption is not infeasibility')
    print(json.dumps(result, allow_nan=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--seconds', type=float, default=1.0)
    parser.add_argument('--full-seconds', type=float, default=3.0)
    parser.add_argument('--wall-seconds', type=float, default=180.0)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    maximum = 3 if args.child and args.case == 'whole-five-clue' else 1
    if not 0 < args.seconds <= maximum or not 0 < args.full_seconds <= 3 or not 0 < args.wall_seconds <= 180:
        parser.error('This harness permits at most 1 s ordinary, 3 s whole-board, and 180 s wall time')
    if args.child:
        if args.case is None or args.variant is None:
            parser.error('Child requires a case and variant')
        return child(args)
    if args.output is None:
        parser.error('--output is required')
    if args.output.exists():
        parser.error('Use a new output path to preserve prior evidence')
    started = time.monotonic()
    before = source_hashes()
    report = {
        'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(),
        'source_sha256_before': before, 'variants': VARIANTS, 'seed': args.seed,
        'limits': {'wall_seconds': args.wall_seconds, 'ordinary_seconds': args.seconds,
                   'whole_seconds': args.full_seconds, 'workers_per_child': 1,
                   'soft_peak_memory_mb': 2048, 'hard_child_timeout_extra_seconds': 8},
        'results': [], 'non_claims': [
            'Timeout and stop are inconclusive; regional infeasibility is conditional on the frozen outside.',
            'Branches and maximum depth are not coverage percentages and may not be comparable across branching schemes.',
            'Short synthetic cases do not establish full-puzzle speed; desktop load is uncontrolled.',
            'No solver/app/runtime state is changed and no network or GPU work is performed.',
        ],
    }
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    exhausted = False
    for index, name in enumerate(args.cases):
        ordered = args.variants[index % len(args.variants):] + args.variants[:index % len(args.variants)]
        budget = args.full_seconds if name == 'whole-five-clue' else args.seconds
        for variant in ordered:
            remaining = args.wall_seconds - (time.monotonic() - started)
            if remaining < budget + 8:
                exhausted = True
                break
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--child',
                       '--case', name, '--variant', variant, '--seconds', str(budget), '--seed', str(args.seed)]
            beginning = time.monotonic()
            try:
                run = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                     text=True, timeout=min(budget + 8, remaining))
                if run.returncode:
                    result = {'case': name, 'variant': variant, 'outcome': 'error',
                              'returncode': run.returncode, 'stderr': run.stderr[-4000:]}
                else:
                    result = json.loads(run.stdout)
            except subprocess.TimeoutExpired:
                result = {'case': name, 'variant': variant, 'outcome': 'external_timeout'}
            result['total_process_seconds'] = time.monotonic() - beginning
            report['results'].append(result)
            save()
            keys = ('case', 'variant', 'outcome', 'branches', 'max_depth', 'elapsed_seconds', 'full_board_score')
            print(json.dumps({key: result[key] for key in keys if key in result}), flush=True)
        if exhausted:
            break
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.monotonic() - started, wall_cap_stopped=exhausted,
                  source_sha256_after=source_hashes())
    report['sources_unchanged'] = report['source_sha256_after'] == before and all(
        r.get('source_sha256', before) == before for r in report['results'])
    save()
    return int(any(r['outcome'] == 'error' for r in report['results']))


if __name__ == '__main__':
    raise SystemExit(main())
