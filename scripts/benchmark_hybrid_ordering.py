"""Bounded offline comparison of unguided DFS and GPU-board ordering hints.

Both variants get the same complete five-clue problem and CPU search budget.
The GPU pool already exists; sampling/compilation time is reported separately,
so this single pair is not an end-to-end hybrid speedup benchmark.
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

from exact_model import constraint_edges, from_bundle, problem_hash
from search_resources import peak_memory_mb
from validator import load_bundle, validate_board

VARIANTS = ('baseline', 'guided')
MAX_REPORT_BYTES = 2 * 1024 * 1024
SAMPLE_FIELDS = ('backend', 'device', 'generated_boards', 'selected_boards', 'cpu_validated_boards',
                 'duplicate_candidates', 'elapsed_seconds', 'boards_per_second', 'compile_seconds',
                 'initialization_seconds', 'kernel_seconds')


def checked_hints(document, bundle):
    if not isinstance(document, dict):
        raise ValueError('GPU report must be a JSON object')
    boards, scores = document.get('boards'), document.get('scores')
    if not isinstance(boards, list) or not 1 <= len(boards) <= 64:
        raise ValueError('GPU report requires 1..64 complete candidate boards')
    if not isinstance(scores, list) or len(scores) != len(boards):
        raise ValueError('GPU report requires one score per board')
    for board, score in zip(boards, scores):
        checked = validate_board(board, bundle)
        if not checked['valid'] or type(score) is not int or checked['score'] != score:
            raise ValueError('GPU hint failed independent piece/frame/five-clue/score validation')
    # Preserve the exact hint sequence; votes and checkpoint identity depend on it.
    return [list(board) for board in boards], {key: document[key] for key in SAMPLE_FIELDS if key in document}


def read_hints(path, bundle):
    with Path(path).open('rb') as stream:
        payload = stream.read(MAX_REPORT_BYTES + 1)
    if len(payload) > MAX_REPORT_BYTES:
        raise ValueError('GPU sampling report exceeds 2 MiB')
    boards, metadata = checked_hints(json.loads(payload), bundle)
    return boards, metadata, hashlib.sha256(payload).hexdigest()


def child(args):
    prepared = time.monotonic()
    bundle = load_bundle()
    problem = from_bundle(bundle)
    if problem.size != 16 or len(problem.fixed) != 5 or len(constraint_edges(problem)) != 480:
        raise AssertionError('Comparison requires the complete five-clue puzzle')
    boards, metadata, source_hash = read_hints(args.hints, bundle)
    preparation_seconds = time.monotonic() - prepared
    interrupted, sampled_at = [], [0.0]
    def should_stop():
        now = time.monotonic()
        if now - sampled_at[0] >= .1:
            sampled_at[0] = now
            if peak_memory_mb() > args.memory_mb:
                interrupted.append('memory_limit')
        return bool(interrupted)
    begin = time.monotonic()
    from exact_dfs import solve
    remaining = max(0.0, args.seconds - (time.monotonic() - begin))
    result = solve(problem, seconds=remaining, seed=args.seed, workers=1,
                   hints=boards if args.variant == 'guided' else None, should_stop=should_stop)
    result.pop('checkpoint', None)
    result.update(variant=args.variant, scope='full-16x16-five-clue-puzzle-all-480-edges',
                  problem_sha256=problem_hash(problem), hint_input_sha256=source_hash,
                  gpu_hint_pool_size=len(boards), gpu_hint_score_min=min(metadata_score(bundle, b) for b in boards),
                  gpu_hint_score_max=max(metadata_score(bundle, b) for b in boards),
                  gpu_sampling=metadata, budget_seconds=args.seconds,
                  preparation_seconds=preparation_seconds,
                  process_task_seconds=time.monotonic() - begin,
                  peak_memory_mb=peak_memory_mb(), workers_requested=1)
    if result['outcome'] == 'solved':
        checked = validate_board(result.get('board'), bundle)
        if not checked['complete']:
            raise AssertionError('DFS result failed independent full-board 480/480 validation')
        result.update(independently_validated=True, full_board_score=checked['score'])
    if interrupted:
        result['resource_interruption'] = 'memory_limit'
        if result['outcome'] == 'infeasible':
            raise AssertionError('Resource interruption must not report infeasibility')
    print(json.dumps(result, allow_nan=False))
    return 0


def metadata_score(bundle, board):
    return validate_board(board, bundle)['score']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hints', type=Path, required=True, help='Existing independently checked GPU sampling report')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--memory-mb', type=int, default=2048)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60 or not 128 <= args.memory_mb <= 4096:
        parser.error('Invalid resource caps')
    if not 0 <= args.seed < 2**32:
        parser.error('seed must be a uint32 integer')
    if args.child:
        if args.variant is None:
            parser.error('--child requires --variant')
        return child(args)
    if args.output is None:
        parser.error('--output required')
    args.hints = args.hints.resolve()
    bundle = load_bundle()
    boards, metadata, expected_hash = read_hints(args.hints, bundle)
    report = {'schema_version': 1, 'experiment': 'DFS ordering hints from an existing GPU sample pool',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'platform': platform.platform(), 'seed': args.seed,
              'source_report_name': args.hints.name, 'hint_input_sha256': expected_hash,
              'gpu_hint_pool_size': len(boards), 'gpu_sampling': metadata,
              'limits': {'cpu_seconds_per_variant': args.seconds, 'workers': 1,
                         'soft_peak_memory_mb': args.memory_mb, 'hard_timeout_extra_seconds': 15},
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ('exact_model.py', 'exact_dfs.py', 'search_resources.py', 'scripts/benchmark_hybrid_ordering.py')},
              'results': [],
              'limitations': ['One sequential isolated-process pair; desktop and BOINC load are uncontrolled.',
                              'Sampling and compilation happened earlier and are not charged to either CPU search budget.',
                              'Both variants independently revalidate the same source pool during untimed preparation.',
                              'Hints only break ties after the existing value-support ordering; no domain is frozen or removed.',
                              'Branches, conflicts and propagation calls are operation counts, not percentages of puzzle completion.',
                              'Two timeouts cannot establish which algorithm solves the full puzzle faster.',
                              'This experiment cannot prove that GPU hint generation is the best algorithm or an end-to-end speedup.']}
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, args.output)
    for variant in VARIANTS:
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--child', '--variant', variant,
                   '--hints', str(args.hints), '--seconds', str(args.seconds), '--seed', str(args.seed),
                   '--memory-mb', str(args.memory_mb)]
        begin = time.monotonic()
        try:
            run = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=args.seconds + 15)
            if run.returncode:
                result = {'variant': variant, 'outcome': 'error', 'returncode': run.returncode, 'stderr': run.stderr[-4000:]}
            else:
                result = json.loads(run.stdout)
                if result.get('hint_input_sha256') != expected_hash:
                    raise ValueError('GPU sample report changed during comparison')
        except subprocess.TimeoutExpired:
            result = {'variant': variant, 'outcome': 'external_timeout'}
        result['total_process_seconds'] = time.monotonic() - begin
        report['results'].append(result)
        save()
        fields = ('variant', 'outcome', 'elapsed_seconds', 'branches', 'conflicts', 'propagations', 'matching_checks', 'matching_failures', 'max_depth', 'hint_count', 'peak_memory_mb')
        print(json.dumps({key: result[key] for key in fields if key in result}), flush=True)
    outcomes = {r['outcome'] for r in report['results']}
    report['contradictory_completed_results'] = {'solved', 'infeasible'} <= outcomes
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    return int(report['contradictory_completed_results'] or 'error' in outcomes)


if __name__ == '__main__':
    raise SystemExit(main())
