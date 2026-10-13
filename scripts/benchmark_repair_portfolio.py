"""Bounded offline ablation of parent diversity and structured repair operators.

By default six sequential cases use seeds 17 and 41 in reversed mode order.
--modes and --seeds allow a smaller bounded regression after an isolated fix. Each solver
gets at most 45 seconds, one CPU worker and GPU batch 16. Every child has a
75-second external timeout and a cooperative 2048 MiB host-memory stop.
Optional warm-up has a separate cooperative 20-second budget before the search.
Only the new report and its sibling artifact directory may be written.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODES = ('baseline', 'diverse-only', 'combined')
SEEDS = (17, 41)
MEMORY_MB = 2048
PROCESS_SECONDS = 75
WARMUP_SECONDS = 20
PROTECTED = (
    'STOP', 'best.json', 'status.json', 'discoveries.sqlite',
    'dashboard-settings.json', 'checkpoint.npz', 'checkpoint.replicas-4096.npz',
    'checkpoint.replicas-8192.npz', 'exact-dfs-checkpoint.json',
    'exact-hybrid-dfs-checkpoint.json', 'exact-session.json', 'hybrid-hints.json',
)
SOURCE_FILES = (
    'smart_repair.py', 'repair_parents.py', 'repair_diversify.py',
    'repair_regions.py', 'binary_pairwise.py', 'gpu_assignment.py',
    'gpu_backends.py', 'kernel_port.py', 'kernels.cu', 'exact_model.py',
    'exact_optimize.py', 'validator.py', 'search_resources.py', 'app_paths.py',
    'data/pieces.txt', 'data/puzzle.json', 'data/record466.json',
    'scripts/benchmark_repair_portfolio.py',
)


def sha_file(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    return {name: sha_file(ROOT / name) for name in SOURCE_FILES}


def protected_hashes(home):
    files = {str(Path('runtime') / name): home / 'runtime' / name
             for name in PROTECTED}
    files.update({str(Path('library') / name): home / 'library' / name
                  for name in ('library.sqlite3', 'library.sqlite3-wal')})
    return {name: sha_file(path) for name, path in files.items()}


def board_hash(board):
    return hashlib.sha256(json.dumps(list(board), separators=(',', ':')).encode()).hexdigest()


def environment():
    versions = {}
    for name in ('numpy', 'ortools', 'python-sat', 'cupy-cuda12x', 'pyopencl'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    def git(*arguments):
        run = subprocess.run(['git', *arguments], cwd=ROOT, capture_output=True,
                             text=True, timeout=10, check=False)
        return run.stdout.strip() if run.returncode == 0 else None
    return {'python': sys.version, 'python_executable': sys.executable,
            'platform': platform.platform(), 'machine': platform.machine(),
            'processor': platform.processor(), 'dependencies': versions,
            'git_commit': git('rev-parse', 'HEAD'),
            'git_dirty_porcelain': git('status', '--porcelain')}


def network_guard():
    attempts = []
    def audit(event, arguments):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            attempts.append(event)
            raise RuntimeError('Network operation forbidden in offline benchmark: ' + event)
    sys.addaudithook(audit)
    return attempts


def old_parents(problem, path, stopped):
    """Historical score/hash top-32 query; independently validate afterward."""
    from exact_model import check_solution
    from dataclasses import replace
    if not path.is_file() or stopped():
        return []
    structural = replace(problem, edges=())
    parents = []
    interrupted = False
    def progress():
        nonlocal interrupted
        interrupted = bool(stopped())
        return int(interrupted)
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro',
                                 uri=True, timeout=1)) as db:
        db.set_progress_handler(progress, 1000)
        try:
            for (blob,) in db.execute(
                    'SELECT placement FROM boards WHERE placement IS NOT NULL '
                    'ORDER BY score DESC,public_sha LIMIT 32'):
                if stopped():
                    break
                if not isinstance(blob, bytes) or len(blob) != 2 * problem.size**2:
                    continue
                board = list(struct.unpack('<' + str(problem.size**2) + 'H', blob))
                if check_solution(structural, board) and board not in parents:
                    parents.append(board)
        except sqlite3.OperationalError:
            if not interrupted:
                raise
        finally:
            db.set_progress_handler(None, 0)
    return parents


def parent_summary(problem, parents, reference, bundle):
    from exact_model import grid_edges
    from validator import validate_board
    edges = grid_edges(problem.size)
    def mismatches(board):
        return {i for i, (a, b, sa, sb) in enumerate(edges)
                if problem.faces[board[a]][sa] != problem.faces[board[b]][sb]}
    records = []
    for board in parents:
        checked = validate_board(board, bundle)
        if not checked['valid']:
            raise AssertionError('Independent parent piece/frame/clue validation failed')
        records.append({'sha256': board_hash(board), 'score': checked['score'],
                        'state_distance_from_reference': sum(a != b for a, b in zip(board, reference))})
    masks = [mismatches(board) for board in parents]
    values = {'different_piece_cells': [], 'different_state_cells': [],
              'mismatch_symmetric_difference': [], 'mismatch_jaccard_distance': []}
    for a, b in itertools.combinations(range(len(parents)), 2):
        values['different_piece_cells'].append(sum(x//4 != y//4 for x, y in zip(parents[a], parents[b])))
        values['different_state_cells'].append(sum(x != y for x, y in zip(parents[a], parents[b])))
        union = masks[a] | masks[b]
        delta = len(masks[a] ^ masks[b])
        values['mismatch_symmetric_difference'].append(delta)
        values['mismatch_jaccard_distance'].append(delta / len(union) if union else 0.0)
    distances = {key: {'minimum': min(items), 'mean': statistics.mean(items), 'maximum': max(items)}
                 if items else {'minimum': None, 'mean': None, 'maximum': None}
                 for key, items in values.items()}
    return {'count': len(parents), 'distinct_boards': len({record['sha256'] for record in records}),
            'records': records, 'pairwise_distances': distances,
            'all_independently_validated': True}



def warm_up(problem, reference, bundle, args, stopped):
    """Warm actual-board assignment and native solvers without changing seeds."""
    from gpu_assignment import AssignmentEvaluator, AssignmentCancelled
    from smart_repair import assignment_repair, assignment_repair_batch, independent_cells
    from validator import validate_board

    started = time.monotonic()
    deadline = started + WARMUP_SECONDS
    def expired():
        return stopped() or time.monotonic() >= deadline
    def require_budget():
        if expired():
            raise TimeoutError('Warm-up exceeded its cooperative 20-second/resource budget')
    cells = independent_cells(problem, seed=args.seed, parity=0)
    require_budget()
    assignment_started = time.monotonic()
    if args.backend == 'cpu':
        result = assignment_repair(problem, reference, cells, should_stop=expired)
        stats = {'cost_backend': 'cpu', 'cost_device': 'CPU'}
    else:
        evaluator = AssignmentEvaluator(problem, backend=args.backend, batch_size=16,
                                        cache_dir=args.case_dir / 'gpu-cache',
                                        allow_cpu_fallback=False, should_stop=expired)
        try:
            result = assignment_repair_batch(problem, [(reference, cells)],
                                             evaluator=evaluator, should_stop=expired)[0]
        except AssignmentCancelled as exc:
            raise TimeoutError('GPU warm-up reached its time/resource cap') from exc
        stats = evaluator.stats()
    assignment_seconds = time.monotonic() - assignment_started
    checked = validate_board(result['board'], bundle)
    if not checked['valid'] or checked['score'] != result['score'] or result['score'] < 466:
        raise AssertionError('Warm-up assignment failed independent validation')
    if not result['optimal']:
        raise TimeoutError('Warm-up assignment did not finish within its budget')
    require_budget()
    cp_started = time.monotonic()
    from ortools.sat.python import cp_model
    require_budget()
    model = cp_model.CpModel()
    bit = model.new_bool_var('warmup_bit')
    model.add(bit == 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = min(1., max(0., deadline - time.monotonic()))
    status = solver.solve(model)
    if status != cp_model.OPTIMAL or solver.value(bit) != 1:
        raise RuntimeError('Single-worker native CP-SAT warm-up failed')
    cp_seconds = time.monotonic() - cp_started
    require_budget()
    report = {'enabled': True, 'seconds': time.monotonic() - started,
              'cooperative_budget_seconds': WARMUP_SECONDS,
              'assignment_seconds': assignment_seconds, 'cp_sat_seconds': cp_seconds,
              'assignment_score': checked['score'], 'assignment_optimal': result['optimal'],
              'assignment_sha256': board_hash(result['board']),
              'independently_validated': True, 'cp_sat_workers': 1,
              'same_gpu_cache_as_search': True, 'used_as_search_seed': False,
              **stats}
    if checked['score'] > 466:
        report['improved_board'] = list(result['board'])
    return report

def child(args):
    attempts = network_guard()
    from exact_model import from_bundle, problem_hash
    from repair_parents import cached_parents
    from search_resources import peak_memory_mb
    from smart_repair import solve
    from validator import load_bundle, validate_board

    started = time.monotonic()
    source_before = source_hashes()
    resource_reasons = set()
    last_memory_check = float('-inf')
    peak = 0.0
    def stopped():
        nonlocal last_memory_check, peak
        now = time.monotonic()
        if now - last_memory_check >= .05:
            last_memory_check = now
            peak = max(peak, peak_memory_mb())
            if peak > MEMORY_MB:
                resource_reasons.add('host_memory_limit')
        if now >= started + PROCESS_SECONDS - 5:
            resource_reasons.add('child_wall_budget')
        return bool(resource_reasons)

    bundle = load_bundle()
    problem = from_bundle(bundle)
    reference = list(bundle.record_board)
    reference_check = validate_board(reference, bundle)
    if not reference_check['valid'] or reference_check['score'] != 466:
        raise AssertionError('Official reference failed independent validation')
    library = args.home / 'library' / 'library.sqlite3'
    preparation_start = time.monotonic()
    preparation_deadline = preparation_start + 5
    def preparation_stop():
        return stopped() or time.monotonic() >= preparation_deadline
    parents = old_parents(problem, library, preparation_stop) if args.child == 'baseline' else cached_parents(
        problem, library, limit=32, seed=args.seed, candidate_limit=256,
        max_seconds=5, should_stop=stopped)
    preparation_seconds = time.monotonic() - preparation_start
    summary = parent_summary(problem, parents, reference, bundle)
    improvements = []
    progress_samples = []
    last_verified = reference_check['score']

    def improved(item):
        nonlocal last_verified
        checked = validate_board(item['board'], bundle)
        if not checked['valid'] or checked['score'] != item['score'] or checked['score'] <= last_verified:
            raise AssertionError('Improvement callback failed independent validation/monotonicity')
        last_verified = checked['score']
        improvements.append(dict(item, board=list(item['board']),
                                 independently_validated=True, sha256=board_hash(item['board'])))

    def progress(item):
        # A bounded sample aids debugging without changing the local dashboard.
        if len(progress_samples) < 128:
            progress_samples.append(dict(item))

    warmup = warm_up(problem, reference, bundle, args, stopped) if args.warmup else {'enabled': False, 'seconds': 0.}
    enabled = args.child == 'combined'
    solve_started = time.monotonic()
    result = solve(problem, reference, seconds=args.seconds, seed=args.seed, workers=1,
                   parents=parents, should_stop=stopped, on_improvement=improved,
                   on_progress=progress, gpu_backend=None if args.backend == 'cpu' else args.backend,
                   gpu_batch_size=16, gpu_cache_dir=args.case_dir / 'gpu-cache',
                   gpu_allow_cpu_fallback=False, diversify=enabled,
                   region_repair_enabled=enabled)
    solve_seconds = time.monotonic() - solve_started
    checked = validate_board(result['board'], bundle)
    if not checked['valid'] or checked['score'] != result['score'] or result['score'] < 466:
        raise AssertionError('Final board failed independent validation/nonregression')
    if result['score'] != last_verified:
        raise AssertionError('Final score disagrees with independently validated callbacks')
    source_after = source_hashes()
    report = {'mode': args.child, 'seed': args.seed, 'backend_requested': args.backend,
              'passed': source_before == source_after and not attempts,
              'sources_unchanged': source_before == source_after,
              'source_sha256_before': source_before, 'source_sha256_after': source_after,
              'problem_sha256': problem_hash(problem), 'reference_sha256': board_hash(reference),
              'reference_score': reference_check['score'], 'parent_selection_seconds': preparation_seconds,
              'parents': summary, 'warmup': warmup, 'solver_wall_seconds': solve_seconds,
              'total_child_seconds': time.monotonic() - started,
              'peak_host_memory_mb': max(peak, peak_memory_mb()),
              'resource_interruptions': sorted(resource_reasons),
              'network_attempts': attempts, 'improvements': improvements,
              'progress_samples': progress_samples, 'result': result,
              'final_board_sha256': board_hash(result['board']),
              'final_independent_validation': checked}
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report['passed'] else 1


def inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--home', type=Path, default=ROOT.parent)
    parser.add_argument('--backend', choices=('cuda', 'opencl', 'cpu'), default='cuda')
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--warmup', action='store_true', help='Warm GPU assignment and native CPU solvers before timing search')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--child', choices=MODES)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument('--case-dir', type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 45:
        parser.error('--seconds must be positive and at most 45')
    if len(set(args.modes)) != len(args.modes) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Modes and seeds must be distinct')
    args.home = args.home.resolve()
    protected_dirs = tuple((args.home / name).resolve() for name in ('runtime', 'library', 'data'))
    if args.child:
        if args.case_dir is None:
            parser.error('--child requires --case-dir')
        args.case_dir = args.case_dir.resolve()
        if any(inside(args.case_dir, directory) for directory in protected_dirs):
            parser.error('Child artifact directory must be outside runtime/library/data')
        args.case_dir.mkdir(parents=True, exist_ok=True)
        return child(args)
    if args.output is None:
        parser.error('--output is required')
    args.output = args.output.resolve()
    artifacts = args.output.with_name(args.output.name + '.artifacts')
    if args.output.exists() or artifacts.exists():
        parser.error('Use a new output and sibling artifact directory to preserve prior evidence')
    if any(inside(args.output, directory) for directory in protected_dirs):
        parser.error('Output must be outside canonical runtime/library/data')
    before = source_hashes()
    protected_before = protected_hashes(args.home)
    if protected_before.get(str(Path('runtime') / 'STOP')) is None:
        parser.error('Canonical runtime/STOP must exist before this isolated benchmark')
    report = {'schema': 'eternity-repair-portfolio-benchmark/v1',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'environment': environment(), 'source_sha256_before': before,
              'protected_files_sha256_before': protected_before,
              'limits': {'solver_seconds': args.seconds, 'child_process_timeout_seconds': PROCESS_SECONDS,
                         'parent_preparation_seconds': 5, 'warmup_enabled': args.warmup,
                         'warmup_cooperative_seconds': WARMUP_SECONDS if args.warmup else 0,
                         'shared_child_cooperative_seconds': PROCESS_SECONDS - 5,
                         'soft_host_memory_mb': MEMORY_MB,
                         'cupy_gpu_memory_limit_bytes': MEMORY_MB * 1024**2,
                         'cpu_workers': 1, 'gpu_batch_size': 16, 'cases': len(args.modes)*len(args.seeds)},
              'backend': args.backend, 'seeds': args.seeds, 'modes': args.modes, 'results': [],
              'non_claims': [
                  'Bounded samples do not establish faster time to 480 or full-puzzle infeasibility.',
                  'Baseline disables new operators and uses historical score/hash top-32 parents; shared solver code remains current.',
                  'Each case starts with a fresh GPU cache. Optional warm-up is timed separately and uses the same cache/process as search.',
                  'Warm-up results are independently validated but do not replace the original reference or selected search parents.',
                  'Pairwise parent diversity is descriptive and is not a measure of solved search coverage.',
                  'Host-memory limit is cooperative; a native allocation may exceed it before the next check.',
                  'No canonical best, checkpoint, library, dashboard, BOINC or network state is written.',
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    started = time.monotonic()
    save()
    try:
        for seed_index, seed in enumerate(args.seeds):
            order = args.modes if seed_index == 0 else tuple(reversed(args.modes))
            for mode in order:
                case_dir = artifacts / (str(seed) + '-' + mode)
                case_dir.mkdir()
                temporary = case_dir / 'tmp'
                temporary.mkdir()
                env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONHASHSEED='0',
                           PYTHONDONTWRITEBYTECODE='1', CUPY_GPU_MEMORY_LIMIT=str(MEMORY_MB * 1024**2),
                           CUPY_CACHE_DIR=str(case_dir / 'gpu-cache'),
                           CUDA_CACHE_PATH=str(case_dir / 'cuda-driver-cache'), TEMP=str(temporary),
                           TMP=str(temporary), TMPDIR=str(temporary))
                command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()),
                           '--child', mode, '--seed', str(seed), '--backend', args.backend,
                           '--seconds', str(args.seconds), '--home', str(args.home),
                           '--case-dir', str(case_dir)]
                if args.warmup:
                    command.append('--warmup')
                case_started = time.monotonic()
                print(json.dumps({'event': 'case_started', 'mode': mode, 'seed': seed,
                                  'backend': args.backend}), flush=True)
                try:
                    run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                         text=True, timeout=PROCESS_SECONDS)
                    if run.returncode == 0:
                        item = json.loads(run.stdout)
                    else:
                        try:
                            item = json.loads(run.stdout)
                        except (ValueError, TypeError):
                            item = {'mode': mode, 'seed': seed, 'passed': False,
                                    'outcome': 'child_error', 'stdout': run.stdout[-4000:]}
                        item.update(returncode=run.returncode, stderr=run.stderr[-8000:])
                except subprocess.TimeoutExpired:
                    item = {'mode': mode, 'seed': seed, 'passed': False,
                            'outcome': 'external_process_timeout'}
                item['total_process_seconds'] = time.monotonic() - case_started
                report['results'].append(item)
                save()
                final = item.get('result', {})
                print(json.dumps({'event': 'case_finished', 'mode': mode, 'seed': seed,
                                  'passed': item.get('passed'), 'score': final.get('score'),
                                  'neighborhoods': final.get('neighborhoods'),
                                  'diversifications': final.get('diversifications'),
                                  'region_calls': final.get('region_calls'),
                                  'process_seconds': item['total_process_seconds']}), flush=True)
    finally:
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      total_wall_seconds=time.monotonic() - started,
                      source_sha256_after=source_hashes(),
                      protected_files_sha256_after=protected_hashes(args.home))
        report['sources_unchanged'] = before == report['source_sha256_after'] and all(
            item.get('source_sha256_before', before) == before and item.get('source_sha256_after', before) == before
            for item in report['results'])
        report['protected_files_unchanged'] = protected_before == report['protected_files_sha256_after']
        report['passed'] = (len(report['results']) == len(args.modes)*len(args.seeds) and report['sources_unchanged']
                            and report['protected_files_unchanged']
                            and all(item.get('passed') for item in report['results']))
        save()
    print(json.dumps({'event': 'complete', 'passed': report['passed'],
                      'sources_unchanged': report['sources_unchanged'],
                      'protected_files_unchanged': report['protected_files_unchanged'],
                      'output': str(args.output)}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
