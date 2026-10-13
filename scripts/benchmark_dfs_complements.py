"""Sequential paired ablation of exactly equivalent support-mask implementations.

No network, GPU, application state, or sustained search. Each child has <=3s
search, a 2GiB cooperative memory cap and an external budget+8s timeout.
"""
from __future__ import annotations
import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import exact_dfs
from exact_model import check_solution, from_bundle, generated_problem, problem_hash
from search_resources import peak_memory_mb
from validator import load_bundle, validate_board

SOURCES = ('exact_dfs.py', 'exact_rules.py', 'exact_model.py', 'validator.py',
           'search_resources.py', 'scripts/benchmark_dfs_complements.py', 'test_dfs_supported.py')

def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def inline_supported(search, source, source_side, target_side):
    allowed = 0
    for color, states in search.colors[source_side].items():
        support = source & states
        if support:
            candidates = search.colors[target_side].get(color, 0)
            piece_states = 15 << ((support.bit_length() - 1) & ~3)
            if support & ~piece_states == 0:
                candidates &= ~piece_states
            allowed |= candidates
    return allowed


def precomputed_supported(search, source, source_side, target_side):
    allowed = 0
    for color, states in search.colors[source_side].items():
        support = source & states
        if support:
            candidates = search.colors[target_side].get(color, 0)
            other_pieces = search.other_piece_masks[(support.bit_length() - 1) >> 2]
            if support & other_pieces == 0:
                candidates &= other_pieces
            allowed |= candidates
    return allowed


def child(args):
    if args.case == 'generated-8':
        problem, reference = generated_problem(8, 11, 909, clue_count=1)
        assert check_solution(problem, reference)
    else:
        problem = from_bundle(load_bundle())
    if args.variant == 'inline-bit-length':
        exact_dfs._Search.supported = inline_supported
    else:
        exact_dfs._Search.supported = precomputed_supported
        original_build = exact_dfs._Search.build
        def build(search):
            domains = original_build(search)
            search.other_piece_masks = tuple(~(15 << (4 * piece)) for piece in range(search.n))
            return domains
        exact_dfs._Search.build = build
    profiler = cProfile.Profile() if args.profile else None
    interrupted, sampled = [], [0.]
    def stop():
        now = time.monotonic()
        if now - sampled[0] >= .1:
            sampled[0] = now
            if peak_memory_mb() > 2048:
                interrupted.append('memory_limit')
        return bool(interrupted)
    started = time.monotonic()
    if profiler is not None:
        profiler.enable()
    answer = exact_dfs.solve(problem, seconds=args.seconds, seed=17, options={}, should_stop=stop)
    if profiler is not None:
        profiler.disable()
    answer.pop('checkpoint', None)
    answer.update(case=args.case, variant=args.variant, source_sha256=hashes(),
                  problem_sha256=problem_hash(problem), seed=17, requested_seconds=args.seconds,
                  profile_enabled=args.profile, peak_memory_mb=peak_memory_mb(),
                  task_wall_seconds=time.monotonic() - started, resource_interruptions=interrupted)
    if answer['board'] is not None:
        assert check_solution(problem, answer['board'])
        answer['independent_task_validation'] = True
        if problem.size == 16:
            checked = validate_board(answer['board'])
            assert checked['valid'] and checked['complete']
            answer['independent_official_validation'] = checked
    if profiler is not None:
        stats = pstats.Stats(profiler)
        answer['profile'] = [{'file': key[0], 'line': key[1], 'function': key[2],
                              'primitive_calls': value[0], 'total_calls': value[1],
                              'self_seconds': value[2], 'cumulative_seconds': value[3]}
                             for key, value in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:40]]
    print(json.dumps(answer, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--case', choices=('generated-8', 'whole-five-clue'))
    parser.add_argument('--variant', choices=('inline-bit-length', 'precomputed'))
    parser.add_argument('--seconds', type=float, default=3.)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3:
        parser.error('Seconds must be in (0,3]')
    if args.child:
        if not args.case or not args.variant:
            parser.error('Child case and variant required')
        child(args)
        return
    if args.output is None or args.output.exists():
        parser.error('A new output path is required')
    started = time.monotonic()
    before = hashes()
    report = dict(schema_version=1, started_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256_before=before, python=sys.version, platform=platform.platform(),
                  processor=platform.processor(), seed=17, options=exact_dfs.normalize_options(),
                  limits=dict(workers=1, search_seconds=3, profile_seconds=2,
                              soft_memory_mb=2048, extra_process_seconds=8, wall_seconds=90),
                  interpretation='Equivalent support operation; finite timings are exploratory and not time-to-480 evidence.',
                  non_claims=['A timeout is inconclusive.', 'Desktop timing variability remains.',
                              'No canonical state/cache/network/GPU work.'], results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tasks = []
    for repeat in range(2):
        variants = ('inline-bit-length', 'precomputed') if repeat == 0 else ('precomputed', 'inline-bit-length')
        for case in ('generated-8', 'whole-five-clue'):
            tasks.extend((repeat, case, variant, False) for variant in variants)
    tasks.extend((0, 'whole-five-clue', variant, True) for variant in ('inline-bit-length', 'precomputed'))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
    for repeat, case, variant, profile in tasks:
        budget = 2 if profile else 3
        if time.monotonic() - started + budget + 8 > 90:
            report['wall_cap_stopped'] = True
            break
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--child', '--case', case,
                   '--variant', variant, '--seconds', str(budget)] + (['--profile'] if profile else [])
        beginning = time.monotonic()
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=budget+8)
            if completed.returncode:
                answer = dict(outcome='error', returncode=completed.returncode, stderr=completed.stderr[-4000:])
            else:
                answer = json.loads(completed.stdout)
        except subprocess.TimeoutExpired:
            answer = dict(outcome='external_timeout')
        answer.update(repeat=repeat, case=case, variant=variant, profile_enabled=profile,
                      total_process_seconds=time.monotonic()-beginning)
        report['results'].append(answer)
        save()
        print(json.dumps({key:answer[key] for key in ('repeat','case','variant','profile_enabled','outcome','branches','elapsed_seconds') if key in answer}), flush=True)
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-started,
                  source_sha256_after=hashes())
    report['sources_unchanged'] = before == report['source_sha256_after'] and all(r.get('source_sha256') == before for r in report['results'])
    save()
    if any(r['outcome'] in ('error', 'external_timeout') for r in report['results']):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
