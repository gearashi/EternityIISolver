"""Reproduce the bounded queue setup-cache ablation, without puzzle search.

The workload matches research/round2/queue-cache-setup-20261013.json: one schema
and checkpoint validation followed by three dispatch/accept task+hash passes
for each initial ticket. Each fixture checkpoint is made with a zero-second DFS
budget. Semantic DFS replay is unchanged and deliberately outside this setup-
only comparison. No network, GPU, canonical runtime or board-cache writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exact_dfs
import exact_parallel as parallel
from exact_model import from_bundle, problem_hash
from search_resources import peak_memory_mb
from validator import load_bundle

SOURCE_FILES = ('exact_parallel.py', 'exact_dfs.py', 'exact_model.py', 'validator.py',
                'search_resources.py', 'scripts/benchmark_queue_setup.py')


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def fixture(problem, seed, options):
    """Fresh deterministic cursors only: zero-second DFS never explores a node."""
    queue = parallel._new_queue(problem, seed, options)
    for ordinal, ticket in enumerate(queue['tickets']):
        task = parallel._ticket_problem(problem, queue['partition'], ticket['state'])
        answer = exact_dfs.solve(task, seconds=0, seed=parallel._ticket_seed(seed, ordinal), options=options)
        if answer['outcome'] != 'timeout' or answer['branches'] != 0:
            raise AssertionError('Zero-budget checkpoint fixture unexpectedly searched')
        ticket['checkpoint'] = answer['checkpoint']
        ticket['counters'] = {key: answer[key] for key in ticket['counters']}
    return parallel._seal(queue)


def trial(problem, sealed, seed, options, enabled):
    """Identical input/checkpoint bytes and task sequence in both modes."""
    started = time.monotonic()
    cache = parallel._TaskCache(problem, enabled=enabled)
    expected = parallel._new_queue(cache.problem, seed, options, task_cache=cache)
    validated = parallel._validate_queue(json.loads(json.dumps(sealed)), expected,
                                         cache.problem, semantic=False, task_cache=cache)
    identities = []
    for _ in range(3):
        for ordinal, ticket in enumerate(validated['tickets']):
            task = parallel._resolve_task(cache.problem, validated['partition'],
                                          ticket['state'], ticket['path'], cache)
            identity = cache.identity(ticket['state'], ticket['path'], task=task)
            checked = parallel._check_checkpoint(ticket['checkpoint'], task,
                parallel._ticket_seed(seed, ordinal), validated['options_hash'],
                identity=identity, hints_hash=cache.hint_identity)
            identities.append((identity, checked['checksum']))
    fingerprint = hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest()
    return {'cached': enabled, 'seconds': time.monotonic() - started,
            'setup_cache': cache.stats(), 'fingerprint': fingerprint,
            'peak_memory_mb': peak_memory_mb()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--wall-seconds', type=float, default=40)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Use a new output path to preserve previous evidence')
    if not 1 <= args.repeats <= 3 or not 0 < args.wall_seconds <= 40:
        parser.error('At most three paired repeats and 40 seconds are permitted')
    before = hashes()
    bundle = load_bundle()
    problem = from_bundle(bundle)
    options = exact_dfs.normalize_options()
    seed = 17
    sealed = fixture(problem, seed, options)
    report = {'schema': 1, 'utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'platform': platform.platform(),
              'problem_sha256': problem_hash(problem), 'source_sha256': before,
              'seed': seed, 'options': options, 'root_tickets': len(sealed['tickets']),
              'fixture': sealed,
              'fixture_sha256': hashlib.sha256(parallel._canonical(sealed)).hexdigest(),
              'bundle_input_sha256': bundle.metadata['input_sha256'],
              'setup_sequence': 'One queue schema/checkpoint validation, then three task+identity resolutions per pending ticket, modeling dispatch/accept/setup reuse. Semantic DFS replay unchanged and omitted from these setup-only measurements.',
              'limits': {'repeats': args.repeats, 'max_wall_seconds': args.wall_seconds,
                         'workers': 1, 'soft_memory_mb': 1024, 'dfs_fixture_seconds': 0,
                         'setup_reuse_passes': 3, 'gpu_count': 0},
              'limit_scope': 'Wall/memory limits are polled between the six-or-fewer setup trials. Each trial is bounded by the official 16x16 puzzle, four initial tickets and three reuse passes. Fixture construction occurs before the measured wall budget and only requests zero-second DFS checkpoints.',
              'non_claims': ['Setup reuse does not change pruning, frontier coverage or time to solve Eternity II.',
                            'Exploratory order-reversed desktop timing, no CPU search or GPU work.',
                            'No canonical runtime writes or network.'], 'trials': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    expected_fingerprint = None
    for repeat in range(args.repeats):
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            if time.monotonic() - started > args.wall_seconds or peak_memory_mb() > 1024:
                raise RuntimeError('Setup benchmark resource limit')
            result = trial(problem, sealed, seed, options, enabled)
            if expected_fingerprint is None:
                expected_fingerprint = result['fingerprint']
            if result['fingerprint'] != expected_fingerprint:
                raise AssertionError('Cache changed a task or checkpoint identity')
            report['trials'].append(dict(repeat=repeat, **result))
            print(json.dumps(dict(repeat=repeat, **result)), flush=True)
    report['source_sha256_after'] = hashes()
    report['sources_unchanged'] = report['source_sha256_after'] == before
    report['wall_seconds'] = time.monotonic() - started
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0 if report['sources_unchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
