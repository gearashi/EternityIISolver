"""Bounded offline measurement of the fast compiled engines on the five-clue puzzle.

Runs fast-dfs and then fast-score in fresh temporary state, each for a fixed
time, and records speed, depth and the time at which each better complete
board appeared. Every board is checked with the release validator. Nothing
is uploaded and no saved search is touched.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fast_search
from exact_model import from_bundle, problem_hash
from validator import load_bundle, validate_board


def processor():
    try:
        for line in Path('/proc/cpuinfo').read_text(encoding='utf-8', errors='replace').splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def measure(problem, bundle, mode, seconds, workers, seed, helper):
    timeline, last = [], {}
    with tempfile.TemporaryDirectory(prefix='eternity-fast-benchmark-') as state:
        started = time.monotonic()

        def on_progress(progress):
            last.update(progress)
            score = progress.get('engine_best_score')
            if type(score) is int and (not timeline or score > timeline[-1]['score']):
                timeline.append({'score': score, 'seconds': round(time.monotonic() - started, 1),
                                 'placements': progress.get('branches')})

        result = fast_search.solve(problem, seconds, seed=seed, workers=workers, mode=mode, state_dir=state,
                                   helper=helper, on_progress=on_progress)
        elapsed = time.monotonic() - started
        boards = []
        for path in sorted((Path(state) / 'boards').glob('*.json')) if mode == 'score' else ():
            record = json.loads(path.read_text(encoding='utf-8'))
            report = validate_board(record['board'], bundle)
            if not report['valid'] or report['score'] != record['score']:
                raise RuntimeError(f'Benchmark board failed release validation: {path.name}')
            boards.append({'score': report['score'], 'board': record['board'],
                           'sha256': hashlib.sha256(json.dumps(record['board'], separators=(',', ':')).encode()).hexdigest()})
    boards.sort(key=lambda item: item['score'])
    summary = {'mode': mode, 'outcome': result['outcome'], 'requested_seconds': seconds, 'elapsed_seconds': round(elapsed, 2),
               'workers': workers, 'seed': seed, 'placements': result['branches'],
               'placements_per_second': round(result['branches'] / elapsed) if elapsed else None,
               'deepest_placement': result['max_depth']}
    if mode == 'score':
        summary.update(profile=result.get('profile'), complete_boards=result['complete_boards'], restarts=result['restarts'],
                       best_score=result['engine_best_score'], final_target_score=result.get('target_score'),
                       best_score_timeline=timeline, validated_boards=boards[-3:],
                       all_validated_scores=[item['score'] for item in boards])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--strict-seconds', type=float, default=60)
    parser.add_argument('--score-seconds', type=float, default=600)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    helper = fast_search.helper_path()
    if not helper.is_file():
        raise SystemExit('Build the helper first: python scripts/build_fast_helper.py')
    bundle = load_bundle()
    problem = from_bundle(bundle)
    report = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'full 16x16 puzzle, five fixed clues, all 480 edges', 'problem_sha256': problem_hash(problem),
              'helper': fast_search.HELPER_VERSION, 'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
              'processor': processor(), 'logical_cpus': os.cpu_count(), 'system': platform.platform(),
              'python': platform.python_version(), 'reference_score': 466,
              'runs': [measure(problem, bundle, 'strict', args.strict_seconds, args.workers, args.seed, helper),
                       measure(problem, bundle, 'score', args.score_seconds, args.workers, args.seed, helper)],
              'interpretation': 'Bounded timings on one machine. A time limit is inconclusive; score search is heuristic. '
                                'No run here reached or exceeded the 466 reference unless best_score says so.'}
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
