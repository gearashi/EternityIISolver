"""Replay exact DFS on the bundled reference board's southeast 9x9 region."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exact_dfs import solve
from exact_model import check_solution, constraint_edges, from_bundle, problem_hash
from exact_worker import DEFAULT_SEED, PRUNING_PROFILES, dfs_profile_options
from validator import load_bundle, validate_board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PRUNING_PROFILES, default='baseline')
    parser.add_argument('--seconds', type=float, default=1020.0, help='DFS search budget in seconds')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    bundle = load_bundle(ROOT / 'data')
    parent = bundle.record_board
    checked = validate_board(parent, bundle)
    if not checked['valid'] or checked['score'] != 466:
        raise RuntimeError('Bundled reference failed verification')
    region = {16*r+c for r in range(7,16) for c in range(7,16)}
    original = from_bundle(bundle, board=parent, free_cells=region)
    faces = bundle.oriented_edges
    edges = tuple(e for e in constraint_edges(original)
                  if e[0] in region and e[1] in region
                  and faces[parent[e[0]]][e[2]] == faces[parent[e[1]]][e[3]])
    problem = replace(original, edges=edges)
    if not check_solution(problem, parent):
        raise RuntimeError('Bundled witness failed the regional constraints')
    result = solve(problem, seconds=args.seconds, seed=args.seed,
                   options=dfs_profile_options(args.profile))
    board = result['board']
    if result['outcome'] == 'infeasible' or (result['outcome'] == 'solved' and board is None):
        raise RuntimeError('Solver contradicted a known regional solution')
    if board is not None and not check_solution(problem, board):
        raise RuntimeError('Returned witness failed regional verification')
    print(json.dumps(dict(scope='southeast 9x9, reference-matched internal edges',
                          problem_sha256=problem_hash(problem), edges=len(edges),
                          profile=args.profile, seed=args.seed, seconds_limit=args.seconds,
                          outcome=result['outcome'], branches=result['branches'],
                          probe_trials=result['probe_checks'],
                          total_choices=result['branches']+result['probe_checks'],
                          elapsed_seconds=result['elapsed_seconds'],
                          witness_verified=board is not None), indent=2))


if __name__ == '__main__':
    main()
