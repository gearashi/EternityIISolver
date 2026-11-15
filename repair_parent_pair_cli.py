"""Rank board pairs using weights fitted to synthetic, independently scored repairs."""
from itertools import combinations, islice
import json

from repair_parent_learning import features, vector
from repair_policy import PairSelector

# The fitted coefficients are from scripts/benchmark_parent_learning.py, seeds
# 190..201. Each training target is the verified gain above the entire nine-
# board bank, divided by its edge count. No official board was used in fitting.
WEIGHTS = (-0.0062267516878746695, -0.03323085322896659, 0.13300425913534958,
           0.06966756759035149, 0.01997048015253168, -0.05558788147035692,
           0.0633858686695897, -0.10245388743190394, -0.046238918366636746,
           -0.010795223930967596)


def choose_pair(problem, boards):
    bank = list(islice(boards,65))
    if not 2 <= len(bank) <= 64:
        raise ValueError('Supply two to 64 boards')
    selector = PairSelector(problem)
    checked = [selector._parent(board) for board in bank]
    shape = [features(problem, row[0]) for row in checked]
    best_pair = None
    best_estimate = float('-inf')
    count = 0
    for i,j in combinations(range(len(bank)),2):
        if checked[i][0] == checked[j][0]:
            continue
        x = vector(problem, selector, checked[i], checked[j], shape[i], shape[j])
        estimate = sum(a*b for a,b in zip(WEIGHTS,x))
        count += 1
        if estimate > best_estimate:
            best_pair = i,j
            best_estimate = estimate
    if best_pair is None:
        raise ValueError('Supply at least two distinct legal boards')
    return dict(pair=best_pair, estimated_gain_fraction=best_estimate, pairs_scored=count,
                parents_checked=len(bank), bank_best=max(row[3] for row in checked))


def main(argv=None):
    import argparse
    from pathlib import Path

    from app_paths import resource_root
    from exact_model import from_bundle
    from smart_repair import cycle_recombine
    from validator import board_from_json, load_bundle, validate_board

    parser = argparse.ArgumentParser(description='Rank local boards and recombine one pair offline.')
    parser.add_argument('boards', nargs='+', type=Path)
    parser.add_argument('--seconds', type=float, default=2., help='time budget for the selected recombination')
    parser.add_argument('--data', type=Path)
    args = parser.parse_args(argv)
    if not 2 <= len(args.boards) <= 64 or not 0.01 <= args.seconds <= 3600:
        parser.error('supply 2–64 local files and a time budget from 0.01 to 3600 seconds')
    bundle = load_bundle(args.data if args.data else resource_root()/'data')
    problem = from_bundle(bundle)
    boards = [board_from_json(json.loads(path.read_text(encoding='utf-8-sig'))) for path in args.boards]
    try:
        picked = choose_pair(problem,boards)
    except ValueError as exc:
        parser.error(str(exc))
    i,j = picked['pair']
    result = cycle_recombine(problem,boards[i],boards[j],seconds=args.seconds,seed=0)
    verified = validate_board(result['board'],bundle)
    if not verified['valid'] or verified['score'] != result['score']:
        raise RuntimeError('official validation of the recombination failed')
    print(json.dumps({**picked,'pair_files':[str(args.boards[i]),str(args.boards[j])],
                      'child_score':result['score'],'gain_over_bank':max(0,result['score']-picked['bank_best']),
                      'branches':result.get('branches',0),'optimal_for_pair':bool(result['optimal']),
                      'board':result['board']}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
