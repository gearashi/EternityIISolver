"""Bounded, offline CUDA fast-score experiment. Run with the source CUDA environment."""
import time
START = time.perf_counter()
import argparse
import hashlib
import json
from pathlib import Path
import sys


def official(args):
    from cuda_score_search import run_ladder
    from exact_model import from_bundle
    from fast_search import default_score_profile
    from validator import load_bundle, validate_board
    bundle = load_bundle()
    problem = from_bundle(bundle)
    profile = default_score_profile(problem)
    boards = []

    def save(candidate):
        board = candidate['board']
        check = validate_board(board, bundle)
        if not check['valid'] or check['score'] != candidate['score']:
            raise RuntimeError('CUDA benchmark candidate failed the release validator')
        sha = hashlib.sha256(json.dumps(board, separators=(',', ':')).encode()).hexdigest()
        record = dict(candidate, seed=args.seed, sha256=sha)
        (args.out / f'board-{check["score"]}-{sha}.json').write_text(json.dumps(record, indent=2)+'\n')
        boards.append({k:v for k,v in record.items() if k!='board'})

    result = run_ladder(problem, profile, args.seed, args.lanes, START, args.seconds,
                        args.out / 'kernel-cache', save)
    nodes = result['branches']
    kernel_seconds = result['search_kernel_seconds']
    return dict(engine='experimental-cuda-fast-score', seed=args.seed, lanes=args.lanes,
                device=result['gpu'], profile=profile['name'], requested_seconds=args.seconds,
                placements=nodes, placements_per_wall_second=result['nodes_per_second'],
                placements_per_search_kernel_second=nodes/kernel_seconds if kernel_seconds else None,
                deepest_placement=result['max_depth'], best_score=result['score'], boards=boards,
                **{key:result[key] for key in ('elapsed_seconds', 'initialization_seconds', 'order_kernel_seconds',
                   'search_kernel_seconds','max_search_launch_ms','max_order_launch_ms','total_launches',
                   'stages','restarts','complete_boards')})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--seconds',type=float,required=True)
    parser.add_argument('--lanes',type=int,default=2048)
    args=parser.parse_args()
    if not 0 <= args.seed < 2**64 or not 1 <= args.lanes <= 4096 or not 0 < args.seconds <= 86400:
        parser.error('seed must fit uint64, lanes must be in 1..4096, seconds in (0,86400]')
    args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.repo.resolve()))
    summary=official(args)
    text=json.dumps(summary,indent=2)
    (args.out/'summary.json').write_text(text+'\n')
    print(text)
