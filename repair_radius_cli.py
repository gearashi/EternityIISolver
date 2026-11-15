"""Offline command-line interface for bounded one- through eight-cell score repair."""
import argparse
import json
from pathlib import Path

from app_paths import resource_root
from exact_model import from_bundle
from repair_radius import search as search_pairs
from repair_triples import search as search_triples
from validator import board_from_json, load_bundle, validate_board


def main(argv=None):
    parser=argparse.ArgumentParser(description='Find a strict gain or exhaust the requested chosen changed-cell neighborhood.')
    parser.add_argument('board',type=Path,help='Explicit JSON input with oriented cell states')
    parser.add_argument('--radius',type=int,choices=(1,2,3,4,5,6,7,8),default=2)
    parser.add_argument('--data',type=Path,default=None,help='Override the bundled official pieces and clues')
    parser.add_argument('--max-work',type=int,default=None,help='Optional counted work cap')
    parser.add_argument('--seconds',type=float,default=120,help='Total time budget for native radii 4 through 8')
    parser.add_argument('--helper',type=Path,help='Explicit path to a trusted compiled radius helper')
    args=parser.parse_args(argv)
    bundle=load_bundle(args.data if args.data is not None else resource_root()/'data')
    board=board_from_json(json.loads(args.board.read_text(encoding='utf-8-sig')))
    if not validate_board(board,bundle)['valid']:
        parser.error('board violates official piece, frame or five-clue constraints')
    if args.max_work is not None and args.max_work<0:
        parser.error('--max-work must be nonnegative')
    p=from_bundle(bundle)
    if args.radius>=4:
        from repair_native_radius import search
        result=search(p,board,radius=args.radius,max_work=args.max_work,seconds=args.seconds,helper=args.helper)
    elif args.radius==3:
        result=search_triples(p,board,max_work=args.max_work)
    else:
        result=search_pairs(p,board,radius=args.radius,max_work=args.max_work)
    verified=validate_board(result['board'],bundle)
    if not verified['valid'] or verified['score']!=result['score']:
        raise RuntimeError('official board verifier rejected radius result')
    print(json.dumps(result))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
