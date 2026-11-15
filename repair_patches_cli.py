"""Combine compatible user-provided candidate boards as offline repair patches."""
import argparse
import json
from pathlib import Path

from app_paths import resource_root
from exact_model import from_bundle
from repair_patches import compose
from validator import load_bundle, board_from_json, validate_board


def main(argv=None):
    parser=argparse.ArgumentParser(description='Choose a compatible subset of local patch boards for one reference.')
    parser.add_argument('reference',type=Path)
    parser.add_argument('patches',nargs='+',type=Path)
    parser.add_argument('--seconds',type=float,default=2.)
    parser.add_argument('--data',type=Path,default=None)
    args=parser.parse_args(argv)
    if len(args.patches)>64:
        parser.error('at most 64 patch boards are supported')
    bundle=load_bundle(args.data if args.data else resource_root()/'data')
    def read(path):
        b=board_from_json(json.loads(path.read_text(encoding='utf-8-sig')))
        if not validate_board(b,bundle)['valid']:
            parser.error(f'board fails five-clue validation: {path.name}')
        return b
    reference=read(args.reference)
    result=compose(from_bundle(bundle),reference,[read(path) for path in args.patches],seconds=args.seconds)
    checked=validate_board(result['board'],bundle)
    if not checked['valid'] or checked['score']!=result['score']:
        raise RuntimeError('official validator rejected combined patches')
    print(json.dumps(result))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
