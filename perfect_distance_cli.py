"""Inspect a certified necessary distance from one legal board to 480."""
import argparse
import json
from pathlib import Path

from exact_model import from_bundle, problem_hash
from library_cache import local_board_hash
from perfect_distance import DistanceBounds, packing, verify_packing
from regional_distance import DonorDistance
from regional_distance_replay import verify_exact_packing
from validator import board_from_json,load_bundle,validate_board


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('board',type=Path,help='Board JSON exported with oriented cell states')
    parser.add_argument('--method', choices=('edges','independent'), default='edges',
                        help='Check donors and internal edges, or use independent cell supports')
    args=parser.parse_args(argv)
    bundle=load_bundle()
    try:
        board=board_from_json(json.loads(args.board.read_text(encoding='utf-8-sig')))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        parser.error(str(exc))
    validation=validate_board(board,bundle)
    if not validation['valid']:
        parser.error('Input violates a physical piece, frame or official clue constraint')
    p=from_bundle(bundle)
    bounds=DistanceBounds(p,board)
    cuts=bounds.cuts()
    work=dict(initial_subsets=bounds.subsets,initial_state_visits=bounds.setup_states,
              initial_edge_checks=bounds.setup_edges)
    if args.method=='edges':
        stronger=DonorDistance(p,board)
        improved=[]
        for region,minimum in cuts:
            checked,_=stronger.improve(region,minimum,level='edges',extra=3,max_work=100000)
            improved.append((region,checked))
        cuts=improved
        work.update(donor_setup=stronger.setup,donor_work=stronger.work,
                    donor_subsets=stronger.subsets,donor_placements=stronger.placements)
    rows,branches,status=packing(cuts,len(board),scale=16,seconds=4.)
    result=(verify_exact_packing(p,board,rows,16) if args.method=='edges'
            else verify_packing(p,board,rows,16))
    print(json.dumps(dict(target_score=480,parent_score=validation['score'],
                          units='changed cell states including rotations',problem_hash=problem_hash(p),
                          parent_local_sha256=local_board_hash(board),
                          regional_certificates=rows,method=args.method,work=work,packing_solver=status,packing_branches=branches,
                          **result),indent=2))
    return 0
