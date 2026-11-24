"""Compare exhaustive tiny-fixture GPU traversal with the shipped C++ search."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def check(args, case, size, clues, profile, budget):
    import numpy as np
    from cuda_score_search import Search
    from exact_model import generated_problem
    from fast_search import _piece_text, _profile_arguments, score_board
    problem, _ = generated_problem(size, 4, 13, clue_count=clues)
    output = args.out / case
    output.mkdir()
    pieces = output / 'pieces.txt'
    pieces.write_text(_piece_text(problem),encoding='ascii')
    lanes = 4
    seed = 20261124
    command = [str(args.helper.resolve()),'--pieces',str(pieces),'--size',str(size),'--out',str(output),
               '--seed',str(seed),'--threads',str(lanes),'--count','--keep','0']
    for cell,state in problem.fixed.items():
        command += ['--clue',f'{cell}:{state}']
    command += _profile_arguments(profile,size**2,budget)
    print('CMD',subprocess.list2cmdline(command),flush=True)
    done = subprocess.run(command,capture_output=True,text=True,timeout=90)
    print('EXIT',done.returncode,'stdout',done.stdout,'stderr',done.stderr,flush=True)
    assert done.returncode in (0,3)
    expected = [dict() for _ in range(lanes)]
    for path in output.glob('board_*.json'):
        record=json.loads(path.read_text())
        expected[record['thread']][record['thread_nodes']]=(tuple(record['board']),record['claimed_score'])
    status=json.loads((output/'status.json').read_text())
    assert sum(map(len,expected)) == status['complete_boards']
    search=Search(problem,profile,lanes,seed)
    search.reset(budget)
    actual=[dict() for _ in range(lanes)]
    for _ in range(20000):
        flags,_=search.step(137)
        for lane in np.flatnonzero(flags==2):
            board=search.board[:,int(lane)].get().astype(int).tolist()
            breaks=int(search.broken[-1,int(lane)].get())
            score=2*size*(size-1)-breaks
            assert score_board(problem, board) == score
            nodes=int(search.nodes[int(lane)].get())
            actual[lane][nodes]=(tuple(board),score)
            search.status[int(lane)] = 0  # continue siblings to full exhaustion
        if np.all(flags==1):
            break
    else:
        raise AssertionError('Tiny search did not exhaust')
    assert expected==actual, 'CPU and GPU complete-board trace differs'
    nodes=search.nodes.get().tolist()
    assert sum(nodes)==status['nodes'] and status['max_depth']==int(search.deepest.max().get())
    print('PASS',case,'CPU/GPU full traversal identical complete boards, scores and placement ordinals;',
          'complete per lane',list(map(len,actual)),'placements per lane',nodes,'max depth',status['max_depth'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--helper',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.repo.resolve()))
    os.environ['CUPY_CACHE_DIR']=str((args.out/'kernel-cache').resolve())
    base=dict(breaks=(3,6,7),depletion_colors=(),depletion_schedule=(),restart_nodes=0)
    check(args,'strict',3,1,base,0)
    check(args,'score',3,1,dict(base,depletion_colors=(1,2),depletion_schedule=((2,0),(6,3))),2)
