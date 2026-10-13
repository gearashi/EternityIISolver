"""Bounded ablation of native bulk arc loading on identical CPU matrices."""
from pathlib import Path
import hashlib
import json
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exact_model import generated_problem, problem_hash
from gpu_assignment import cpu_cost_matrix
from smart_repair import independent_cells, _assignment_from_matrix


def main():
    start=time.monotonic();deadline=start+20
    def stopped(): return time.monotonic()>=deadline
    problem,answer=generated_problem(16,5,7,clue_count=2)
    rng=random.Random(7);matrices=[];input_hashes=[]
    for index in range(3):
        board=list(answer);groups={}
        for cell in range(256):
            if cell not in problem.fixed:
                groups.setdefault(sum(c==0 for c in problem.faces[board[cell]]),[]).append(cell)
        for cells in groups.values():
            pieces=[board[c]//4 for c in cells];rng.shuffle(pieces)
            for c,piece in zip(cells,pieces): board[c]=rng.choice([s for s in problem.domains[c] if s//4==piece])
        cells=independent_cells(problem,seed=7+index,parity=index if index<2 else None)
        matrices.append(cpu_cost_matrix(problem,board,cells,should_stop=stopped))
        input_hashes.append(hashlib.sha256(json.dumps([board,cells],separators=(',',':')).encode()).hexdigest())
    for bulk in (False,True):
        for matrix in matrices: _assignment_from_matrix(problem,matrix,bulk_arcs=bulk,should_stop=stopped)
    samples=[]
    for repeat in range(5):
        for index,matrix in enumerate(matrices):
            outcomes={}
            for bulk in ((False,True) if repeat%2==0 else (True,False)):
                before=time.perf_counter()
                result=_assignment_from_matrix(problem,matrix,bulk_arcs=bulk,should_stop=stopped)
                elapsed=time.perf_counter()-before
                if not result['optimal']: raise RuntimeError('Benchmark hit its 20-second limit')
                outcomes[bulk]=result
                samples.append({'repeat':repeat,'case':index,'bulk':bulk,'seconds':elapsed,'score':result['score']})
            if outcomes[False]['board']!=outcomes[True]['board']: raise RuntimeError('Bulk/scalar result mismatch')
    root=Path(__file__).resolve().parents[1]
    report={'schema_version':1,'synthetic':True,'purpose':'Isolate arc loading plus exact assignment, excluding matrix generation',
            'problem_sha256':problem_hash(problem),'input_sha256':input_hashes,'python':sys.version,
            'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('smart_repair.py','gpu_assignment.py', 'scripts/benchmark_assignment_arcs.py')},
            'sample_count':len(samples),'samples':samples,'elapsed_seconds':time.monotonic()-start,
            'all_boards_equal':True,'seconds_budget':20}
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
