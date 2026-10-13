"""Actual-board paired ablation of immutable validation-view reuse; max60s."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import time
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exact_model import from_bundle, problem_hash
from gpu_assignment import AssignmentEvaluator
from smart_repair import assignment_repair_batch, independent_cells
from validator import load_bundle, validate_board


def main():
    started=time.monotonic();deadline=started+60
    def stopped(): return time.monotonic()>=deadline
    root=Path(__file__).resolve().parents[1]
    files=('gpu_assignment.py','smart_repair.py','exact_model.py','scripts/benchmark_assignment_validation.py')
    sources={name:(root/name).read_bytes() for name in files}
    source_hashes={name:hashlib.sha256(data).hexdigest() for name,data in sources.items()}
    bundle=load_bundle();problem=from_bundle(bundle)
    jobs=[(bundle.record_board,independent_cells(problem,i,parity=i if i<2 else None)) for i in range(16)]
    job_hash=hashlib.sha256(json.dumps(jobs,separators=(',',':')).encode()).hexdigest()
    oracle_evaluator=AssignmentEvaluator(problem,backend='cpu',should_stop=stopped)
    oracle=assignment_repair_batch(problem,jobs,evaluator=oracle_evaluator,reuse_validation=False)
    samples=[]
    with tempfile.TemporaryDirectory(prefix='eternity-assignment-validation-') as cache:
        evaluator=AssignmentEvaluator(problem,backend='cuda',cache_dir=cache,should_stop=stopped)
        # Compiler/driver setup and both paths are warm before paired timing.
        for reuse in (False,True):
            assignment_repair_batch(problem,jobs,evaluator=evaluator,reuse_validation=reuse)
        for repeat in range(3):
            outputs={}
            for reuse in ((False,True) if repeat%2==0 else (True,False)):
                before=time.perf_counter()
                result=assignment_repair_batch(problem,jobs,evaluator=evaluator,reuse_validation=reuse)
                elapsed=time.perf_counter()-before
                if not all(item['optimal'] for item in result): raise RuntimeError('Benchmark hit its60-second limit')
                if [item['board'] for item in result]!=[item['board'] for item in oracle]:
                    raise RuntimeError('GPU validation ablation changed a board versus CPU reference')
                reports=[validate_board(item['board'],bundle) for item in result]
                if not all(report['valid'] and report['score']==item['score'] for report,item in zip(reports,result)):
                    raise RuntimeError('Actual puzzle validator rejected a benchmark board')
                outputs[reuse]=result
                samples.append({'repeat':repeat,'reuse_validation':reuse,'seconds':elapsed,
                                'scores':[item['score'] for item in result]})
        device=dict(evaluator.stats())
    if source_hashes!={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}:
        raise RuntimeError('Source changed during benchmark')
    report={'schema_version':1,'synthetic':False,'purpose':'Paired complete CUDA cost+CPU matching batch with repeated model validation versus immutable structural view',
            'problem_sha256':problem_hash(problem),'jobs_sha256':job_hash,'input_score':466,'neighborhoods_per_batch':16,
            'python':sys.version,'source_sha256':source_hashes,'samples':samples,
            'all_boards_equal_to_cpu_reference':True,'all_outputs_independently_validated':True,
            'seconds_budget':60,'elapsed_seconds':time.monotonic()-started,**device}
    destination=root/'docs/evidence/assignment-validation-20261013.json'
    destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(destination.with_suffix('.sources.zip'),'w',compression=zipfile.ZIP_DEFLATED) as snapshot:
        for name,data in sources.items(): snapshot.writestr(name,data)
    print(json.dumps({'saved':str(destination),'elapsed_seconds':report['elapsed_seconds'],
                      'samples':[dict(repeat=s['repeat'],reuse_validation=s['reuse_validation'],seconds=s['seconds']) for s in samples],
                      'cost_backend':device['cost_backend'],'device':device['cost_device']},indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
