"""Assemble and audit existing repair benchmarks; this command never runs a solver.

The v2 manifest records THIS evidence audit. Earlier benchmark executions remain
unchanged primary input reports, with their original commands, timing and guards.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import zipfile

PREFIX = 'repair-candidates-20261020'
RISKS = [
    'Finite local experiments do not establish time to reach 467 or 480, nor impossibility of this puzzle.',
    'A separate approximately 0.81-second CPU diagnostic briefly overlapped portfolio seed 17 near the pairing/adaptive boundary; exact overlap is not recorded. Treat small timing differences as inconclusive.',
    'Only two portfolio seeds and one parent bank were measured; parent-cycle counts and neighborhood throughput are proxies, not solution-quality improvements.',
    'Wall-clock budgets and the 2048 MiB host cap were cooperative, not OS process memory or CPU-affinity limits. Each child also had an external 75-second timeout.',
    'Socket auditing rejected Python connection attempts; it is not an operating-system firewall or proof about arbitrary native-library networking.',
    'Archive-source snapshots identify measured bytes. They can differ from subsequent release defaults or packaging edits, which are not rerun by this audit.',
    'The structured-repair baseline disables the new candidate flags, while retaining existing portfolio operators; it is not an execution of the previous released binary.',
    'This audit pins existing primary evidence, rather than retroactively claiming its packaging command originally ran those experiments.'
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    require(path.stat().st_size < 64 * 1024**2, f'Unbounded evidence file: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def stage(source, destination):
    source = source.resolve()
    content = source.read_bytes()
    if destination.exists():
        require(destination.read_bytes() == content, f'Refusing to replace different evidence: {destination}')
    else:
        destination.write_bytes(content)
    require(digest(source) == digest(destination), 'Staged evidence hash mismatch')
    return destination


def board_hash(board):
    return hashlib.sha256(json.dumps(board, separators=(',', ':')).encode()).hexdigest()


def board_validator(pieces_bytes, puzzle_bytes):
    metadata = json.loads(puzzle_bytes)
    require(metadata['size'] == 16, 'Unsupported puzzle size')
    require(metadata['input_sha256']['pieces.txt'] == hashlib.sha256(pieces_bytes).hexdigest(), 'Piece-data hash mismatch')
    pieces = [tuple(map(int, line.split())) for line in pieces_bytes.decode().splitlines() if line.strip()]
    require(len(pieces) == 256 and all(len(p) == 4 for p in pieces), 'Invalid piece count')
    faces = []
    for u, d, l, r in pieces:
        faces.extend(((u,r,d,l),(l,u,r,d),(d,l,u,r),(r,d,l,u)))
    fixed = {x['cell']:x['state'] for x in metadata['fixed_clues']}
    require(fixed == {34:831,45:1019,135:554,210:723,221:992}, 'Unexpected official clues')
    def score(board):
        require(type(board) is list and len(board) == 256, 'Invalid board size')
        require(all(type(s) is int and 0 <= s < 1024 for s in board), 'Invalid board state')
        require(len({s//4 for s in board}) == 256, 'Repeated physical piece')
        require(all(board[c] == s for c,s in fixed.items()), 'Changed fixed clue')
        oriented = [faces[s] for s in board]
        for c, sides in enumerate(oriented):
            outside = (c < 16, c % 16 == 15, c >= 240, c % 16 == 0)
            require(all((color == 0) == exterior for color,exterior in zip(sides,outside)), 'Invalid border/corner placement')
        return sum(oriented[c][1] == oriented[c+1][3] for c in range(256) if c%16 < 15) + sum(oriented[c][2] == oriented[c+16][0] for c in range(240))
    return score


def verify_report(report, expected_sources, fixture_sha, archive, score_board, parent_records):
    require(report['schema'] == 'eternity-repair-candidates-benchmark/v1', 'Unexpected raw report schema')
    require(report.get('passed') is True and bool(report.get('finished_utc')), 'Benchmark did not finish successfully')
    require(report['source_sha256_before'] == report['source_sha256_after'] == expected_sources, 'Benchmark source hashes differ')
    require(report['protected_files_sha256_before'] == report['protected_files_sha256_after'], 'Protected files changed during benchmark')
    require(report['fixture_sha256'] == fixture_sha, 'Different parent fixtures')
    require(report['parents'] == parent_records, 'Parent records do not match independently verified fixture')
    require(all(report[x] is True for x in ('sources_unchanged','protected_files_unchanged','fixture_unchanged')), 'Failed benchmark preservation guard')
    require(report['network_attempts'] == [], 'Benchmark attempted network access')
    for name, observed in expected_sources.items():
        require(hashlib.sha256(archive.read(name)).hexdigest() == observed, f'Archived source differs: {name}')
    actual = {(c['seed'],c['case']) for c in report['results']}
    expected = {(seed,case) for seed in report['seeds'] for case in report['cases']}
    require(actual == expected and len(actual) == len(report['results']), 'Missing/duplicate benchmark cases')
    embedded_boards = 0
    def walk(value):
        nonlocal embedded_boards
        if isinstance(value, dict):
            if isinstance(value.get('board'), list) and 'score' in value:
                require(score_board(value['board']) == value['score'], 'Reported final board score is incorrect')
                embedded_boards += 1
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for case in report['results']:
        require(case['passed'] is True and case['result']['passed'] is True and case['exit_code'] == 0, 'A child failed')
        require(case['source_sha256_before'] == case['source_sha256_after'] == expected_sources, 'Child source hashes differ')
        require(case['fixture_sha256_before'] == case['fixture_sha256_after'] == fixture_sha, 'Child parent fixture changed')
        require(case['network_attempts'] == [] and case['resource_interruptions'] == [], 'Child networking/resource interruption')
        require(case['command'], 'Missing original child argv')
        walk(case['result'])
    return embedded_boards


def compact_direct(report):
    result = {}
    for child in report['results']:
        case = child['case']; data = child['result']
        if case == 'validation':
            timing = {mode: statistics.median(row['seconds'] for row in data['records'] if row['mode']==mode) for mode in ('rebuild','reuse')}
            result[case] = dict(valid_cases=data['valid_cases'],invalid_cases=data['invalid_cases'],median_seconds=timing,
                                observed_microbenchmark_ratio=timing['rebuild']/timing['reuse'],records=data['records'])
        elif case == 'ties':
            result[case] = {k:data[k] for k in ('all_matrices_equal','all_primary_optima_equal','all_repairs_valid','distinct_equal_score_results_per_neighborhood','records','device_stats')}
        elif case == 'pairing':
            result[case] = {}
            for mode in ('random','structured'):
                rows = [x for x in data['records'] if x['mode']==mode]
                result[case][mode] = {
                    'active_cycles':[x['features']['active_cycles'] for x in rows],
                    'mean_active_cycles':statistics.mean(x['features']['active_cycles'] for x in rows),
                    'parent_scores':[x['features']['parent_scores'] for x in rows],
                    'result_scores':[x['score'] for x in rows],
                    'gains_over_better_parent':[x['score']-x['reference_score'] for x in rows],
                    'selection_seconds':sum(x['selection_seconds'] for x in rows),
                    'cycle_seconds':sum(x['cycle_seconds'] for x in rows)}
        elif case.startswith('regions-'):
            result[case] = {k:data[k] for k in ('score','calls','elapsed_seconds','region_seconds','policy_state')}
    return result


def portfolio_rows(report):
    rows = []
    for child in report['results']:
        data = child['result']; run = data['result']
        rows.append(dict(seed=child['seed'],mode=data['mode'],flags=data['flags'],score=run['score'],
                         neighborhoods=run['neighborhoods'],recombinations=run['recombinations'],
                         elapsed_seconds=run['elapsed_seconds'],region_calls=run.get('region_calls'),
                         region_improvements=run.get('region_improvements'),region_seconds=run.get('region_seconds'),
                         warmup_seconds=child['warmup']['seconds'],cost_backend=run['cost_backend']))
    for row in rows:
        base = next(x for x in rows if x['seed']==row['seed'] and x['mode']=='baseline')
        row['neighborhood_ratio_to_same_seed_baseline'] = row['neighborhoods']/base['neighborhoods'] if base['neighborhoods'] else None
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--direct',type=Path,required=True)
    parser.add_argument('--portfolio',type=Path,required=True)
    parser.add_argument('--opencl',type=Path,required=True)
    parser.add_argument('--checkpoint-log',type=Path)
    parser.add_argument('--fixture','--parents',dest='fixture',type=Path,help='Shared committed fixture for a portable audit replay')
    parser.add_argument('--regressions',type=Path,help='Completed separate source regression evidence JSON')
    parser.add_argument('--audit-output',type=Path,help='Fresh summary path inside the repository for replay; manifest uses a sibling .manifest.json path')
    args = parser.parse_args(argv)
    repo = args.repo.resolve(); destination = repo/'docs/evidence'
    require(destination.is_dir(), 'Evidence directory must already exist')
    raw_inputs = [(args.direct,'repair-candidates-20261020.json'),
                  (args.portfolio,'repair-candidate-portfolios-20261020.json'),
                  (args.opencl,'repair-candidates-opencl-20261020.json')]
    for source,name in raw_inputs:
        stage(source,destination/name)
    fixtures = [args.fixture] if args.fixture else [source.with_name(source.name+'.artifacts')/'parents.json' for source,_ in raw_inputs]
    if not args.fixture and not all(path.is_file() for path in fixtures):
        fixtures = [destination/(PREFIX+'.parents.json')]
    require(len({digest(path) for path in fixtures}) == 1, 'Original benchmark fixtures differ')
    staged_fixture = stage(fixtures[0],destination/(PREFIX+'.parents.json'))
    staged_script = stage(Path(__file__),destination/'repair-candidates-audit-20261020.py')
    source_zip = destination/(PREFIX+'.sources.zip')
    paths = [destination/name for _,name in raw_inputs] + [staged_fixture,source_zip,staged_script]
    if args.checkpoint_log:
        paths.append(stage(args.checkpoint_log,destination/'repair-candidates-checkpoint-20261020.log'))
    regression_path = stage(args.regressions,destination/'release-regressions-20261020.json') if args.regressions else None
    if regression_path:
        paths.append(regression_path)
    result_path = args.audit_output.resolve() if args.audit_output else destination/'repair-candidates-audit-20261020.json'
    require(result_path.is_relative_to(repo) and result_path.parent.is_dir(), 'Audit output must be in an existing repository directory')
    manifest_path = result_path.with_name(result_path.stem+'.manifest.json') if args.audit_output else destination/'repair-candidates-manifest.json'
    require(not result_path.exists() and not manifest_path.exists(), 'Audit outputs must be fresh')
    declared_results = [result_path.relative_to(repo).as_posix()]
    started_at = datetime.now(timezone.utc).isoformat(); started = time.monotonic()
    before = {path:digest(path) for path in paths}
    network = []
    def deny_network(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            network.append(event)
            raise PermissionError('Evidence audit is offline')
    sys.addaudithook(deny_network)
    direct,portfolio,opencl = [read_json(destination/name) for _,name in raw_inputs]
    fixture = read_json(staged_fixture)
    with zipfile.ZipFile(source_zip) as archive:
        require(len(archive.infolist()) <= 64 and sum(x.file_size for x in archive.infolist()) < 16*1024**2, 'Unbounded source archive')
        score_board = board_validator(archive.read('data/pieces.txt'),archive.read('data/puzzle.json'))
        parent_records = [{'sha256':board_hash(board),'score':score_board(board)} for board in fixture['parents']]
        require(len(parent_records)==32 and len({x['sha256'] for x in parent_records})==32, 'Expected 32 distinct frozen parents')
        verified_boards = sum(verify_report(report,direct['source_sha256_before'],digest(staged_fixture),archive,score_board,parent_records) for report in (direct,portfolio,opencl))
    regression_summary = None
    if regression_path:
        regressions = read_json(regression_path)
        require(regressions.get('passed') is True, 'Separate regression run did not pass')
        require(regressions['source_sha256_before'] == regressions['source_sha256_after'], 'Regression source hashes changed during tests')
        require(regressions['commands'] and all(command['exit_code'] == 0 for command in regressions['commands']), 'A regression command failed')
        require(any('test_repair_checkpoint' in command['argv'] and 'test_repair_continuation' in command['argv'] for command in regressions['commands']), 'Checkpoint regression coverage missing')
        regression_summary = {'path':regression_path.relative_to(repo).as_posix(), 'passed':True,
                              'source_files_with_equal_before_after_hashes':len(regressions['source_sha256_before']),
                              'commands':[{'argv':command['argv'],'exit_code':command['exit_code'],'seconds':command['seconds']} for command in regressions['commands']],
                              'benchmark_source_differences':{name:{'benchmark':value,'regression':regressions['source_sha256_before'][name]} for name,value in direct['source_sha256_before'].items() if name in regressions['source_sha256_before'] and value != regressions['source_sha256_before'][name]},
                              'scope':'Separate finite functional regression suite. Source hashes are the observations in that run; this audit does not rerun tests.'}
    direct_summary = compact_direct(direct); rows = portfolio_rows(portfolio)
    require(set(portfolio['seeds']) == {17,41} and len(rows)==12, 'Expected two-seed six-mode portfolio')
    require(opencl['backend']=='opencl' and opencl['cases']==['ties'], 'Unexpected OpenCL follow-up scope')
    artifact = {
        'schema':'eternity-repair-candidates-evidence-audit/v1','passed':True,
        'scope':'Audit of preserved benchmark evidence, not another solver run',
        'source_files_verified':len(direct['source_sha256_before']),
        'parent_boards_independently_validated':len(parent_records),
        'embedded_result_boards_independently_validated':verified_boards,
        'direct':direct_summary,'portfolio':rows,'opencl':compact_direct(opencl),'separate_regressions':regression_summary,
        'decisions':{
            'reuse_structure':{'default':True,'reason':'Identical finite validity outcomes and lower validation cost; portfolio neighborhood counts are reported separately.'},
            'neutral_ties':{'default':False,'reason':'Correct equal-primary objectives, but no distinct results on these elite neighborhoods and no demonstrated score gain; experimental API only.'},
            'pair_selection':{'default':False,'reason':'More permutation-cycle choices, but selected lower-scoring parents and no demonstrated score gain; experimental API only.'},
            'adaptive_regions':{'default':False,'reason':'Exploration and bounded allocations function, but both region runs retained 466; search benefit inconclusive; experimental API only.'},
            'checkpoint_resume':{'default':True,'reason':'Separate finite continuation and rejection regression tests; no speed-to-solution claim from these benchmarks.',
                                 'separate_test_report':regression_path.relative_to(repo).as_posix() if regression_path else None,
                                 'separate_test_log':'docs/evidence/repair-candidates-checkpoint-20261020.log' if args.checkpoint_log else None}},
        'original_experiments':[{'report':name,'started_at':report['started_utc'],'finished_at':report['finished_utc'],
                                 'elapsed_seconds':report['elapsed_seconds'],'limits':report['limits'],'seeds':report['seeds'],'backend':report['backend']}
                                for (_,name),report in zip(raw_inputs,(direct,portfolio,opencl))],
        'residual_risks':RISKS,'network_attempts':network}
    after = {path:digest(path) for path in paths}
    require(before == after and not network, 'Audit input or network preservation failed')
    result_path.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    run_git = subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True)
    dirty = bool(subprocess.run(['git','status','--porcelain'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip())
    manifest = {
        'schema_version':2,'claim_id':'EII-REPAIR-CANDIDATE-EVIDENCE-AUDIT-20261020',
        'repository':{'commit':run_git.stdout.strip(),'dirty':dirty},
        'command':list(sys.orig_argv),
        'environment':{'software':[{'name':'Python','version':platform.python_version()},{'name':'platform','version':platform.platform()}],
                       'hardware':platform.machine()+'; '+platform.processor()},
        'mathematics':{
            'assertion_tested':'Preserved reports have unchanged benchmark input/source/protected hashes, archived measured source bytes, matching public parent fixtures, and independently legal complete boards with their stated scores; compact metrics are derived from the raw observations.',
            'coefficient_domain':'Exact integers for states, edge colors and score; measured floating-point wall-clock durations.',
            'conventions':'16x16 row-major board; state=4*(piece ID-1)+clockwise rotation; outside edges gray; all five oriented clues fixed; 480 adjacency edges. Manifest run describes the evidence audit, while original benchmark commands and timings remain in input reports.',
            'inputs':[path.relative_to(repo).as_posix() for path in paths],
            'bounds':{'source_files':len(direct['source_sha256_before']),'frozen_parents':32,'direct_cases':len(direct['results']),
                      'portfolio_cases':len(portfolio['results']),'opencl_cases':len(opencl['results']),
                      'original_experiment_limits':[x['limits'] for x in artifact['original_experiments']]},
            'non_claims':RISKS},
        'randomness':{'used':False,'generator':'No random draws in the audit; original experiment seeds are preserved in reports.','seed':None},
        'run':{'started_at':started_at,'runtime_seconds':time.monotonic()-started,'exit_status':0,'status':'completed',
               'resource_limits':{'memory_bytes':None,'cpu_seconds':None,'max_cores':None,'max_threads':None}},
        'outputs':[{'path':declared_results[0],'sha256':digest(result_path),'kind':'result'}],
        'checks':['Raw before/after source, fixture and protected-file hashes agree for every case.',
                  'All measured source bytes match archived ZIP members.',
                  'All shared parent boards and embedded result boards were checked from archived piece definitions.',
                  'All child processes report successful completion with no recorded Python networking or resource interruption.',
                  'Audit input bytes were hashed before and after its checks; no original benchmark report was modified.'],
        'result':'Preserved finite evidence passed integrity and board-validity checks. Reuse is supported; three optional search heuristics lack demonstrated solution-quality benefit. Resume correctness is covered by separate regression tests.',
        'residual_risks':RISKS,
        'input_artifacts':[{'path':path.relative_to(repo).as_posix(),'sha256':before[path],'sha256_after':after[path]} for path in paths],
        'declared_results':declared_results}
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'manifest':str(manifest_path),'summary':str(result_path),'inputs':len(paths),'boards':verified_boards}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
