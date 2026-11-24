"""Offline CPU/hybrid worker for the complete five-clue Eternity II problem.

The persistent dashboard owns HTTP. This worker never starts a server, imports
a GPU backend unless hybrid or GPU matching is selected. It never contacts the project or
treats a regional repair as a solution.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import struct
import threading
import time

from app_paths import state_root
from dfs_profiles import PRUNING_PROFILES, dfs_profile_options, validate_pivot_engine
from exact_model import constraint_edges, from_bundle, problem_hash
from process_control import RunLock
from search_resources import peak_memory_mb
from validator import load_bundle, validate_board


MEMORY_SOFT_LIMIT_MB = 2048.0
# The 256-cell domains alone exceed the small-instance engine defaults.
FULL_BOARD_REGION_WORK = 10_000_000
UNBOUNDED_SECONDS = 1.0e12  # Finite native API horizon, approximately 31,688 years.
SCOPE = 'full-16x16-five-clue-puzzle-all-480-edges'
ENGINES = ('dfs', 'sat', 'sat-channels', 'cp-sat', 'hybrid', 'optimize', 'core-repair', 'matching', 'partitioned', 'sat-optimize', 'gpu-matching', 'learning-dfs', 'learning-rules', 'regional-master', 'refined-dd', 'impact-dfs', 'fast-dfs', 'fast-score', 'cuda-fast-score')
REPAIR_ENGINES = ('optimize', 'core-repair', 'matching', 'sat-optimize', 'gpu-matching', 'fast-score', 'cuda-fast-score')
# Compiled helper engines: fast-dfs is an exhaustive all-480 search, fast-score a high-score heuristic.
FAST_ENGINES = ('fast-dfs', 'fast-score')
PUBLISHED_BOARD_ENGINES = FAST_ENGINES + ('cuda-fast-score',)
FAST_MAX_WORKERS = 64
DEFAULT_SEED = 20261007


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.025)


def _read_json(path, limit):
    with Path(path).open('rb') as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError('Local JSON exceeds its size limit')
    return json.loads(payload)


def _backend(engine):
    if engine in ('dfs', 'hybrid'):
        from exact_dfs import solve
    elif engine == 'impact-dfs':
        from impact_dfs import solve
    elif engine == 'learning-rules':
        from explained_rules import solve
    elif engine == 'learning-dfs':
        from explained_dfs import solve
    elif engine == 'regional-master':
        from regional_master import solve
    elif engine == 'refined-dd':
        from refined_diagrams import solve
    elif engine in ('sat', 'sat-channels'):
        from exact_sat import solve
    elif engine == 'optimize':
        from exact_optimize import optimize as solve
    elif engine == 'core-repair':
        from exact_optimize import core_guided_repair as solve
    elif engine == 'sat-optimize':
        from exact_sat_optimize import optimize as solve
    elif engine in ('matching', 'gpu-matching'):
        from smart_repair import solve
    elif engine == 'partitioned':
        from exact_parallel import solve
    elif engine == 'cuda-fast-score':
        from cuda_score_search import solve
    elif engine in FAST_ENGINES:
        from fast_search import solve
    else:
        from exact_cp import solve
    return solve


def _settings(engine, seconds, seed, workers, port, backend='auto', replicas=4096, pruning='baseline', gpu_repair_batch=16, repair_quality='baseline'):
    if repair_quality not in ('baseline','guided','guided-lift','bounded','learned','partition-bounds','price-bounds'): raise ValueError('Invalid repair quality policy')
    if engine not in ENGINES:
        raise ValueError('Unknown exact engine')
    if seconds is not None and (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
        raise ValueError('seconds must be finite and nonnegative, or omitted')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')
    if type(workers) is not int or not 1 <= workers <= (FAST_MAX_WORKERS if engine in FAST_ENGINES else 4 if engine in ('cp-sat', 'optimize', 'partitioned') else 1):
        raise ValueError('CP-SAT, optimization and partitioned DFS permit one to four workers; fast compiled search permits 1 to 64; other CPU engines require one')
    if engine == 'cuda-fast-score' and backend == 'opencl':
        raise ValueError('CUDA fast-score requires the NVIDIA CUDA backend')
    if pruning not in PRUNING_PROFILES:
        raise ValueError('Unknown DFS pruning profile')
    validate_pivot_engine(pruning, engine, workers)
    if type(gpu_repair_batch) is not int or not 1 <= gpu_repair_batch <= 64:
        raise ValueError('GPU repair batch must be 1..64 neighborhoods')
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('port must be 1..65535')
    if backend not in ('auto', 'cuda', 'opencl') or type(replicas) is not int or not 32 <= replicas <= 32768:
        raise ValueError('Hybrid requires a valid GPU backend and batch size32..32768')
    return {'backend': backend, 'replicas': replicas, 'engine': engine, 'seconds': seconds, 'seed': seed, 'workers': workers,
            'port': port, 'memory_soft_limit_mb': MEMORY_SOFT_LIMIT_MB, 'scope': SCOPE, 'pruning': pruning, 'gpu_repair_batch': gpu_repair_batch, 'repair_quality': repair_quality}


def _known_cached_board(home, board, digest):
    """Exact placement lookup only: absence never establishes a discovery."""
    database = Path(home) / 'library' / 'library.sqlite3'
    if not database.is_file():
        return None
    connection = None
    try:
        connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=.5)
        blob = struct.pack('<256H', *board)
        found = connection.execute('SELECT 1 FROM boards WHERE local_sha=? AND placement=? LIMIT 1', (digest, blob)).fetchone()
        return True if found else None
    except (OSError, sqlite3.Error):
        return None
    finally:
        if connection is not None:
            connection.close()


class _PreparationEnded(Exception):
    def __init__(self, outcome):
        self.outcome = outcome


def _load_or_sample_hints(path, bundle, fingerprint, seed, backend, replicas, seconds,
                          cache_dir, should_stop, on_progress, sampler_factory=None):
    """Freeze a bounded, independently checked hint set for reproducible resume."""
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if path.exists():
        payload = _read_json(path, 2 * 1024 * 1024)
        checksum = payload.pop('checksum',None)
        if checksum != digest(payload) or payload.get('version') != 1 or payload.get('problem_hash') != fingerprint or payload.get('seed') != seed:
            raise ValueError('Hybrid hints are incompatible or corrupt; use --fresh-exact to restart')
        boards, scores, sampling = payload.get('boards'),payload.get('scores'),payload.get('sampling')
        fresh = False
    else:
        if should_stop():
            raise _PreparationEnded('stopped')
        if sampler_factory is None:
            from hybrid_sampling import sample
            sampler_factory = sample
        sampled = sampler_factory(bundle,seconds=seconds,replicas=replicas,seed=seed,backend=backend,
                                  cache_dir=cache_dir,should_stop=should_stop,on_progress=on_progress)
        if should_stop():
            raise _PreparationEnded('stopped')
        boards,scores = sampled.get('boards'),sampled.get('scores')
        sampling = {key:value for key,value in sampled.items() if key not in ('boards','scores')}
        fresh = True
    if not isinstance(boards,list) or not 1 <= len(boards) <= 64 or not isinstance(scores,list) or len(scores) != len(boards) or not isinstance(sampling,dict):
        raise ValueError('Hybrid requires1..64 saved candidate boards')
    for board,score in zip(boards,scores):
        report = validate_board(board,bundle)
        if not report['valid'] or type(score) is not int or report['score'] != score:
            raise ValueError('Hybrid candidate failed independent validation')
    if fresh:
        payload = {'version':1,'problem_hash':fingerprint,'seed':seed,'boards':boards,'scores':scores,'sampling':sampling}
        _atomic_json(path,dict(payload,checksum=digest(payload)))
    return boards,sampling


def run_task(state_dir=None, *, engine='dfs', seconds=None, seed=DEFAULT_SEED, workers=1,
             preserve_stop=False, port=8765, fresh_exact=False, backend='auto', replicas=4096, pruning='baseline', gpu_repair_batch=16, repair_quality='baseline', solver_factory=None, sampler_factory=None):
    """Run one exact search, returning an exit code; injected factories are for tests.

    A solver factory receives the engine name and returns its solve callable.
    Native SAT/CP search state is not resumed. DFS checkpoints are used only
    when the selected implementation exposes its explicit checkpoint API.
    """
    settings = _settings(engine, seconds, seed, workers, port, backend, replicas, pruning, gpu_repair_batch, repair_quality)
    settings['fresh_exact'] = bool(fresh_exact)
    home = state_root(state_dir)
    runtime = home / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    try:
        lock = RunLock(runtime / 'run.lock').acquire()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1  # Never replace the active owner's status or STOP marker.

    started = time.monotonic()
    stop_event = threading.Event()
    heartbeat_done = threading.Event()
    guard = threading.RLock()
    old_signals = {}
    heartbeat = None
    next_poll = 0.0
    background_error = None
    result = None
    repair_codec = None
    repair_checkpoint_path = None
    bundle = None
    best = None
    local_library = None
    status = {'state': 'starting', 'pid': os.getpid(), 'method': 'exact',
              'search_method': 'exact-' + engine, 'exact_engine': engine,
              'compute_device': 'CPU', 'gpu': 'CPU exact search', 'workers': workers,
              'port': port, 'state_dir': str(home), 'start_time': _now(),
              'last_update': _now(), 'elapsed_seconds': 0.0, 'best_score': None,
              'source_best_score': 466, 'branches': 0, 'conflicts': 0,
              'exact_counters_available': engine in ('dfs', 'hybrid', 'partitioned') + PUBLISHED_BOARD_ENGINES,
              'pruning': pruning, 'repair_quality': repair_quality, 'neighborhoods': 0, 'recombinations': 0,
              'diversification_attempts': 0, 'diversifications': 0, 'tabu_rejections': 0,
              'region_calls': 0, 'region_improvements': 0, 'region_seconds': 0.0, 'parent_count': 0,
              'exact_phase': 'starting', 'scope': SCOPE,
              'external_network_enabled': False,
              'library': {'enabled': False, 'network_enabled': False, 'mode': 'offline'},
              'counter_semantics': 'local exact-search branches/conflicts; not BOINC nodes or credit',
              'memory_soft_limit_mb': MEMORY_SOFT_LIMIT_MB, 'resumed': False,
              'resume_capability': 'none; each native SAT/CP run starts a new search'}
    if engine in ('matching', 'gpu-matching'):
        status.update(branches=None, conflicts=None,
                      counter_semantics='completed repair neighborhoods, recombinations, diversifications and region searches; not exhaustive branches or BOINC credit')
    if engine == 'cuda-fast-score':
        from cuda_score_search import LANES
        status.update(backend='cuda', compute_device='NVIDIA CUDA', gpu='NVIDIA CUDA', gpu_lanes=LANES,
                      resume_capability='fresh GPU search on each start',
                      counter_semantics='accepted GPU piece placements under a heuristic mismatch budget')
    session = {'schema_version': 1, 'settings': settings, 'started_at': status['start_time'],
               'scope': SCOPE, 'external_network_enabled': False}

    def save_status():
        with guard:
            status.update(last_update=_now(), elapsed_seconds=round(time.monotonic() - started, 3))
            _atomic_json(runtime / 'status.json', status)

    def request_stop(*_):
        stop_event.set()
        with guard:
            status.setdefault('stop_reason', 'signal')
        try:
            (runtime / 'STOP').write_text('Exact search stopped by a process signal\n', encoding='utf-8')
        except OSError:
            pass  # The in-memory flag still cooperatively stops native search.

    def should_stop():
        nonlocal next_poll, background_error
        if stop_event.is_set():
            return True
        stamp = time.monotonic()
        if stamp < next_poll:
            return False
        with guard:
            if stamp < next_poll:
                return stop_event.is_set()
            next_poll = stamp + 0.1
            try:
                if (runtime / 'STOP').exists():
                    status['stop_reason'] = 'stop_file'
                    stop_event.set()
                memory = peak_memory_mb()
                status['peak_memory_mb'] = round(memory, 2)
                if memory >= MEMORY_SOFT_LIMIT_MB:
                    status['stop_reason'] = 'memory_soft_limit'
                    stop_event.set()
            except Exception as exc:
                background_error = exc
                status['stop_reason'] = 'resource_monitor_error'
                stop_event.set()
        return stop_event.is_set()

    def on_progress(progress):
        if not isinstance(progress, dict):
            return
        with guard:
            for key in ('branches', 'conflicts', 'max_depth', 'propagations', 'matching_checks',
                        'matching_failures', 'build_seconds', 'variables', 'clauses', 'table_rows',
                        'neighborhoods', 'recombinations', 'rounds', 'free_cells', 'cores',
                        'diversification_attempts', 'diversifications', 'tabu_rejections', 'region_calls',
                        'region_improvements', 'region_seconds', 'parent_count', 'region_cache_hits',
                        'quality_ejection_calls', 'quality_recombination_calls', 'quality_candidate_evaluations',
                        'quality_heuristic_state_evaluations', 'quality_path_expansions', 'quality_native_branches',
                        'quality_native_conflicts', 'quality_table_rows', 'quality_accepted_piece_changes',
                        'quality_returned_piece_changes', 'quality_returned_board_improvements', 'quality_adoptions',
                        'quality_deterministic_work', 'quality_accepted_board_improvements',
                        'quality_true_score_gain', 'quality_stopped_calls',
                        'tickets_total', 'tickets_completed', 'split_tickets', 'queue_records', 'cost_batches', 'cost_pairs',
                        'cost_kernel_seconds', 'cost_elapsed_seconds', 'target_score', 'hitting_set_seconds',
                        'gac_checks', 'gac_removed', 'independent_completions', 'probe_checks', 'probe_rejections',
                        'complete_boards', 'restarts', 'nodes_per_second', 'engine_best_score'):
                value = progress.get(key)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    status[key] = value
                    if key == 'branches':
                        status['exact_counters_available'] = True
            for key,value in progress.items():
                if key.startswith('region_reasoning_') and type(value) in (int,float) and math.isfinite(value) and value>=0: status[key]=value
            for key in ('component_counters_session','rule_counters_session','reasoning_counters_session'):
                if isinstance(progress.get(key),dict):
                    status[key]={k:v for k,v in progress[key].items() if isinstance(k,str) and type(v) in (int,float) and math.isfinite(v) and v>=0}
            for key in ('cost_backend', 'cost_device', 'gpu'):
                if isinstance(progress.get(key), str):
                    status[key] = progress[key][:160]
            phase = progress.get('phase')
            if isinstance(phase, str):
                status['exact_phase'] = phase[:80]
            elif engine in ('dfs', 'hybrid', 'partitioned'):
                status['exact_phase'] = 'searching'
            if isinstance(progress.get('backend'), str):
                status['backend'] = progress['backend'][:80]

    def on_improvement(candidate):
        nonlocal best, local_library
        if not isinstance(candidate, dict):
            raise RuntimeError('Repair backend returned an invalid candidate record')
        board = candidate.get('board')
        report = validate_board(board, bundle)
        if not report['valid'] or type(candidate.get('score')) is not int or report['score'] != candidate['score']:
            raise RuntimeError('Repair candidate failed independent board and score validation')
        with guard:
            promotes_global = report['score'] > status['best_score']
            if not promotes_global and engine not in PUBLISHED_BOARD_ENGINES:
                return
            board = list(board)
            digest = hashlib.sha256(struct.pack('<256H', *board)).hexdigest()
            known = _known_cached_board(home, board, digest)
            source = 'existing local library arrangement' if known else 'local search result; discovery status unverified'
            record = {'score': report['score'], 'board': board, 'validated': True, 'complete': report['complete'],
                      'library_known': known, 'source': source, 'sha256_uint16le': digest, 'found_at': _now(),
                      'search_method': 'exact-' + engine, 'scope': SCOPE,
                      'project_acceptance_confirmed': False, 'automatic_upload': False}
            if engine in PUBLISHED_BOARD_ENGINES:
                from library_cache import LibraryMonitor, database_busy, queue_local_result
                directory = home / 'library'
                marker = queue_local_result(directory, record)
            (home / 'results').mkdir(exist_ok=True)
            destination = home / 'results' / f'{report["score"]}-{digest}.json'
            _atomic_json(destination, record)
            if engine in ('fast-score', 'cuda-fast-score') and report['score'] > status.get('run_best_score', -1):
                status.update(run_best_score=report['score'], run_best_sha256=digest)
            if promotes_global:
                _atomic_json(runtime / 'best.json', record)
                best = board
                status.update(best_score=report['score'], board_source=source, library_known=known)
            save_status()
        if engine in PUBLISHED_BOARD_ENGINES:
            try:
                if local_library is None:
                    # Opening the cache also replays any older pending results.
                    local_library = LibraryMonitor(directory, pieces_path=bundle.data_dir / 'pieces.txt', db_timeout=0)
                else:
                    local_library.replay_local_results(marker)
            except sqlite3.OperationalError as exc:
                if not database_busy(exc):
                    raise

    def heartbeat_loop():
        nonlocal background_error
        while not heartbeat_done.wait(0.5):
            try:
                should_stop()
                save_status()
            except Exception as exc:
                background_error = exc
                stop_event.set()
                return

    try:
        if not preserve_stop:
            (runtime / 'STOP').unlink(missing_ok=True)
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_signals[sig] = signal.signal(sig, request_stop)
        bundle = load_bundle()
        best = list(bundle.record_board)
        baseline = validate_board(best, bundle)
        if not baseline['valid'] or baseline['score'] != 466:
            raise RuntimeError('Bundled 466 board failed validation')
        status['best_score'] = baseline['score']
        prior_path = runtime / 'best.json'
        prior_valid = False
        if prior_path.exists():
            try:
                prior = _read_json(prior_path, 131072)
                report = validate_board(prior.get('board'), bundle)
                if report['valid'] and report['score'] >= status['best_score']:
                    best = list(prior['board'])
                    status['best_score'] = report['score']
                    prior_valid = True
                    status['library_known'] = True if prior.get('library_known') is True else None
                    if isinstance(prior.get('source'), str):
                        status['board_source'] = prior['source'][:160]
                else:
                    status['saved_best_warning'] = 'Saved board rejected; using the verified bundled reference'
            except (OSError, ValueError, TypeError, AttributeError):
                status['saved_best_warning'] = 'Saved board could not be validated; using the bundled reference'
        if not prior_valid:
            _atomic_json(prior_path, {'score': status['best_score'], 'board': best, 'validated': True,
                                     'source': 'bundled five-clue reference; credited to Jef'})
        save_status()
        if should_stop():
            result = {'outcome': 'stopped', 'board': None, 'branches': 0, 'conflicts': 0}
        elif validate_board(best, bundle)['complete']:
            result = {'outcome': 'solved', 'board': best, 'branches': 0, 'conflicts': 0}
        else:
            status['exact_phase'] = 'building_problem'
            problem = from_bundle(bundle)
            if problem.size != 16 or len(problem.fixed) != 5 or len(constraint_edges(problem)) != 480:
                raise RuntimeError('Exact worker requires the complete five-clue puzzle')
            fingerprint = problem_hash(problem)
            session['problem_sha256'] = fingerprint
            solver = (solver_factory or _backend)(engine)
            solve_args = {'problem': problem, 'seconds': UNBOUNDED_SECONDS if seconds is None else seconds,
                          'seed': seed, 'workers': workers, 'should_stop': should_stop,
                          'on_progress': on_progress}
            if engine in ('dfs', 'hybrid', 'partitioned'):
                solve_args['options'] = dfs_profile_options(pruning)
            if engine in ('learning-dfs','learning-rules','regional-master','refined-dd','impact-dfs'):
                status['resume_capability']='fresh search; no saved frontier for this experimental engine'
            if engine == 'sat-channels':
                solve_args.update(piece_channels=True, small_pairwise=True)
            if engine in ('learning-dfs', 'learning-rules'):
                # The full official root exceeds the small experimental API
                # cap before either learning engine can try a placement.
                solve_args['max_work'] = 100_000_000
            if engine == 'learning-dfs':
                solve_args['matching'] = True
            if engine == 'regional-master':
                solve_args.update(minimize_interfaces=True, allocation_cores=True,
                                  color_conservation=True, color_coupling='edge_colors', hall_capacity=True,
                                  cache_witnesses=True, max_work=FULL_BOARD_REGION_WORK)
            if engine == 'refined-dd':
                solve_args.update(ordering='domain', max_work=FULL_BOARD_REGION_WORK)
            if engine == 'impact-dfs': solve_args.update(ordering='impact', discrepancy_limit=None)
            if engine == 'partitioned':
                queue_path = runtime / 'exact-partitioned-queue.json'
                if fresh_exact and queue_path.exists():
                    queue_path.replace(runtime / f'{queue_path.stem}.archived-{time.time_ns()}.json')
                    status['previous_checkpoint_archived'] = True
                solve_args['queue_path'] = queue_path
                status['resumed'] = queue_path.exists()
                status['resume_capability'] = 'explicit partition queue with per-task DFS frontiers'
            if engine == 'cuda-fast-score':
                solve_args.update(on_improvement=on_improvement, cache_dir=runtime / 'gpu-cache' / 'cuda')
            elif engine in REPAIR_ENGINES:
                solve_args.update(reference=list(best), on_improvement=on_improvement)
                status['resume_capability'] = 'saved best board only; starts a fresh repair search'
                if engine in ('matching', 'gpu-matching'):
                    from smart_repair import cached_parents
                    if repair_quality != 'baseline': solve_args['repair_quality'] = repair_quality
                    if engine == 'gpu-matching':
                        solve_args.update(gpu_backend=backend, gpu_batch_size=gpu_repair_batch,
                                          gpu_cache_dir=runtime / 'gpu-cache', gpu_allow_cpu_fallback=False)
                        status['compute_device'] = 'GPU placement costs + CPU exact assignment'
                    if 'checkpoint' in inspect.signature(solver).parameters:
                        from repair_checkpoint import CheckpointCodec, load_checkpoint
                        from smart_repair import checkpoint_settings
                        keys = inspect.signature(checkpoint_settings).parameters
                        repair_codec = CheckpointCodec(problem, checkpoint_settings(**{
                            key: value for key, value in solve_args.items() if key in keys}))
                        repair_checkpoint_path = runtime / f'exact-{engine}-checkpoint.json'
                        status['resume_capability'] = 'structured repair population, random state, policy and completed neighborhoods'
                        def on_repair_checkpoint(checkpoint):
                            # Validate an independent detached copy before atomic replacement.
                            decoded = repair_codec.decode(checkpoint)
                            if decoded['best_score'] > status['best_score']:
                                on_improvement({'board': decoded['best'], 'score': decoded['best_score']})
                            with guard:
                                _atomic_json(repair_checkpoint_path, checkpoint)
                                status['checkpoint_saved_at'] = _now()
                        solve_args['on_checkpoint'] = on_repair_checkpoint
                        if fresh_exact and repair_checkpoint_path.exists():
                            repair_checkpoint_path.replace(repair_checkpoint_path.with_name(
                                f'{repair_checkpoint_path.stem}.archived-{time.time_ns()}.json'))
                            status['previous_checkpoint_archived'] = True
                        if repair_checkpoint_path.exists():
                            try:
                                checkpoint = load_checkpoint(repair_checkpoint_path)
                                decoded = repair_codec.decode(checkpoint)
                            except (OSError, ValueError, TypeError, AttributeError) as exc:
                                raise ValueError('Repair checkpoint is incompatible or unreadable; use --fresh-exact to archive it and restart') from exc
                            solve_args['checkpoint'] = checkpoint
                            status['resumed'] = True
                            status['cached_parent_count'] = len(decoded['bank'])
                            on_improvement({'board': decoded['best'], 'score': decoded['best_score']})
                            solve_args['reference'] = list(best)
                    if not status['resumed']:
                        try:
                            preparation_budget = 5.0 if seconds is None else max(0.0, min(5.0, seconds - (time.monotonic() - started)))
                            solve_args['parents'] = cached_parents(problem, home / 'library' / 'library.sqlite3', limit=32,
                                                                 seed=seed, should_stop=should_stop, max_seconds=preparation_budget)
                        except (OSError, ValueError, sqlite3.Error) as exc:
                            solve_args['parents'] = []
                            status['cached_parents_warning'] = f'Local parent cache unavailable: {type(exc).__name__}'
                        status['cached_parent_count'] = len(solve_args['parents'])
            if engine in FAST_ENGINES:
                from fast_search import has_saved_search
                fast_dir = runtime / engine
                if fresh_exact and fast_dir.exists():
                    fast_dir.replace(runtime / f'{engine}.archived-{time.time_ns()}')
                    status['previous_checkpoint_archived'] = True
                solve_args.update(state_dir=fast_dir, mode='score' if engine == 'fast-score' else 'strict')
                status['resumed'] = has_saved_search(fast_dir)
                status['resume_capability'] = 'compiled search checkpoints, one per worker thread'
                status['compute_device'] = 'CPU compiled search'
                if engine == 'fast-score':
                    solve_args.update(on_board=on_improvement, on_improvement=None)
                    status['counter_semantics'] = 'compiled search placements and complete boards under a heuristic break schedule; not an exhaustive search, BOINC nodes or credit'
            checkpoint_path = runtime / ('exact-hybrid-dfs-checkpoint.json' if engine == 'hybrid' else 'exact-dfs-checkpoint.json')
            hint_solution = None
            if engine == 'hybrid':
                hint_path = runtime / 'hybrid-hints.json'
                if fresh_exact:
                    for local_file in (hint_path, checkpoint_path):
                        if local_file.exists():
                            local_file.replace(local_file.with_name(f'{local_file.stem}.archived-{time.time_ns()}.json'))
                if checkpoint_path.exists() and not hint_path.exists():
                    raise ValueError('Hybrid checkpoint has no saved hints; use --fresh-exact to restart')
                status['compute_device'] = 'GPU sampling then CPU DFS'
                status['exact_phase'] = 'loading_hints' if hint_path.exists() else 'gpu_sampling'
                save_status()
                if seconds is not None and time.monotonic() - started >= seconds:
                    raise _PreparationEnded('timeout')
                def sampling_progress(progress):
                    with guard:
                        status['exact_phase'] = 'gpu_sampling'
                        for source, dest in (('generated_boards','sampled_boards'),('boards_per_second','sampling_boards_per_second'),('compile_seconds','sampling_compile_seconds')):
                            value = progress.get(source)
                            if type(value) in (int,float) and math.isfinite(value) and value >= 0:
                                status[dest] = value
                        save_status()
                sample_seconds = 3.0 if seconds is None else min(3.0,max(0.0,seconds-(time.monotonic()-started)))
                def sampling_stop():
                    nonlocal next_poll
                    if (runtime / 'STOP').exists():
                        next_poll = 0.0
                    return should_stop() or (seconds is not None and time.monotonic()-started >= seconds)
                hints, sampling = _load_or_sample_hints(hint_path, bundle, fingerprint, seed,
                    backend, replicas, sample_seconds, runtime / 'gpu-cache', sampling_stop,
                    sampling_progress, sampler_factory)
                if should_stop():
                    raise _PreparationEnded('stopped')
                solve_args['hints'] = hints
                status.update(hybrid_hint_count=len(hints), sampled_boards=sampling.get('generated_boards',0),
                    sampling_boards_per_second=sampling.get('boards_per_second',0),
                    sampling_compile_seconds=sampling.get('compile_seconds',0),
                    sampling_initialization_seconds=sampling.get('initialization_seconds',0),
                    sampling_backend=sampling.get('backend'), sampling_device=sampling.get('device'))
                session['hybrid_sampling'] = sampling
                hint_solution = next((board for board in hints if validate_board(board,bundle)['complete']),None)
            if hint_solution is None and engine in ('dfs', 'hybrid') and 'checkpoint' in inspect.signature(solver).parameters:
                status['resume_capability'] = 'DFS explicit frontier checkpoint'
                def on_checkpoint(checkpoint):
                    with guard:
                        _atomic_json(checkpoint_path, checkpoint)
                        status['checkpoint_saved_at'] = _now()
                solve_args['on_checkpoint'] = on_checkpoint
                if fresh_exact and checkpoint_path.exists():
                    checkpoint_path.replace(runtime / f'{checkpoint_path.stem}.archived-{time.time_ns()}.json')
                    status['previous_checkpoint_archived'] = True
                if checkpoint_path.exists():
                    try:
                        checkpoint = _read_json(checkpoint_path, 4 * 1024 * 1024)
                        if checkpoint.get('problem_hash') != fingerprint or checkpoint.get('seed') != seed:
                            raise ValueError('Checkpoint does not match this puzzle and seed')
                    except (OSError, ValueError, TypeError, AttributeError) as exc:
                        raise ValueError('DFS checkpoint is incompatible or unreadable; use --fresh-exact to archive it and restart') from exc
                    solve_args['checkpoint'] = checkpoint
                    status['resumed'] = True
            status.update(state='running', exact_phase='building')
            save_status()
            _atomic_json(runtime / 'exact-session.json', session)
            heartbeat = threading.Thread(target=heartbeat_loop, name='eternity-exact-status', daemon=True)
            heartbeat.start()
            if seconds is not None:
                solve_args['seconds'] = max(0.0, seconds - (time.monotonic() - started))
            try:
                result = {'outcome':'solved','board':hint_solution} if hint_solution is not None else solver(**solve_args)
            except ValueError as exc:
                if engine in ('dfs', 'hybrid', 'partitioned', 'matching', 'gpu-matching') + FAST_ENGINES and status['resumed']:
                    raise ValueError(f'Saved search could not be resumed; use --fresh-exact to archive it and restart: {exc}') from exc
                raise
            heartbeat_done.set()
            heartbeat.join()
            if engine in ('dfs', 'hybrid') and result.get('checkpoint') is not None:
                _atomic_json(checkpoint_path, result['checkpoint'])
                status['checkpoint_saved_at'] = _now()
            if repair_codec is not None and result.get('checkpoint') is not None:
                on_repair_checkpoint(result['checkpoint'])
        next_poll = 0.0
        should_stop()  # Recheck a late STOP or memory limit before any negative conclusion.
        if background_error is not None:
            raise RuntimeError('Exact worker resource/status monitoring failed') from background_error
        outcome = result.get('outcome')
        allowed = ('solved', 'timeout', 'stopped', 'improved', 'infeasible_for_target', 'unchanged') if engine in REPAIR_ENGINES else ('solved', 'infeasible', 'timeout', 'stopped')
        if outcome not in allowed:
            raise RuntimeError('Exact backend returned an invalid outcome')
        if engine in REPAIR_ENGINES and result.get('board') is not None:
            on_improvement(result)
        if engine in REPAIR_ENGINES and validate_board(best, bundle)['complete']:
            outcome = 'solved'
            result = dict(result, board=best)
        on_progress(result)
        status['exact_outcome'] = outcome
        if outcome == 'solved':
            board = result.get('board')
            report = validate_board(board, bundle)
            if not report['complete']:
                raise RuntimeError('Exact result failed independent full-board 480/480 validation')
            on_improvement({'board': board, 'score': 480})
            status.update(state='solved', best_score=480, exact_phase='complete')
        elif stop_event.is_set():
            # A late stop takes precedence over an untrusted backend UNSAT flag.
            status.update(state='stopped', exact_outcome='stopped', exact_phase='stopped')
        elif outcome == 'infeasible':
            if status['resumed'] or result.get('proof_scope') == 'remaining_checkpoint_frontier' or result.get('coverage_verified') is False:
                status.update(state='checkpoint_frontier_exhausted', exact_phase='complete',
                              proof_scope='remaining_checkpoint_frontier', coverage_verified=False,
                              exact_conclusion='The resumed DFS frontier is exhausted; this does not prove the complete puzzle impossible')
            else:
                status.update(state='infeasible', exact_phase='complete', proof_scope='entire_supplied_problem',
                              exact_conclusion='The selected backend reports no solution for all 480 edges and the five fixed clues; no proof certificate was checked')
        elif outcome == 'timeout':
            limit=result.get('termination','timeout')
            limit=limit if limit in ('work_limit','node_limit','refinement_limit','query_limit') else 'timeout'
            status.update(state='finished_bounded_run', exact_phase=limit,search_termination=limit,
                          exact_conclusion='Search limit reached; no feasibility conclusion')
        elif outcome in ('improved', 'infeasible_for_target', 'unchanged'):
            status.update(state='finished_bounded_run', exact_phase='finished', coverage_verified=False,
                          proof_scope=result.get('proof_scope', 'selected_repair_target'),
                          exact_conclusion='Repair target search ended. Saved improvements remain available; this is not a proof that the complete puzzle is impossible')
        else:
            status.update(state='stopped', exact_phase='stopped')
        if repair_checkpoint_path is not None and status['state'] == 'solved' and repair_checkpoint_path.exists():
            repair_checkpoint_path.replace(repair_checkpoint_path.with_name(
                f'{repair_checkpoint_path.stem}.completed-{time.time_ns()}.json'))
            status['completed_checkpoint_archived'] = True
        if engine in ('dfs', 'hybrid') and status['state'] in ('solved', 'infeasible', 'checkpoint_frontier_exhausted'):
            checkpoint_path = runtime / ('exact-hybrid-dfs-checkpoint.json' if engine == 'hybrid' else 'exact-dfs-checkpoint.json')
            if checkpoint_path.exists():
                checkpoint_path.replace(runtime / f'{checkpoint_path.stem}.completed-{time.time_ns()}.json')
                status['completed_checkpoint_archived'] = True
    except _PreparationEnded as exc:
        if not should_stop() and seconds is not None and time.monotonic()-started >= seconds:
            exc.outcome = 'timeout'
        result = {'outcome': exc.outcome, 'board': None}
        status.update(state='stopped' if exc.outcome == 'stopped' else 'finished_bounded_run',
                      exact_outcome=exc.outcome, exact_phase=exc.outcome)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            status.update(state='stopped', exact_outcome='stopped', exact_phase='stopped')
        else:
            status.update(state='error', exact_outcome='error', exact_phase='error', error=f'{type(exc).__name__}: {exc}')
            print(status['error'], flush=True)
    finally:
        heartbeat_done.set()
        if heartbeat is not None:
            heartbeat.join()
        try:
            save_status()
            session.update(finished_at=_now(), outcome=status.get('exact_outcome'), status=dict(status),
                           elapsed_seconds=time.monotonic() - started,
                           native_search_state_resumed=False if engine not in ('dfs', 'hybrid', 'partitioned') + FAST_ENGINES else status['resumed'],
                           repair_search_state_resumed=engine in ('matching', 'gpu-matching') and status['resumed'])
            if isinstance(result, dict):
                session['result'] = {key: value for key, value in result.items() if key not in ('board', 'checkpoint')}
            _atomic_json(runtime / 'exact-session.json', session)
        finally:
            if local_library is not None:
                local_library.close()
            for sig, previous in old_signals.items():
                signal.signal(sig, previous)
            lock.close()
    return 1 if status['state'] == 'error' else 0


def main(argv=None, solver_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=ENGINES, default='dfs')
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--seconds', type=float, help='Omit to keep searching until stopped or resolved')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--backend', choices=('auto','cuda','opencl'), default='auto')
    parser.add_argument('--replicas', type=int, default=4096, help='Hybrid GPU sample batch32..32768')
    parser.add_argument('--repair-quality', choices=('baseline','guided','guided-lift','bounded','learned','partition-bounds','price-bounds'), default='baseline', help='Optional guided repair uses experimental targeted chains and three-parent recombination')
    parser.add_argument('--gpu-repair-batch', type=int, default=16, help='GPU matching neighborhoods per batch (1..64)')
    parser.add_argument('--pruning', choices=PRUNING_PROFILES, default='baseline', help='DFS pruning profile; changing profile requires a compatible checkpoint or --fresh-exact')
    parser.add_argument('--preserve-stop', action='store_true')
    parser.add_argument('--fresh-exact', action='store_true', help='Archive the selected DFS, fast-search or structured-repair checkpoint and start a fresh search')
    parser.add_argument('--port', type=int, default=8765, help='Dashboard metadata only; no server is started')
    args = parser.parse_args(argv)
    try:
        return run_task(**vars(args), solver_factory=solver_factory)
    except (ValueError, OSError) as exc:
        print(f'{type(exc).__name__}: {exc}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
