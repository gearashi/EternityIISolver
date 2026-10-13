"""Offline CPU workers on exhaustive, disjoint DFS partitions.

Each ticket fixes a different state of one cell. Workers cannot access queue
files: only the parent optionally persists an explicitly supplied queue_path.
Checksums detect corruption, not forged history. Exhaustion after any checkpoint
resume is conservatively reported only for the recorded remaining frontier.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
import time

import exact_dfs
from exact_model import ExactProblem, check_solution, initial_domains, problem_hash

QUEUE_VERSION = 1
MAX_QUEUE_BYTES = 64 * 1024 * 1024
INITIAL_SLICE_SECONDS = 2.0
MAX_SLICE_SECONDS = 30.0
WORKER_MEMORY_MB = 512
_CHILD_STOP = None


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def _seal(value):
    body = {key: val for key, val in value.items() if key != 'checksum'}
    return dict(body, checksum=hashlib.sha256(_canonical(body)).hexdigest())


def _options_identity(options):
    version = None
    if options['extra_rules'] is not None:
        from exact_rules import RULES_VERSION
        version = RULES_VERSION
    return hashlib.sha256(_canonical({'options': options, 'rules_version': version})).hexdigest()


def _ticket_seed(seed, ordinal):
    return (seed + ordinal * 0x9e3779b97f4a7c15) % (1 << 64)


def _partition(problem):
    domains = initial_domains(problem)
    eligible = [cell for cell, domain in enumerate(domains) if len(domain) > 1 and cell not in problem.fixed]
    if any(not domain for domain in domains) or not eligible:
        return {'cell': None, 'states': []}
    cell = min(eligible, key=lambda c: (len(domains[c]), c))
    return {'cell': cell, 'states': sorted(domains[cell])}


def _ticket_problem(problem, partition, state):
    if partition['cell'] is None:
        return problem
    domains = list(problem.domains)
    domains[partition['cell']] = (state,)
    return ExactProblem(problem.size, problem.faces, dict(problem.fixed), tuple(domains), problem.edges)


def _new_queue(problem, seed, options):
    partition = _partition(problem)
    tickets = []
    for state in partition['states'] if partition['cell'] is not None else (None,):
        identifier = 'whole' if state is None else 'cell-%d-state-%d' % (partition['cell'], state)
        tickets.append({'id': identifier, 'state': state, 'status': 'pending', 'checkpoint': None,
                        'slices': 0, 'slice_seconds': INITIAL_SLICE_SECONDS, 'coverage_verified': False,
                        'board': None, 'resource_interruption': None, 'counters': {'branches': 0, 'conflicts': 0, 'max_depth': 0}})
    return {'version': QUEUE_VERSION, 'dfs_checkpoint_version': exact_dfs.CHECKPOINT_VERSION,
            'ordering_version': exact_dfs.ORDERING_VERSION, 'problem_hash': problem_hash(problem),
            'options': options, 'options_hash': _options_identity(options), 'seed': seed,
            'partition': partition, 'tickets': tickets}


def _check_checkpoint(checkpoint, problem, seed, options_hash):
    hints_hash = exact_dfs._hint_data(None, problem.size ** 2)[1]
    return exact_dfs._checked_checkpoint(checkpoint, problem_hash(problem), seed, problem.size ** 2,
                                         hints_hash, options_hash)


def _validate_queue(value, expected, problem):
    if not isinstance(value, dict) or set(value) != set(expected) | {'checksum'}:
        raise ValueError('Invalid parallel queue fields')
    checksum = value['checksum']
    if not isinstance(checksum, str) or checksum != _seal(value)['checksum']:
        raise ValueError('Parallel queue checksum mismatch')
    for key in expected:
        if key != 'tickets' and _canonical(value[key]) != _canonical(expected[key]):
            raise ValueError('Parallel queue problem, options, seed, or partition mismatch: ' + key)
    tickets = value['tickets']
    if not isinstance(tickets, list) or len(tickets) != len(expected['tickets']):
        raise ValueError('Parallel queue must contain every partition exactly once')
    for ordinal, (ticket, template) in enumerate(zip(tickets, expected['tickets'])):
        if not isinstance(ticket, dict) or set(ticket) != set(template):
            raise ValueError('Invalid parallel ticket fields')
        if type(ticket['state']) is not type(template['state']) or ticket['id'] != template['id'] or ticket['state'] != template['state']:
            raise ValueError('Parallel ticket identity/order mismatch')
        if ticket['status'] not in ('pending', 'infeasible', 'solved'):
            raise ValueError('Invalid parallel ticket status')
        if type(ticket['slices']) is not int or not 0 <= ticket['slices'] < 2**63:
            raise ValueError('Invalid parallel slice count')
        duration = ticket['slice_seconds']
        if type(duration) not in (int, float) or not math.isfinite(duration) or not INITIAL_SLICE_SECONDS <= duration <= MAX_SLICE_SECONDS:
            raise ValueError('Invalid parallel slice duration')
        if type(ticket['coverage_verified']) is not bool:
            raise ValueError('Invalid ticket coverage flag')
        if ticket['resource_interruption'] not in (None, 'memory_limit'):
            raise ValueError('Invalid ticket resource interruption')
        counters = ticket['counters']
        if not isinstance(counters, dict) or set(counters) != set(template['counters']) or any(type(v) is not int or not 0 <= v < 2**63 for v in counters.values()):
            raise ValueError('Invalid parallel ticket counters')
        task = _ticket_problem(problem, value['partition'], ticket['state'])
        if ticket['status'] == 'pending':
            if ticket['board'] is not None or ticket['coverage_verified']:
                raise ValueError('Pending ticket cannot claim completion')
            if ticket['checkpoint'] is not None:
                ticket['checkpoint'] = _check_checkpoint(ticket['checkpoint'], task, _ticket_seed(value['seed'], ordinal), value['options_hash'])
        elif ticket['checkpoint'] is not None:
            raise ValueError('Completed ticket must not contain a checkpoint')
        if ticket['status'] == 'solved':
            if not check_solution(task, ticket['board']) or not check_solution(problem, ticket['board']):
                raise ValueError('Saved parallel solution failed independent validation')
        elif ticket['board'] is not None:
            raise ValueError('Unsolved ticket must not contain a board')
    return {key: value[key] for key in expected}


def _load(path):
    if path.stat().st_size > MAX_QUEUE_BYTES:
        raise ValueError('Parallel queue exceeds its size limit')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key in parallel queue')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON constant in parallel queue: ' + value)
    with path.open('rb') as stream:
        raw = stream.read(MAX_QUEUE_BYTES + 1)
    if len(raw) > MAX_QUEUE_BYTES:
        raise ValueError('Parallel queue exceeds its size limit')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=invalid_constant)


def _save(path, queue):
    if path is None:
        return
    payload = _canonical(_seal(queue)) + b'\n'
    if len(payload) > MAX_QUEUE_BYTES:
        raise ValueError('Parallel queue exceeds its size limit; previous queue preserved')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _initialize_worker(stop_event):
    global _CHILD_STOP
    _CHILD_STOP = stop_event


def _run_ticket(problem, seconds, seed, checkpoint, options):
    from search_resources import peak_memory_mb
    sampled = [0.0]
    memory_stop = [False]
    def stop():
        if _CHILD_STOP is not None and _CHILD_STOP.is_set():
            return True
        now = time.monotonic()
        if now - sampled[0] >= .1:
            sampled[0] = now
            memory_stop[0] = peak_memory_mb() > WORKER_MEMORY_MB
        return memory_stop[0]
    result = exact_dfs.solve(problem, seconds=seconds, seed=seed, workers=1,
                             checkpoint=checkpoint, options=options, should_stop=stop)
    result['resource_interruption'] = 'memory_limit' if memory_stop[0] else None
    if memory_stop[0] and result['outcome'] == 'infeasible':
        raise AssertionError('Resource interruption is not infeasibility')
    return result


def _frontier(checkpoint):
    if checkpoint is None:
        return None
    return checkpoint.get('stack'), checkpoint.get('rng_state')


def _solve(problem, seconds=60, seed=0, workers=1, should_stop=None, on_progress=None,
          queue_path=None, options=None):
    """Explore disjoint cell-state tickets with at most four spawned workers.

    A queue is read/written only at an explicitly supplied path. Existing queues
    must match the problem, options and seed exactly; incompatible files are
    preserved. Completed tickets are never scheduled again. Cancellation is
    cooperative and every unfinished ticket retains its latest valid frontier.
    Restored completed counters/cursors are untrusted history, not a global
    infeasibility proof. Return ``queue`` for inspection when queue_path is None.
    """
    started = time.monotonic()
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Parallel DFS supports one to four workers')
    if type(seed) is not int or not -(2**63) <= seed < 2**64:
        raise ValueError('seed must be a bounded integer')
    options = exact_dfs.normalize_options(options)
    deadline = started + seconds
    expected = _new_queue(problem, seed, options)
    path = None if queue_path is None else Path(queue_path)
    restored = path is not None and path.exists()
    queue = _validate_queue(_load(path), expected, problem) if restored else expected
    tickets = queue['tickets']
    used_checkpoint = any(ticket['checkpoint'] is not None for ticket in tickets)
    active = {}
    pending = deque(index for index, ticket in enumerate(tickets) if ticket['status'] == 'pending')
    reason = None
    board = next((ticket['board'] for ticket in tickets if ticket['status'] == 'solved'), None)
    last_progress = started
    resource_interruptions = []

    def report(outcome):
        complete = sum(ticket['status'] != 'pending' for ticket in tickets)
        conditional = restored or used_checkpoint
        return {'backend': 'dfs-parallel', 'outcome': outcome, 'board': None if board is None else list(board),
                'workers': workers, 'elapsed_seconds': time.monotonic() - started,
                'branches': sum(t['counters']['branches'] for t in tickets),
                'conflicts': sum(t['counters']['conflicts'] for t in tickets),
                'max_depth': max(t['counters']['max_depth'] for t in tickets),
                'tickets_total': len(tickets), 'tickets_completed': complete,
                'active_ticket_ids': sorted(tickets[index]['id'] for index in active.values()),
                'completed_ticket_ids': [t['id'] for t in tickets if t['status'] != 'pending'],
                'partition': json.loads(json.dumps(queue['partition'])), 'problem_hash': queue['problem_hash'],
                'options': json.loads(json.dumps(options)), 'resumed': restored,
                'resource_interruptions': list(resource_interruptions),
                'checkpoint_slices_resumed': used_checkpoint,
                'proof_scope': 'remaining_checkpoint_frontier' if conditional else 'entire_supplied_problem',
                'coverage_verified': outcome == 'infeasible' and not conditional and all(t['coverage_verified'] for t in tickets),
                'solution_verified': board is not None,
                'queue_path': None if path is None else str(path)}

    def accept(ordinal, result):
        nonlocal board, reason
        ticket = tickets[ordinal]
        task = _ticket_problem(problem, queue['partition'], ticket['state'])
        outcome = result.get('outcome')
        if outcome not in ('solved', 'infeasible', 'timeout', 'stopped'):
            raise RuntimeError('Parallel worker returned an invalid outcome')
        for key in ticket['counters']:
            value = result.get(key)
            if type(value) is not int or not 0 <= value < 2**63:
                raise RuntimeError('Parallel worker returned invalid counters')
        candidate = result.get('board')
        if outcome == 'solved':
            if not check_solution(task, candidate) or not check_solution(problem, candidate):
                raise AssertionError('Parallel worker solution failed independent validation')
            ticket.update(status='solved', checkpoint=None, board=list(candidate), coverage_verified=False)
            board, reason = list(candidate), 'solved'
        elif outcome == 'infeasible':
            if candidate is not None:
                raise RuntimeError('Infeasible worker must not return a board')
            ticket.update(status='infeasible', checkpoint=None, board=None,
                          coverage_verified=bool(result.get('coverage_verified')))
        else:
            if candidate is not None:
                raise RuntimeError('Unfinished worker must not return a board')
            checkpoint = _check_checkpoint(result.get('checkpoint'), task, _ticket_seed(seed, ordinal), queue['options_hash'])
            if ticket['checkpoint'] is not None and _frontier(ticket['checkpoint']) == _frontier(checkpoint):
                ticket['slice_seconds'] = min(MAX_SLICE_SECONDS, ticket['slice_seconds'] * 2)
            ticket['checkpoint'] = checkpoint
            pending.append(ordinal)
        resource = result.get('resource_interruption')
        if resource not in (None, 'memory_limit'):
            raise RuntimeError('Invalid worker resource interruption')
        ticket['resource_interruption'] = resource
        if resource is not None:
            resource_interruptions.append({'ticket_id': ticket['id'], 'reason': resource})
            if reason != 'solved':
                reason = 'stopped'
        ticket['slices'] += 1
        ticket['counters'] = {key: result[key] for key in ticket['counters']}
        _save(path, queue)

    if not restored:
        _save(path, queue)
    if board is not None:
        reason = 'solved'
    elif not pending:
        reason = 'infeasible'
    elif should_stop is not None and should_stop():
        reason = 'stopped'
    elif time.monotonic() >= deadline:
        reason = 'timeout'
    if reason is None:
        context = multiprocessing.get_context('spawn')
        stop_event = context.Event()
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                   initializer=_initialize_worker, initargs=(stop_event,))
        try:
            while pending or active:
                if reason is None:
                    if should_stop is not None and should_stop():
                        reason = 'stopped'
                    elif time.monotonic() >= deadline:
                        reason = 'timeout'
                if reason is not None:
                    stop_event.set()
                while reason is None and pending and len(active) < workers:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        reason = 'timeout'
                        stop_event.set()
                        break
                    ordinal = pending.popleft()
                    ticket = tickets[ordinal]
                    task = _ticket_problem(problem, queue['partition'], ticket['state'])
                    used_checkpoint = used_checkpoint or ticket['checkpoint'] is not None
                    future = pool.submit(_run_ticket, task, min(ticket['slice_seconds'], remaining),
                                         _ticket_seed(seed, ordinal), ticket['checkpoint'], options)
                    active[future] = ordinal
                if not active:
                    break
                done, _ = wait(tuple(active), timeout=.05, return_when=FIRST_COMPLETED)
                for future in done:
                    ordinal = active.pop(future)
                    accept(ordinal, future.result())
                now = time.monotonic()
                if on_progress is not None and now - last_progress >= .25:
                    last_progress = now
                    on_progress(report(reason or 'running'))
            if reason is None:
                reason = 'infeasible'
        finally:
            stop_event.set()
            pool.shutdown(wait=True, cancel_futures=True)
            _save(path, queue)
    result = report(reason)
    if path is None:
        result['queue'] = _seal(queue)
    if on_progress is not None:
        on_progress(report(reason))
    return result


@contextmanager
def _queue_lock(path):
    """A persistent adjacent lock file prevents simultaneous queue ownership."""
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + '.lock')
    with lock_path.open('a+b') as stream:
        if lock_path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        acquired = False
        try:
            if os.name == 'nt':
                import msvcrt
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise RuntimeError('Parallel queue is already in use') from exc
            else:
                import fcntl
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise RuntimeError('Parallel queue is already in use') from exc
            acquired = True
            yield
        finally:
            if acquired:
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def solve(problem, seconds=60, seed=0, workers=1, should_stop=None, on_progress=None,
          queue_path=None, options=None):
    """Run bounded, disjoint CPU search tickets; see _solve for proof scope.

    An explicit queue_path also creates an adjacent .lock file. A second run
    cannot use the same queue concurrently. Frozen executables must invoke
    multiprocessing.freeze_support() before their command-line entry point.
    """
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Parallel DFS supports one to four workers')
    if type(seed) is not int or not -(2**63) <= seed < 2**64:
        raise ValueError('seed must be a bounded integer')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    if on_progress is not None and not callable(on_progress):
        raise ValueError('on_progress must be callable')
    options = exact_dfs.normalize_options(options)
    path = None if queue_path is None else Path(queue_path)
    with _queue_lock(path):
        return _solve(problem, seconds, seed, workers, should_stop, on_progress, path, options)

