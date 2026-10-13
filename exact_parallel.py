"""Offline CPU workers on exhaustive, disjoint DFS partitions.

Initial tickets fix distinct states of one cell; idle workers receive disjoint
regions extracted from a yielded ticket's remaining DFS frontier. Workers cannot access queue
files: only the parent optionally persists an explicitly supplied queue_path.
Checksums detect corruption, not forged history. Exhaustion after any checkpoint
resume is conservatively reported only for the recorded remaining frontier.
"""
from __future__ import annotations

from collections import OrderedDict, deque
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
from exact_model import ExactProblem, check_solution, constraint_edges, initial_domains, problem_hash

QUEUE_VERSION = 2
MAX_TICKETS = 1024
MAX_SPLIT_FANOUT = 256
MAX_QUEUE_BYTES = 64 * 1024 * 1024
INITIAL_SLICE_SECONDS = 2.0
MAX_SLICE_SECONDS = 30.0
WORKER_MEMORY_MB = 512
USE_TASK_CACHE = True  # Bounded setup reuse; ablate with _TaskCache(enabled=False).
MAX_CACHED_TASKS = 64
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


def _partition(problem, *, _base_domains=None):
    domains = initial_domains(problem) if _base_domains is None else _base_domains
    eligible = [cell for cell, domain in enumerate(domains) if len(domain) > 1 and cell not in problem.fixed]
    if any(not domain for domain in domains) or not eligible:
        return {'cell': None, 'states': []}
    cell = min(eligible, key=lambda c: (len(domains[c]), c))
    return {'cell': cell, 'states': sorted(domains[cell])}


def _ticket_problem(problem, partition, state, path=(), *, _base_domains=None):
    domains = list(initial_domains(problem) if _base_domains is None else _base_domains)
    if partition['cell'] is not None:
        domains[partition['cell']] = tuple(s for s in domains[partition['cell']] if s == state)
    n = problem.size ** 2
    edges = tuple(constraint_edges(problem))
    if not isinstance(path, (list, tuple)) or len(path) > 4 * n:
        raise ValueError('Invalid parallel ticket decision path')
    for decision in path:
        if not isinstance(decision, (list, tuple)) or len(decision) != 3:
            raise ValueError('Invalid parallel ticket decision')
        kind, target, choice = decision
        if type(target) is not int or type(choice) is not int:
            raise ValueError('Invalid parallel ticket decision values')
        if kind == 'cell' and 0 <= target < n and 0 <= choice < 4 * n:
            domains[target] = tuple(s for s in domains[target] if s == choice)
        elif kind == 'piece' and 0 <= target < n and 0 <= choice < n:
            domains[choice] = tuple(s for s in domains[choice] if s // 4 == target)
        elif kind == 'edge' and 0 <= target < len(edges) and 0 <= choice < 256:
            a, b, sa, sb = edges[target]
            domains[a] = tuple(s for s in domains[a] if problem.faces[s][sa] == choice)
            domains[b] = tuple(s for s in domains[b] if problem.faces[s][sb] == choice)
        else:
            raise ValueError('Invalid parallel ticket decision kind/range')
    return ExactProblem(problem.size, problem.faces, dict(problem.fixed), tuple(domains), problem.edges)



def _snapshot_problem(problem):
    return ExactProblem(problem.size, tuple(tuple(face) for face in problem.faces),
                        dict(problem.fixed), tuple(tuple(domain) for domain in problem.domains),
                        None if problem.edges is None else tuple(tuple(edge) for edge in problem.edges))


class _TaskCache:
    """One solve's detached inputs and bounded, exact restricted-task reuse.

    Unchanged domain tuples share the normalized root storage. At most64
    restricted tasks and1024 hashes are retained; eviction only recomputes data.
    No checkpoint, cursor, inferred domain or proof is memoized here.
    """
    def __init__(self, problem, *, enabled=True):
        self.started = time.monotonic()
        self.problem = _snapshot_problem(problem)
        self.enabled = enabled
        self.base_domains = initial_domains(self.problem) if enabled else None
        self.partition = _partition(self.problem, _base_domains=self.base_domains)
        self.root_identity = problem_hash(self.problem)
        self.hint_identity = exact_dfs._hint_data(None, self.problem.size ** 2)[1]
        self.tasks, self.identities = OrderedDict(), OrderedDict()
        self.task_hits = self.task_builds = self.hash_hits = self.hash_builds = 0

    def _key(self, state, path):
        n = self.problem.size ** 2
        if type(state) is not int and state is not None:
            raise ValueError('Invalid parallel root state')
        if not isinstance(path, (list, tuple)) or len(path) > 4 * n:
            raise ValueError('Invalid parallel ticket decision path')
        result = []
        edges = len(constraint_edges(self.problem))
        for decision in path:
            if not isinstance(decision, (list, tuple)) or len(decision) != 3:
                raise ValueError('Invalid parallel ticket decision')
            kind, target, choice = decision
            if type(target) is not int or type(choice) is not int:
                raise ValueError('Invalid parallel ticket decision values')
            if not ((kind == 'cell' and 0 <= target < n and 0 <= choice < 4 * n) or
                    (kind == 'piece' and 0 <= target < n and 0 <= choice < n) or
                    (kind == 'edge' and 0 <= target < edges and 0 <= choice < 256)):
                raise ValueError('Invalid parallel ticket decision kind/range')
            result.append((kind, target, choice))
        return state, tuple(result)

    def task(self, state, path=()):
        key = self._key(state, path)
        if self.enabled and key in self.tasks:
            self.task_hits += 1
            self.tasks.move_to_end(key)
            return self.tasks[key]
        task = _ticket_problem(self.problem, self.partition, state, key[1],
                               _base_domains=self.base_domains)
        self.task_builds += 1
        if self.enabled:
            self.tasks[key] = task
            if len(self.tasks) > MAX_CACHED_TASKS:
                self.tasks.popitem(last=False)
        return task

    def identity(self, state, path=(), *, task=None):
        key = self._key(state, path)
        if self.enabled and key in self.identities:
            self.hash_hits += 1
            self.identities.move_to_end(key)
            return self.identities[key]
        identity = problem_hash(self.task(state, path) if task is None else task)
        self.hash_builds += 1
        if self.enabled:
            self.identities[key] = identity
            if len(self.identities) > MAX_TICKETS:
                self.identities.popitem(last=False)
        return identity

    def stats(self):
        return {'enabled': self.enabled, 'task_builds': self.task_builds,
                'task_hits': self.task_hits, 'hash_builds': self.hash_builds,
                'hash_hits': self.hash_hits, 'retained_tasks': len(self.tasks),
                'retained_hashes': len(self.identities)}


def _resolve_task(problem, partition, state, path, task_cache):
    if task_cache is None:
        return _ticket_problem(problem, partition, state, path)
    if task_cache.problem is not problem or task_cache.partition != partition:
        raise ValueError('Task cache does not belong to this parallel search')
    return task_cache.task(state, path)

def _frontier_paths(checkpoint):
    """Disjoint conjunctions for every unexplored sibling and active node.

    Replaying the raw decisions may repeat sound propagation, but never visits
    a previously exhausted sibling. No stored inferred domains are trusted.
    Paths also support piece-location and edge-color branching.
    """
    paths, prefix = [], []
    for index, frame in enumerate(checkpoint['stack']):
        if frame['phase'] == 'pending':
            paths.append(list(prefix))
            break
        for choice in frame['candidates'][frame['next']:]:
            paths.append(prefix + [[frame['kind'], frame['cell'], choice]])
        if index < len(checkpoint['stack']) - 1:
            prefix = prefix + [[frame['kind'], frame['cell'], frame['candidates'][frame['next'] - 1]]]
    return paths


def _blank_ticket(identifier, state, parent=None, path=()):
    return {'id': identifier, 'state': state, 'parent': parent, 'path': list(path), 'children': [],
            'status': 'pending', 'checkpoint': None, 'slices': 0,
            'slice_seconds': INITIAL_SLICE_SECONDS, 'coverage_verified': False,
            'board': None, 'resource_interruption': None,
            'counters': {'branches': 0, 'conflicts': 0, 'max_depth': 0}}


def _try_split(queue, ordinal, problem, *, seconds=30, should_stop=None, task_cache=None):
    """Transfer only yielded work; fail closed on growth/serialization caps.

    The retired parent retains its historical counters and exact split basis.
    Children start counters at zero, own disjoint decision paths, and build
    independently. Their future checkpoints bind the restricted child problem.
    """
    tickets = queue['tickets']
    parent = tickets[ordinal]
    if parent['status'] != 'pending' or parent['checkpoint'] is None:
        return []
    relative = _frontier_paths(parent['checkpoint'])
    if not 2 <= len(relative) <= MAX_SPLIT_FANOUT or len(tickets) + len(relative) > MAX_TICKETS:
        return []
    if any(len(parent['path']) + len(path) > 4 * queue['cell_count'] for path in relative):
        return []
    _semantic_checkpoint(queue, ordinal, problem, seconds=seconds, should_stop=should_stop, task_cache=task_cache)
    indices = list(range(len(tickets), len(tickets) + len(relative)))
    children = [_blank_ticket('split-%d' % index, parent['state'], ordinal, parent['path'] + path)
                for index, path in zip(indices, relative)]
    changed_parent = dict(parent, status='split', children=indices)
    prospective = dict(queue, tickets=tickets[:ordinal] + [changed_parent] + tickets[ordinal + 1:] + children)
    if len(_canonical(_seal(prospective))) + 1 > MAX_QUEUE_BYTES:
        return []
    tickets[ordinal] = changed_parent
    tickets.extend(children)
    return indices


def _new_queue(problem, seed, options, *, task_cache=None):
    partition = _partition(problem) if task_cache is None else task_cache.partition
    tickets = []
    for state in partition['states'] if partition['cell'] is not None else (None,):
        identifier = 'whole' if state is None else 'cell-%d-state-%d' % (partition['cell'], state)
        tickets.append(_blank_ticket(identifier, state))
    return {'version': QUEUE_VERSION, 'dfs_checkpoint_version': exact_dfs.CHECKPOINT_VERSION,
            'ordering_version': exact_dfs.ORDERING_VERSION,
            'problem_hash': problem_hash(problem) if task_cache is None else task_cache.root_identity,
            'cell_count': problem.size ** 2,
            'options': options, 'options_hash': _options_identity(options), 'seed': seed,
            'partition': partition, 'tickets': tickets}


def _check_checkpoint(checkpoint, problem, seed, options_hash, *, identity=None, hints_hash=None):
    if hints_hash is None:
        hints_hash = exact_dfs._hint_data(None, problem.size ** 2)[1]
    if identity is None:
        identity = problem_hash(problem)
    return exact_dfs._checked_checkpoint(checkpoint, identity, seed, problem.size ** 2,
                                         hints_hash, options_hash)


def _semantic_checkpoint(queue, ordinal, problem, *, seconds=30, should_stop=None, task_cache=None):
    """Replay, never search, before converting a checkpoint into new ownership.

    Structural checksum validation cannot detect a deleted branch alternative.
    Replay proves that every stored branch still partitions its live domain.
    Interruption propagates before any queue mutation or child submission.
    """
    ticket = queue['tickets'][ordinal]
    task = _resolve_task(problem, queue['partition'], ticket['state'], ticket['path'], task_cache)
    search = exact_dfs._Search(task, seconds, _ticket_seed(queue['seed'], ordinal),
                              should_stop, None, ticket['checkpoint'], None, 5,
                              options=queue['options'])
    search.restore(search.build())


def _validate_queue(value, expected, problem, *, semantic=True, task_cache=None):
    if not isinstance(value, dict) or set(value) != set(expected) | {'checksum'}:
        raise ValueError('Invalid parallel queue fields')
    checksum = value['checksum']
    if not isinstance(checksum, str) or checksum != _seal(value)['checksum']:
        raise ValueError('Parallel queue checksum mismatch')
    for key in expected:
        if key != 'tickets' and _canonical(value[key]) != _canonical(expected[key]):
            raise ValueError('Parallel queue problem, options, seed, or partition mismatch: ' + key)
    tickets = value['tickets']
    roots = expected['tickets']
    if not isinstance(tickets, list) or not len(roots) <= len(tickets) <= MAX_TICKETS:
        raise ValueError('Parallel queue must contain every partition exactly once within its cap')
    ownership = {}
    for ordinal, ticket in enumerate(tickets):
        template = roots[ordinal] if ordinal < len(roots) else _blank_ticket('split-%d' % ordinal, None)
        if not isinstance(ticket, dict) or set(ticket) != set(template):
            raise ValueError('Invalid parallel ticket fields')
        if ordinal < len(roots):
            if any(_canonical(ticket[k]) != _canonical(template[k]) for k in ('id', 'state', 'parent', 'path')):
                raise ValueError('Parallel root ticket identity/order mismatch')
        else:
            parent_id = ticket['parent']
            if type(parent_id) is not int or not 0 <= parent_id < ordinal or ordinal not in ownership:
                raise ValueError('Parallel child lacks unique earlier ownership')
            owner, expected_path = ownership[ordinal]
            if (owner != parent_id or ticket['id'] != 'split-%d' % ordinal
                    or _canonical(ticket['state']) != _canonical(tickets[parent_id]['state'])
                    or _canonical(ticket['path']) != _canonical(expected_path)):
                raise ValueError('Parallel child decision ownership mismatch')
        if ticket['status'] not in ('pending', 'infeasible', 'solved', 'split'):
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
        if counters['max_depth'] > 4 * problem.size ** 2:
            raise ValueError('Invalid parallel ticket depth')
        task = _resolve_task(problem, value['partition'], ticket['state'], ticket['path'], task_cache)
        if ticket['status'] in ('pending', 'split'):
            if ticket['board'] is not None or ticket['coverage_verified']:
                raise ValueError('Unfinished/split ticket cannot claim completion')
            if ticket['checkpoint'] is not None:
                ticket['checkpoint'] = _check_checkpoint(ticket['checkpoint'], task, _ticket_seed(value['seed'], ordinal), value['options_hash'],
                    identity=None if task_cache is None else task_cache.identity(ticket['state'], ticket['path'], task=task),
                    hints_hash=None if task_cache is None else task_cache.hint_identity)
                if any(ticket['checkpoint']['counters'][key] != counters[key] for key in counters):
                    raise ValueError('Parallel checkpoint counters disagree with ticket')
        elif ticket['checkpoint'] is not None:
            raise ValueError('Completed ticket must not contain a checkpoint')
        children = ticket['children']
        if ticket['status'] == 'split':
            if ticket['checkpoint'] is None or not isinstance(children, list):
                raise ValueError('Split ticket requires its exact split basis and children')
            paths = _frontier_paths(ticket['checkpoint'])
            if not 2 <= len(paths) <= MAX_SPLIT_FANOUT or len(children) != len(paths):
                raise ValueError('Split must retain every remaining frontier region')
            if any(type(child) is not int or not max(ordinal + 1, len(roots)) <= child < len(tickets) or child in ownership for child in children) or len(set(children)) != len(children):
                raise ValueError('Split child ownership overlaps or is invalid')
            for child, path in zip(children, paths):
                ownership[child] = (ordinal, ticket['path'] + path)
        elif children != []:
            raise ValueError('Only split tickets may own children')
        if ticket['status'] == 'solved':
            if not check_solution(task, ticket['board']) or not check_solution(problem, ticket['board']):
                raise ValueError('Saved parallel solution failed independent validation')
        elif ticket['board'] is not None:
            raise ValueError('Unsolved ticket must not contain a board')
    if semantic:
        for ordinal, ticket in enumerate(tickets):
            if ticket['status'] == 'split':
                _semantic_checkpoint(value, ordinal, problem, task_cache=task_cache)
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
          queue_path=None, options=None, *, task_cache=None):
    """Explore disjoint initial and dynamically split tickets with at most four spawned workers.

    A queue is read/written only at an explicitly supplied path. Existing queues
    must match the problem, options and seed exactly; incompatible files are
    preserved. Queue v2 records each split basis and complete child ownership.
    Completed and split parent tickets are never scheduled again. Cancellation is
    cooperative and every unfinished ticket retains its latest valid frontier.
    Restored completed counters/cursors are untrusted history, not a global
    infeasibility proof. Return ``queue`` for inspection when queue_path is None.
    """
    started = time.monotonic() if task_cache is None else task_cache.started
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Parallel DFS supports one to four workers')
    if type(seed) is not int or not -(2**63) <= seed < 2**64:
        raise ValueError('seed must be a bounded integer')
    options = exact_dfs.normalize_options(options)
    deadline = started + seconds
    if task_cache is None:
        task_cache = _TaskCache(problem, enabled=USE_TASK_CACHE)
    problem = task_cache.problem
    expected = _new_queue(problem, seed, options, task_cache=task_cache)
    path = None if queue_path is None else Path(queue_path)
    restored = path is not None and path.exists()
    queue = _validate_queue(_load(path), expected, problem, semantic=False, task_cache=task_cache) if restored else expected
    tickets = queue['tickets']
    used_checkpoint = any(ticket['checkpoint'] is not None for ticket in tickets)
    active = {}
    pending = deque(index for index, ticket in enumerate(tickets) if ticket['status'] == 'pending')
    reason = None
    board = next((ticket['board'] for ticket in tickets if ticket['status'] == 'solved'), None)
    last_progress = started
    resource_interruptions = []

    def report(outcome):
        leaves = [ticket for ticket in tickets if ticket['status'] != 'split']
        complete = sum(ticket['status'] != 'pending' for ticket in leaves)
        conditional = restored or used_checkpoint
        return {'backend': 'dfs-parallel', 'outcome': outcome, 'board': None if board is None else list(board),
                'workers': workers, 'elapsed_seconds': time.monotonic() - started,
                'branches': sum(t['counters']['branches'] for t in tickets),
                'conflicts': sum(t['counters']['conflicts'] for t in tickets),
                'max_depth': max(t['counters']['max_depth'] for t in tickets),
                'tickets_total': len(leaves), 'tickets_completed': complete,
                'split_tickets': len(tickets) - len(leaves), 'queue_records': len(tickets),
                'active_ticket_ids': sorted(tickets[index]['id'] for index in active.values()),
                'completed_ticket_ids': [t['id'] for t in leaves if t['status'] != 'pending'],
                'partition': json.loads(json.dumps(queue['partition'])), 'problem_hash': queue['problem_hash'],
                'options': json.loads(json.dumps(options)), 'resumed': restored,
                'resource_interruptions': list(resource_interruptions),
                'checkpoint_slices_resumed': used_checkpoint,
                'proof_scope': 'remaining_checkpoint_frontier' if conditional else 'entire_supplied_problem',
                'coverage_verified': outcome == 'infeasible' and not conditional and all(t['coverage_verified'] for t in leaves),
                'solution_verified': board is not None, 'setup_cache': task_cache.stats(),
                'queue_path': None if path is None else str(path)}

    def accept(ordinal, result):
        nonlocal board, reason
        ticket = tickets[ordinal]
        task = _resolve_task(problem, queue['partition'], ticket['state'], ticket['path'], task_cache)
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
            # Stable selection if several workers finish before cancellation.
            board = list(next(t['board'] for t in tickets if t['status'] == 'solved'))
            reason = 'solved'
        elif outcome == 'infeasible':
            if candidate is not None:
                raise RuntimeError('Infeasible worker must not return a board')
            ticket.update(status='infeasible', checkpoint=None, board=None,
                          coverage_verified=bool(result.get('coverage_verified')))
        else:
            if candidate is not None:
                raise RuntimeError('Unfinished worker must not return a board')
            checkpoint = _check_checkpoint(result.get('checkpoint'), task, _ticket_seed(seed, ordinal), queue['options_hash'],
                identity=task_cache.identity(ticket['state'], ticket['path'], task=task), hints_hash=task_cache.hint_identity)
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

    if restored:
        try:
            for ordinal, ticket in enumerate(tickets):
                if ticket['status'] == 'split':
                    _semantic_checkpoint(queue, ordinal, problem,
                                         seconds=max(0., deadline - time.monotonic()),
                                         should_stop=should_stop, task_cache=task_cache)
        except exact_dfs._Interrupted as interrupted:
            reason = interrupted.outcome
            board = None
    else:
        _save(path, queue)
    if reason is None:
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
                # No ticket is split while owned by a running worker. At a
                # slice boundary expand only if there is otherwise idle capacity.
                while reason is None and pending and len(active) + len(pending) < workers:
                    expanded = False
                    for ordinal in tuple(pending):
                        try:
                            children = _try_split(queue, ordinal, problem,
                                                  seconds=max(0., deadline - time.monotonic()),
                                                  should_stop=should_stop, task_cache=task_cache)
                        except exact_dfs._Interrupted as interrupted:
                            reason = interrupted.outcome
                            stop_event.set()
                            break
                        if children:
                            pending.remove(ordinal)
                            pending.extend(children)
                            used_checkpoint = True
                            _save(path, queue)
                            expanded = True
                            break
                    if not expanded:
                        break
                while reason is None and pending and len(active) < workers:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        reason = 'timeout'
                        stop_event.set()
                        break
                    ordinal = pending.popleft()
                    ticket = tickets[ordinal]
                    task = _resolve_task(problem, queue['partition'], ticket['state'], ticket['path'], task_cache)
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
    task_cache = _TaskCache(problem, enabled=USE_TASK_CACHE)
    path = None if queue_path is None else Path(queue_path)
    with _queue_lock(path):
        return _solve(task_cache.problem, seconds, seed, workers, should_stop, on_progress, path, options,
                      task_cache=task_cache)

