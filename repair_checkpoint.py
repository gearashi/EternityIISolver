"""Strict bounded JSON snapshots for the structured repair portfolio.

These checkpoints describe heuristic working state, not an exhaustive frontier.
Only completed matching neighborhoods belong in the FIFO; regional timeouts are
never evidence that any larger search region has been exhausted.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import hmac
import json
import math
from pathlib import Path
import random
import re
from types import MappingProxyType

from exact_model import check_solution, constraint_edges, grid_edges, problem_hash

SCHEMA_VERSION = 1
ALGORITHM = 'structured-repair-v2'
MAX_BYTES = 8 * 1024 * 1024
MAX_SEEN = 4096
MAX_BANK = 64
MAX_RECENT = 12
MAX_COUNTER = 2**53 - 1
_HEX = re.compile(r'[0-9a-f]{64}\Z')
COUNT_KEYS = frozenset(('diversification_attempts', 'diversifications', 'tabu_rejections',
                        'region_calls', 'region_improvements', 'region_seconds'))
STATE_KEYS = frozenset(('best', 'best_score', 'bank', 'bank_scores', 'peaks', 'stalls',
                       'recent', 'rng_state', 'completed', 'cycle_calls', 'rounds',
                       'batches', 'counts', 'seen', 'elapsed_total', 'policy_state'))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _copy_json(value, *, allow_tuples=False, max_bytes=MAX_BYTES):
    """Reject non-JSON aliases, recursion, excessive size/depth and nonfinite data."""
    nodes = 0
    def walk(item, depth):
        nonlocal nodes
        nodes += 1
        if nodes > 500000 or depth > 12:
            raise ValueError('Repair checkpoint JSON exceeds structural bounds')
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if abs(item) > 2**64:
                raise ValueError('Repair checkpoint integer exceeds bounds')
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError('Repair checkpoint contains nonfinite numbers')
            return item
        if type(item) is str:
            if len(item) > 4096:
                raise ValueError('Repair checkpoint string exceeds bounds')
            return item
        if type(item) is list or (allow_tuples and type(item) is tuple):
            if len(item) > MAX_SEEN:
                raise ValueError('Repair checkpoint array exceeds bounds')
            return [walk(child, depth + 1) for child in item]
        if type(item) is dict:
            if len(item) > 256 or any(type(key) is not str or len(key) > 128 for key in item):
                raise ValueError('Repair checkpoint object keys exceed bounds')
            return {key: walk(child, depth + 1) for key, child in item.items()}
        raise ValueError('Repair checkpoint contains non-JSON data')
    result = walk(value, 0)
    if len(_canonical(result)) > max_bytes:
        raise ValueError('Repair checkpoint exceeds its byte limit')
    return result


def _integer(value, name, maximum=MAX_COUNTER):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError('Invalid repair checkpoint ' + name)
    return value


def _duration(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e16:
        raise ValueError('Invalid repair checkpoint ' + name)
    return value


def restore_rng_state(value):
    """Decode the supported Random version-3 state without pickle or execution."""
    if type(value) not in (tuple, list) or len(value) != 3 or type(value[0]) is not int or value[0] != 3:
        raise ValueError('Invalid repair checkpoint random state version')
    words, gaussian = value[1], value[2]
    if type(words) not in (tuple, list) or len(words) != 625:
        raise ValueError('Invalid repair checkpoint random state length')
    for word in words[:-1]:
        _integer(word, 'random state word', 2**32 - 1)
    _integer(words[-1], 'random state index', 624)
    if gaussian is not None and (type(gaussian) not in (int, float) or not math.isfinite(gaussian)):
        raise ValueError('Invalid repair checkpoint Gaussian state')
    result = (3, tuple(words), gaussian)
    rng = random.Random()
    rng.setstate(result)
    return result


def neighborhood_key(board, cells, tie_seed=None):
    """Identity of one completed matching request, including its tie objective."""
    if type(board) not in (list, tuple) or not 1 <= len(board) <= 256:
        raise ValueError('Invalid neighborhood board')
    if any(type(state) is not int or not 0 <= state < 4 * len(board) for state in board):
        raise ValueError('Invalid neighborhood placement')
    if type(cells) not in (list, tuple) or len(cells) > len(board):
        raise ValueError('Invalid neighborhood cells')
    if any(type(cell) is not int or not 0 <= cell < len(board) for cell in cells) or len(set(cells)) != len(cells):
        raise ValueError('Invalid neighborhood cell index')
    if tie_seed is not None:
        _integer(tie_seed, 'tie seed', 2**32 - 1)
    return hashlib.sha256(_canonical([list(board), list(cells), tie_seed])).hexdigest()


def load_checkpoint(path):
    """Read only a bounded existing file; reject duplicate keys and JSON aliases."""
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('Repair checkpoint exceeds its byte limit')
    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Repair checkpoint contains duplicate object keys')
            value[key] = item
        return value
    def invalid_constant(value):
        raise ValueError('Repair checkpoint contains invalid JSON constant ' + value)
    try:
        value = json.loads(raw, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('Repair checkpoint JSON cannot be decoded safely') from exc
    return _copy_json(value)


class CheckpointCodec:
    """Validate every restored placement and bind state to exact settings/rules."""
    def __init__(self, problem, settings):
        if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
            raise ValueError('Repair checkpoints require the complete grid')
        self.settings = _copy_json(settings, max_bytes=16384)
        if type(self.settings) is not dict or 'seed' not in self.settings:
            raise ValueError('Repair checkpoint settings require a seed')
        _integer(self.settings['seed'], 'seed', 2**32 - 1)
        self._source = problem
        self.problem_hash = problem_hash(problem)
        self.structure = replace(problem, faces=tuple(tuple(row) for row in problem.faces),
            domains=tuple(tuple(row) for row in problem.domains),
            fixed=MappingProxyType(dict(problem.fixed)), edges=())
        self.edges = grid_edges(problem.size)
        self.max_score = len(self.edges)
        self.fingerprint = hashlib.sha256(_canonical([ALGORITHM, self.problem_hash, self.settings])).hexdigest()

    def _state(self, value):
        if type(value) is not dict or set(value) != STATE_KEYS:
            raise ValueError('Invalid repair checkpoint state fields')
        def board(item):
            if type(item) is not list or not check_solution(self.structure, item):
                raise ValueError('Repair checkpoint board violates pieces, domains, frame or clues')
            return sum(self.structure.faces[item[a]][sa] == self.structure.faces[item[b]][sb]
                       for a, b, sa, sb in self.edges)
        best_score = _integer(value['best_score'], 'best score', self.max_score)
        if board(value['best']) != best_score:
            raise ValueError('Repair checkpoint best score disagrees with board')
        bank = value['bank']
        if type(bank) is not list or not 1 <= len(bank) <= MAX_BANK:
            raise ValueError('Invalid repair checkpoint parent count')
        for name in ('bank_scores', 'peaks', 'stalls', 'recent'):
            if type(value[name]) is not list or len(value[name]) != len(bank):
                raise ValueError('Invalid repair checkpoint parallel parent data')
        for i, parent in enumerate(bank):
            actual = board(parent)
            bank_score = _integer(value['bank_scores'][i], 'parent score', best_score)
            peak = _integer(value['peaks'][i], 'parent peak', best_score)
            if actual != bank_score or peak < bank_score:
                raise ValueError('Repair checkpoint parent score/peak mismatch')
            _integer(value['stalls'][i], 'plateau count')
            recent = value['recent'][i]
            if type(recent) is not list or not 1 <= len(recent) <= MAX_RECENT:
                raise ValueError('Invalid repair checkpoint recent history')
            for prior in recent:
                if board(prior) > best_score:
                    raise ValueError('Repair checkpoint history exceeds saved best')
        restore_rng_state(value['rng_state'])
        for name in ('completed', 'cycle_calls', 'rounds', 'batches'):
            _integer(value[name], name)
        if value['rounds'] < value['completed']:
            raise ValueError('Repair checkpoint completed count exceeds attempts')
        counts = value['counts']
        if type(counts) is not dict or set(counts) != COUNT_KEYS:
            raise ValueError('Invalid repair checkpoint counter fields')
        for name in COUNT_KEYS - {'region_seconds'}:
            _integer(counts[name], name)
        _duration(counts['region_seconds'], 'region seconds')
        if counts['diversifications'] > counts['diversification_attempts'] or counts['region_improvements'] > counts['region_calls']:
            raise ValueError('Repair checkpoint success counters exceed attempts')
        _duration(value['elapsed_total'], 'elapsed time')
        if counts['region_seconds'] > value['elapsed_total'] + 1e-6:
            raise ValueError('Repair checkpoint regional time exceeds elapsed time')
        seen = value['seen']
        if type(seen) is not list or len(seen) > MAX_SEEN or len(seen) > value['completed']:
            raise ValueError('Invalid repair checkpoint completed-neighborhood history')
        if any(type(key) is not str or not _HEX.fullmatch(key) for key in seen) or len(set(seen)) != len(seen):
            raise ValueError('Invalid repair checkpoint completed-neighborhood digest')
        if type(value['policy_state']) is not dict:
            raise ValueError('Invalid repair checkpoint policy state')
        _copy_json(value['policy_state'], max_bytes=65536)
        return value

    def _check_source(self):
        if problem_hash(self._source) != self.problem_hash:
            raise ValueError('Repair checkpoint puzzle changed during search')

    def encode(self, state):
        self._check_source()
        state = self._state(_copy_json(state, allow_tuples=True))
        payload = {'schema_version': SCHEMA_VERSION, 'algorithm': ALGORITHM,
                   'problem_hash': self.problem_hash, 'settings': self.settings,
                   'fingerprint': self.fingerprint, 'state': state}
        payload['checksum'] = hashlib.sha256(_canonical(payload)).hexdigest()
        return _copy_json(payload)

    def decode(self, payload):
        self._check_source()
        payload = _copy_json(payload)
        expected = {'schema_version', 'algorithm', 'problem_hash', 'settings', 'fingerprint', 'state', 'checksum'}
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError('Invalid repair checkpoint envelope')
        checksum = payload.pop('checksum')
        if type(checksum) is not str or not _HEX.fullmatch(checksum) or not hmac.compare_digest(checksum, hashlib.sha256(_canonical(payload)).hexdigest()):
            raise ValueError('Repair checkpoint checksum mismatch')
        if type(payload['schema_version']) is not int or payload['schema_version'] != SCHEMA_VERSION or payload['algorithm'] != ALGORITHM:
            raise ValueError('Unsupported repair checkpoint algorithm/version')
        if payload['problem_hash'] != self.problem_hash or payload['fingerprint'] != self.fingerprint or payload['settings'] != self.settings:
            raise ValueError('Repair checkpoint puzzle/settings mismatch')
        return self._state(payload['state'])
