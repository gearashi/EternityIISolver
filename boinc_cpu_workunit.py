"""Read-only inspection of the observed production CPU workunit profile.

No solver, subprocess, network request, BOINC result, or checkpoint is created.
Ticket arithmetic and prefix geometry are checked separately from the opaque
CPU cfg_hash and production DFS/control-flow compatibility.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import shlex
from typing import Iterator
import xml.etree.ElementTree as ET

from validator import PuzzleBundle, load_bundle

MAX_SMALL = 131072
MAX_CATALOG_BYTES = 16 * 1024 * 1024
MAX_ROOTS = 100000
MAX_TICKETS = 10000000
PREFIX_CELLS = (240, 241, 242, 224, 225, 226, 208, 209, 210)
# Independently extracted from the inverse cell->depth table in the pinned
# production 236 executable. This is a mapping reference, not an assertion
# that an arbitrary currently installed executable was authenticated here.
PREFIX_REFERENCE_SHA256 = '7ea0889103958d0fd1af6365ed0fd14e109acd441e5aecbfa2fbe285513ef60c'
INTERLEAVED_SEED_CONTRACT_SHA256 = PREFIX_REFERENCE_SHA256
_U64_MASK = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_JITTER_MULTIPLIER = 0xD1B54A32D192ED03
FORMULA = 'globalticket=jidx*nroots+cidx; (globalticket-jitter_start*nroots)%shard_count=shard_index'
DEDUP_KEY = '(epoch,cfg_hash,shard_index,globalticket)'
LOGICAL_FILES = ('job.xml', '256pieces.txt', 'campaign_catalog.txt',
                 'campaign_hints.txt', 'campaign_manifest.json', 'exploration_allow.txt')


class InspectionError(ValueError):
    pass


def _int(value, name, lo=0, hi=2**32 - 1):
    if type(value) is not int or not lo <= value <= hi:
        raise InspectionError(f'{name} must be an integer in {lo}..{hi}')
    return value


def _decimal(text, name, lo=0, hi=2**32 - 1):
    if not isinstance(text, str) or not re.fullmatch(r'[0-9]{1,20}', text):
        raise InspectionError(f'Invalid decimal {name}')
    return _int(int(text), name, lo, hi)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InspectionError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _json(data):
    def reject(value):
        raise InspectionError(f'Non-finite JSON constant: {value}')
    try:
        result = json.loads(data.decode('utf-8'), object_pairs_hook=_object, parse_constant=reject)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise InspectionError(f'Invalid JSON: {exc}') from exc
    if not isinstance(result, dict):
        raise InspectionError('JSON document must be an object')
    return result


def _xml(data):
    try:
        text = data.decode('utf-8')
    except UnicodeError as exc:
        raise InspectionError('XML must be UTF-8') from exc
    if '\x00' in text or re.search(r'<!\s*(?:DOCTYPE|ENTITY)', text, re.I):
        raise InspectionError('XML declarations/entities are forbidden')
    try:
        return ET.fromstring(text)
    except ET.ParseError as exc:
        raise InspectionError(f'Invalid XML: {exc}') from exc


def _inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _target_name_allowed(logical_name, source_name):
    patterns = {'job.xml': r'job(?:_[A-Za-z0-9_.-]+)?\.xml',
                '256pieces.txt': r'256pieces\.txt',
                'campaign_catalog.txt': r'(?:campaign_catalog|root_catalog(?:_[A-Za-z0-9_.-]+)?|catalog_sha[0-9a-f]{64}|catalog_frame[0-3]_[0-9a-f]{6})\.txt',
                'campaign_hints.txt': r'(?:campaign_hints|hints_[A-Za-z0-9_.-]+)\.txt',
                'campaign_manifest.json': r'(?:campaign_manifest|campaign_frame[0-3]_r[0-9]+)\.json',
                'exploration_allow.txt': r'(?:exploration_allow|root_allow_[A-Za-z0-9_.-]+|allow_eternity_[A-Za-z0-9_.-]+)\.txt'}
    if logical_name.startswith('ticketlog/'):
        return source_name == Path(logical_name).name
    return re.fullmatch(patterns[logical_name], source_name) is not None


def _stable_bytes(path, limit):
    """Two matching bounded reads; reject replacement or concurrent changes."""
    previous = None
    for _ in range(3):
        with path.open('rb') as handle:
            before = os.fstat(handle.fileno())
            if before.st_size > limit:
                raise InspectionError(f'{path.name} exceeds {limit} bytes')
            data = handle.read(limit + 1)
            after = os.fstat(handle.fileno())
        current = path.stat()
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if len(data) > limit:
            raise InspectionError(f'{path.name} exceeds {limit} bytes')
        if signature(before) == signature(after) == signature(current):
            if previous == (signature(current), data):
                return data
            previous = (signature(current), data)
        else:
            previous = None
    raise InspectionError(f'{path.name} changed while being read')


@dataclass(frozen=True)
class Asset:
    logical_name: str
    source_name: str
    sha256: str
    lf_sha256: str
    size: int
    data: bytes = field(repr=False, compare=False)


def _asset(slot, logical_name, project_root, limit):
    # Callers may read only these logical input names or a selected journal
    # manifest. Never enumerate/read initialization, account or client files.
    if logical_name not in LOGICAL_FILES and not re.fullmatch(r'ticketlog/seg_[0-9a-f]{16}\.manifest\.json', logical_name):
        raise InspectionError('Unsupported logical input name')
    initial = slot / logical_name
    current = initial
    links = []
    for depth in range(4):
        resolved = current.resolve(strict=True)
        roots = (slot,) if project_root is None else (slot, project_root)
        if not any(_inside(resolved, root) for root in roots):
            raise InspectionError('Input resolves outside the permitted slot/project roots')
        if not _target_name_allowed(logical_name, resolved.name):
            raise InspectionError('Input target filename is outside the selected puzzle-input profile')
        data = _stable_bytes(resolved, limit)
        if data.lstrip().startswith(b'<soft_link'):
            if project_root is None:
                raise InspectionError('An explicit project_root is required for BOINC soft links')
            element = _xml(data)
            if element.tag != 'soft_link' or element.attrib or len(element):
                raise InspectionError('Malformed BOINC soft link')
            target = (element.text or '').strip()
            if not target or len(target) > 4096 or '\x00' in target:
                raise InspectionError('Invalid BOINC soft-link target')
            links.append((resolved, data))
            current = resolved.parent / target
            if not _inside(current.resolve(strict=True), project_root):
                raise InspectionError('BOINC soft-link target must stay inside project_root')
            continue
        for link, original in links:
            if _stable_bytes(link, MAX_SMALL) != original:
                raise InspectionError('BOINC soft link changed during inspection')
        return Asset(logical_name, resolved.name, hashlib.sha256(data).hexdigest(),
                     hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest(), len(data), data)
    raise InspectionError('Too many BOINC soft links')


@dataclass(frozen=True)
class CpuJob:
    application: str
    command_line: str
    options: tuple[tuple[str, str | bool], ...]
    shard_index: int
    shard_count: int
    jitter_start: int
    jitter_end: int
    seed: int
    epoch: str
    node_cap: int
    seconds: int
    min_save: int


def parse_job(data):
    root = _xml(data)
    if root.tag != 'job_desc' or root.attrib or len(root) != 1 or root[0].tag != 'task':
        raise InspectionError('Expected exactly one BOINC wrapper task')
    task = root[0]
    if task.attrib:
        raise InspectionError('Unsupported wrapper task attributes')
    fields = {}
    permitted = {'application', 'multi_process', 'checkpoint_filename', 'fraction_done_filename',
                 'time_limit', 'command_line', 'stdout_filename'}
    for child in task:
        if child.tag not in permitted or child.tag in fields or child.attrib or len(child):
            raise InspectionError('Unsupported or duplicate wrapper task field')
        fields[child.tag] = (child.text or '').strip()
    if fields.get('application') != 'bw_runner':
        raise InspectionError('Unsupported CPU application; expected bw_runner')
    command = fields.get('command_line', '')
    try:
        arguments = shlex.split(command, posix=True)
    except ValueError as exc:
        raise InspectionError('Invalid wrapper command quoting') from exc
    flags = {'--force-interleave', '--sigma-tail', '--no-sigma-main'}
    valued = {'--shard-index', '--shard-count', '--force-jitter-start', '--force-jitters', '--threads',
              '--node-cap', '--seconds', '--exact-endgame', '--endgame-probe', '--endgame-ladder',
              '--ring-breaks', '--ncon2-direct', '--mrv-dem-inc', '--main-fast', '--min-save', '--breaks',
              '--root-allow', '--seed', '--epoch', '--tourney'}
    options = {}
    index = 0
    while index < len(arguments):
        key = arguments[index]
        if key in options or key not in flags | valued:
            raise InspectionError(f'Unsupported or duplicate CPU argument: {key}')
        if key in flags:
            options[key] = True
            index += 1
        else:
            if index + 1 >= len(arguments) or arguments[index + 1].startswith('--'):
                raise InspectionError(f'Missing CPU argument value: {key}')
            options[key] = arguments[index + 1]
            index += 2
    required = {'--force-interleave', '--shard-index', '--shard-count', '--force-jitter-start',
                '--force-jitters', '--node-cap', '--seconds', '--min-save', '--root-allow', '--seed', '--epoch'}
    if required - options.keys():
        raise InspectionError('CPU workunit is missing required interleaved-ticket arguments')
    if options['--root-allow'] != 'exploration_allow.txt':
        raise InspectionError('Unsupported root-allow logical filename')
    def number(key, lo=0, hi=2**32 - 1):
        return _decimal(options[key], key, lo, hi)
    shard_count = number('--shard-count', 1, 1000000)
    shard_index = number('--shard-index', 0, shard_count - 1)
    # The verified production branch sign-extends the stored start from int32.
    start, end = number('--force-jitter-start', 0, 2**31 - 1), number('--force-jitters')
    if start >= end:
        raise InspectionError('Jitter interval must be nonempty and half-open')
    epoch = options['--epoch']
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', epoch):
        raise InspectionError('Unsupported epoch identifier')
    for key in ('--threads', '--endgame-probe', '--ring-breaks', '--ncon2-direct', '--mrv-dem-inc', '--main-fast'):
        if key in options:
            number(key, 0, 1000000)
    for key, upper in (('--main-fast', 1), ('--endgame-probe', 20)):
        if key in options:
            number(key, 0, upper)
    for key, pattern in (('--breaks', r'\d{1,3}(,\d{1,3})*'),
                         ('--exact-endgame', r'\d{1,3}:\d{1,20}'),
                         ('--endgame-ladder', r'\d{1,3}:\d{1,3}:\d{1,20}(,\d{1,3}:\d{1,3}:\d{1,20})*')):
        if key in options and not re.fullmatch(pattern, options[key]):
            raise InspectionError(f'Invalid retained CPU option {key}')
    if '--exact-endgame' in options:
        depth, cap = options['--exact-endgame'].split(':')
        _decimal(depth, 'exact-endgame depth', 200, 255)
        _decimal(cap, 'exact-endgame node cap', 1, 2**64 - 1)
    if '--tourney' in options and not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', options['--tourney']):
        raise InspectionError('Unsupported tourney name')
    return CpuJob('bw_runner', command, tuple(options.items()), shard_index, shard_count, start, end,
                  number('--seed'), epoch, number('--node-cap', 1, 2**64 - 1),
                  number('--seconds', 1, 8640000), number('--min-save', 0, 480))


def _lines(data):
    try:
        return [(i, line.strip()) for i, line in enumerate(data.decode('utf-8').splitlines(), 1)
                if line.strip() and not line.lstrip().startswith('#')]
    except UnicodeError as exc:
        raise InspectionError('Text input is not UTF-8') from exc


def parse_catalog(data):
    # Catalogue row number is cidx. Do not silently discard comments or blank
    # rows: their treatment by a different CPU parser has not been established.
    try:
        lines = list(enumerate(data.decode('utf-8').splitlines(), 1))
    except UnicodeError as exc:
        raise InspectionError('Catalogue is not UTF-8') from exc
    if not 1 <= len(lines) <= MAX_ROOTS:
        raise InspectionError('Root catalogue has an unsupported row count')
    roots = []
    for line_number, line in lines:
        pairs = line.strip().split(',')
        if len(pairs) != 9 or any(not re.fullmatch(r'[0-9]{1,3}/[0-3]', pair) for pair in pairs):
            raise InspectionError(f'Invalid nine-piece catalogue row at line {line_number}')
        values = [tuple(map(int, pair.split('/'))) for pair in pairs]
        if any(not 1 <= piece <= 256 for piece, _ in values) or len({p for p, _ in values}) != 9:
            raise InspectionError(f'Invalid or duplicated catalogue piece at line {line_number}')
        roots.append(tuple(4 * (piece - 1) + rotation for piece, rotation in values))
    if len(set(roots)) != len(roots):
        raise InspectionError('Duplicate root catalogue rows')
    return tuple(roots)


def parse_allow(data, nroots):
    indices = tuple(_decimal(line, 'allowed root index', 0, nroots - 1) for _, line in _lines(data))
    if not indices or len(indices) > nroots or len(set(indices)) != len(indices):
        raise InspectionError('Root allow list must contain unique in-range indices')
    return indices


def parse_pieces(data, bundle):
    rows = []
    for _, line in _lines(data):
        parts = line.split()
        if len(parts) != 4:
            raise InspectionError('Pieces must have four U,D,L,R colors per row')
        rows.append(tuple(_decimal(part, 'piece color', 0, 22) for part in parts))
    if tuple(rows) != bundle.pieces_udlr:
        raise InspectionError('CPU piece definitions differ from the trusted official puzzle')
    return tuple(rows)


def parse_hints(data, bundle):
    clues = {}
    for _, line in _lines(data):
        parts = line.split()
        if len(parts) != 4:
            raise InspectionError('Hints must be piece_id row column rotation')
        piece, row, column, rotation = (_decimal(parts[0], 'hint piece', 1, 256),
            _decimal(parts[1], 'hint row', 0, 15), _decimal(parts[2], 'hint column', 0, 15),
            _decimal(parts[3], 'hint rotation', 0, 3))
        cell = row * 16 + column
        if cell in clues:
            raise InspectionError('Duplicate hint cell')
        clues[cell] = 4 * (piece - 1) + rotation
    for turns in range(4):
        expected = {}
        for cell, state in bundle.fixed_clues.items():
            row, column = divmod(cell, 16)
            for _ in range(turns):
                row, column = column, 15 - row
            expected[row * 16 + column] = 4 * (state // 4) + ((state % 4 + turns) % 4)
        if clues == expected:
            return tuple(sorted(clues.items())), turns
    raise InspectionError('Hints do not match a rigid rotation of the official five clues')


def validate_prefix_geometry(catalog, clues, bundle):
    for root_index, states in enumerate(catalog):
        placed = dict(clues)
        for cell, state in zip(PREFIX_CELLS, states):
            if cell in placed and placed[cell] != state:
                raise InspectionError(f'Root {root_index} conflicts with a fixed clue')
            placed[cell] = state
        if len({state // 4 for state in placed.values()}) != len(placed):
            raise InspectionError(f'Root {root_index} repeats a clue piece')
        for cell, state in placed.items():
            faces = bundle.oriented_edges[state]
            outside = (cell < 16, cell % 16 == 15, cell >= 240, cell % 16 == 0)
            if any((color == 0) != exterior for color, exterior in zip(faces, outside)):
                raise InspectionError(f'Root {root_index} has an invalid frame')
            for neighbor, side, opposite in ((cell + 1, 1, 3), (cell + 16, 2, 0)):
                if neighbor in placed and (side != 1 or cell % 16 != 15):
                    if faces[side] != bundle.oriented_edges[placed[neighbor]][opposite]:
                        raise InspectionError(f'Root {root_index} has mismatched prefix edges')


@dataclass(frozen=True)
class JournalIdentity:
    run_id: str
    cfg_hash: str
    epoch: str
    build_id: str
    source_sha256: str
    dedup_key: str = DEDUP_KEY


def parse_journal(asset, job, nroots):
    # Agreement of these fields is a consistency check, not authentication of
    # cfg_hash. This manifest omits J0 and the exact root allow-list binding.
    value = _json(asset.data)
    required = {'format_version': 2, 'record_size': 40, 'endianness': 'little',
                'shard_index': job.shard_index, 'shard_count': job.shard_count,
                'nroots': nroots, 'force_jitters': job.jitter_end, 'force_interleave': 1,
                'node_cap': job.node_cap, 'seed': job.seed, 'min_save': job.min_save,
                'epoch': job.epoch, 'dedup_key': DEDUP_KEY}
    for key, expected in required.items():
        if type(value.get(key)) is not type(expected) or value[key] != expected:
            raise InspectionError(f'Journal manifest {key} disagrees with the inspected CPU workunit')
    for key in ('run_id', 'cfg_hash'):
        if not isinstance(value.get(key), str) or not re.fullmatch(r'[0-9a-f]{16}', value[key]):
            raise InspectionError(f'Invalid opaque journal {key}')
    if not asset.logical_name.endswith('seg_' + value['run_id'] + '.manifest.json'):
        raise InspectionError('Journal filename and run_id disagree')
    build = value.get('build_id')
    if not isinstance(build, str) or not 1 <= len(build) <= 128 or not build.isprintable():
        raise InspectionError('Invalid journal build_id')
    return JournalIdentity(value['run_id'], value['cfg_hash'], job.epoch, build, asset.sha256)


@dataclass(frozen=True)
class Ticket:
    globalticket: int
    jidx: int
    cidx: int
    prefix_states: tuple[int, ...]


@dataclass(frozen=True)
class InterleavedSeedReconstruction:
    """Static 236 constructor seed/state; not a CPU execution validation."""
    globalticket: int
    jidx: int
    cidx: int
    schedule_index: int
    constructor_seed_u64: int
    splitmix_words_u64: tuple[int, int, int, int]
    binary_contract_sha256: str = INTERLEAVED_SEED_CONTRACT_SHA256
    evidence: str = 'static disassembly of solverThreadConfigured<true>, interleaved root path'
    conditional_applicability: str = 'If the dispatcher selects solverThreadConfigured<true>; force-interleave and the reference digest alone do not prove that selection'
    dispatcher_path_verified: bool = False
    current_binary_verified: bool = False
    full_dfs_compatibility_verified: bool = False


def _splitmix_finalizer_u64(value):
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & _U64_MASK
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & _U64_MASK
    return value ^ (value >> 31)


def _interleaved_constructor_state(cidx, jidx):
    # The four words passed to sortTableForRestart come from seed+G through
    # seed+4G. The raw seed itself is not the first SplitMix output.
    seed = ((cidx + 1) * _SPLITMIX_INCREMENT + (jidx + 1) * _JITTER_MULTIPLIER) & _U64_MASK
    return seed, tuple(_splitmix_finalizer_u64((seed + step * _SPLITMIX_INCREMENT) & _U64_MASK)
                       for step in range(1, 5))


def _ticket_ranges(job, nroots, allowed):
    divisor = math.gcd(nroots, job.shard_count)
    period = job.shard_count // divisor
    inverse = pow(nroots // divisor, -1, period) if period != 1 else 0
    for cidx in allowed:
        rhs = job.shard_index + job.jitter_start * nroots - cidx
        if rhs % divisor:
            continue
        residue = ((rhs // divisor) * inverse) % period
        first = job.jitter_start + (residue - job.jitter_start) % period
        if first < job.jitter_end:
            yield cidx, first, period, 1 + (job.jitter_end - 1 - first) // period


@dataclass(frozen=True)
class CpuWorkunit:
    job: CpuJob
    input_sha256: str
    assets: tuple[Asset, ...]
    pieces_udlr: tuple[tuple[int, ...], ...]
    catalog: tuple[tuple[int, ...], ...]
    allowed_indices: tuple[int, ...]
    clues: tuple[tuple[int, int], ...]
    frame_rotation_cw: int
    campaign_revision: int
    root_set_id: str
    manifest_hash_checks: tuple[tuple[str, str], ...]
    journals: tuple[JournalIdentity, ...]
    ticket_count: int

    def reconstruct_interleaved_seed(self, ticket, *, binary_contract_sha256):
        """Opt in to the pinned236 static seed model for a member ticket.

        Supplying the reference digest selects a reconstruction contract; it
        does not authenticate an executable installed on the calling machine.
        The dispatcher selecting solverThreadConfigured<true> has not been
        established here. This is a conditional state reconstruction for that
        path, not a statement that a particular running job uses this state.
        The opaque journal cfg_hash is retained separately and never replaced.
        The command-line --seed is not an input to this verified branch.
        """
        if binary_contract_sha256 != INTERLEAVED_SEED_CONTRACT_SHA256:
            raise InspectionError('Seed reconstruction requires the pinned236 binary contract')
        if dict(self.job.options).get('--force-interleave') is not True:
            raise InspectionError('Seed reconstruction applies only to force-interleave workunits')
        if not isinstance(ticket, Ticket):
            raise InspectionError('Seed reconstruction requires an inspected Ticket')
        _int(ticket.globalticket, 'globalticket', 0, _U64_MASK)
        _int(ticket.jidx, 'jidx', self.job.jitter_start, self.job.jitter_end - 1)
        _int(ticket.cidx, 'cidx', 0, len(self.catalog) - 1)
        if (ticket.cidx not in self.allowed_indices or ticket.prefix_states != self.catalog[ticket.cidx]
                or ticket.globalticket != ticket.jidx * len(self.catalog) + ticket.cidx):
            raise InspectionError('Ticket identity/prefix is not a member of this workunit')
        offset = ticket.globalticket - self.job.jitter_start * len(self.catalog) - self.job.shard_index
        if offset < 0 or offset % self.job.shard_count:
            raise InspectionError('Ticket does not belong to the assigned interleaved shard')
        seed, words = _interleaved_constructor_state(ticket.cidx, ticket.jidx)
        return InterleavedSeedReconstruction(ticket.globalticket, ticket.jidx, ticket.cidx,
                                            offset // self.job.shard_count, seed, words)

    def iter_tickets(self) -> Iterator[Ticket]:
        heap = [(first * len(self.catalog) + cidx, first, cidx, period)
                for cidx, first, period, _ in _ticket_ranges(self.job, len(self.catalog), self.allowed_indices)]
        heapq.heapify(heap)
        while heap:
            globalticket, jidx, cidx, period = heapq.heappop(heap)
            yield Ticket(globalticket, jidx, cidx, self.catalog[cidx])
            jidx += period
            if jidx < self.job.jitter_end:
                heapq.heappush(heap, (jidx * len(self.catalog) + cidx, jidx, cidx, period))

    def summary(self, ticket_limit=10):
        _int(ticket_limit, 'ticket_limit', 0, 1000)
        import itertools
        return {'inspection_schema': 'eternity-cpu-workunit-inspection/v1', 'read_only': True,
                'input_sha256': self.input_sha256, 'cpu_cfg_hash_recomputed': False,
                'production_dfs_compatibility_verified': False,
                'cpu_option_semantics_fully_verified': False,
                'prefix_geometry_verified': True, 'prefix_cells_top_down_row_major': list(PREFIX_CELLS),
                'prefix_mapping_reference': {'binary_sha256': PREFIX_REFERENCE_SHA256,
                    'symbol': '_ZL17BOARD_ORDER_CLUED', 'file_offset_hex': '0x1ca740',
                    'interpretation': 'inverse of cell-to-depth permutation, first9 cells',
                    'current_binary_verified': False},
                'frame_rotation_clockwise_degrees': self.frame_rotation_cw * 90,
                'epoch': self.job.epoch, 'shard_index': self.job.shard_index, 'shard_count': self.job.shard_count,
                'jitter_start': self.job.jitter_start, 'jitter_end_exclusive': self.job.jitter_end,
                'catalog_roots': len(self.catalog), 'allowed_roots': len(self.allowed_indices),
                'derived_ticket_count': self.ticket_count, 'ticket_formula': FORMULA,
                'campaign_revision': self.campaign_revision, 'root_set_id': self.root_set_id,
                'campaign_expected_hash_checks': dict(self.manifest_hash_checks),
                'assets': [{'logical_name': a.logical_name, 'source_name': a.source_name,
                            'sha256': a.sha256, 'lf_sha256': a.lf_sha256, 'bytes': a.size} for a in self.assets],
                'journal_identities': [vars(j) for j in self.journals],
                'journal_identity_authenticated': False,
                'journal_binding_limit': 'Matching fields do not authenticate cfg_hash or fully bind jitter_start and the root allow list',
                'tickets_sample': [vars(t) for t in itertools.islice(self.iter_tickets(), ticket_limit)],
                'tickets_sample_is_complete': self.ticket_count <= ticket_limit}


def inspect_workunit(slot_dir, project_root=None, include_journals=True, bundle=None):
    """Inspect selected immutable in-memory snapshots, never execute a workunit.

    For live BOINC logical soft links, project_root must be explicitly supplied.
    The six input files are re-read after capture to detect cross-file changes.
    The snapshot hash describes exact input bytes, never the CPU cfg_hash.
    """
    slot = Path(slot_dir).resolve(strict=True)
    project = None if project_root is None else Path(project_root).resolve(strict=True)
    if not slot.is_dir() or (project is not None and not project.is_dir()):
        raise InspectionError('Slot and project roots must be directories')
    bundle = load_bundle() if bundle is None else bundle
    assets = tuple(_asset(slot, name, project, MAX_CATALOG_BYTES if name == 'campaign_catalog.txt' else MAX_SMALL)
                   for name in LOGICAL_FILES)
    for original in assets:
        again = _asset(slot, original.logical_name, project, MAX_CATALOG_BYTES if original.logical_name == 'campaign_catalog.txt' else MAX_SMALL)
        if original.sha256 != again.sha256 or original.source_name != again.source_name:
            raise InspectionError('CPU inputs changed during snapshot capture; retry inspection')
    by_name = {asset.logical_name: asset for asset in assets}
    job = parse_job(by_name['job.xml'].data)
    pieces = parse_pieces(by_name['256pieces.txt'].data, bundle)
    catalog = parse_catalog(by_name['campaign_catalog.txt'].data)
    allowed = parse_allow(by_name['exploration_allow.txt'].data, len(catalog))
    clues, rotation = parse_hints(by_name['campaign_hints.txt'].data, bundle)
    validate_prefix_geometry(catalog, clues, bundle)
    campaign = _json(by_name['campaign_manifest.json'].data)
    revision = _int(campaign.get('campaign_revision'), 'campaign revision', 0, 1000000000)
    root_set_id = campaign.get('root_set_id')
    if not isinstance(root_set_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', root_set_id):
        raise InspectionError('Invalid campaign root_set_id')
    if _int(campaign.get('frame_k'), 'frame_k', 0, 3) != rotation:
        raise InspectionError('Campaign frame_k and actual hint rotation disagree')
    expected = campaign.get('expected')
    mapping = {'hints_sha256': 'campaign_hints.txt', 'root_allow_sha256': 'exploration_allow.txt',
               'root_catalog_sha256': 'campaign_catalog.txt'}
    if not isinstance(expected, dict) or set(expected) != set(mapping):
        raise InspectionError('Unsupported campaign expected-hash manifest')
    checks = []
    for key, logical in mapping.items():
        digest = expected[key]
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise InspectionError('Invalid campaign expected SHA-256')
        actual = by_name[logical]
        checks.append((logical, 'raw-match' if digest == actual.sha256 else
                       'lf-match' if digest == actual.lf_sha256 else 'mismatch-runtime-input'))
    count = sum(amount for _, _, _, amount in _ticket_ranges(job, len(catalog), allowed))
    if not 1 <= count <= MAX_TICKETS or (job.jitter_end - 1) * len(catalog) + len(catalog) - 1 > 2**64 - 1:
        raise InspectionError('Derived ticket set is empty or exceeds supported bounds')
    journals = []
    if include_journals:
        directory = slot / 'ticketlog'
        if directory.exists():
            if not _inside(directory.resolve(strict=True), slot):
                raise InspectionError('Journal directory resolves outside the slot')
            names = sorted(path.name for path in directory.glob('seg_*.manifest.json'))
            if len(names) > 64:
                raise InspectionError('Too many journal manifests')
            for name in names:
                asset = _asset(slot, 'ticketlog/' + name, project, MAX_SMALL)
                journals.append(parse_journal(asset, job, len(catalog)))
    binding = json.dumps({asset.logical_name: asset.sha256 for asset in assets}, sort_keys=True,
                         separators=(',', ':')).encode('ascii')
    return CpuWorkunit(job, hashlib.sha256(binding).hexdigest(), assets, pieces, catalog, allowed,
                       clues, rotation, revision, root_set_id, tuple(checks), tuple(journals), count)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slot', type=Path, required=True, help='Live slot or directory of selected copied inputs')
    parser.add_argument('--project-root', type=Path, help='Explicit BOINC project root for logical soft links')
    parser.add_argument('--tickets', type=int, default=10, help='Number of ticket identities to print, at most1000')
    parser.add_argument('--no-journals', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = inspect_workunit(args.slot, args.project_root, not args.no_journals).summary(args.tickets)
    except (InspectionError, OSError, ValueError) as exc:
        print(json.dumps({'valid': False, 'read_only': True, 'error': str(exc)}))
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
