"""Bounded, read-only and seeded parent selection from the local board cache.

Validated elite parents are retained before an optional metadata-only pass
samples logarithmic score-deficit bands. Only a bounded
number of placement blobs are retrieved by rowid; no random SQL sort, cache
writes or network requests occur. Metadata scores guide sampling only: accepted
parents are structurally validated and ranked with independently counted edges.
The heuristic preserves elites and greedily spreads placement/rotation and
mismatch-edge patterns. Diversity is a search heuristic, not a speedup guarantee.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import replace
import heapq
import math
from pathlib import Path
import random
import sqlite3
import struct
import time

from exact_model import check_solution, grid_edges


def _features(problem, board, edges):
    n = len(board)
    mismatch = sum(1 << i for i, (a, b, sa, sb) in enumerate(edges)
                   if problem.faces[board[a]][sa] != problem.faces[board[b]][sb])
    return (frozenset(c * n + state // 4 for c, state in enumerate(board)),
            frozenset(c * 4 * n + state for c, state in enumerate(board)),
            mismatch)


def _distance(left, right, cells):
    """Integer 0..1000 distance; score quality is handled separately by elites."""
    pieces = cells - len(left[0] & right[0])
    states = cells - len(left[1] & right[1])
    union = (left[2] | right[2]).bit_count()
    mismatch = 0 if not union else 400 * (left[2] ^ right[2]).bit_count() // union
    return 400 * pieces // cells + 200 * states // cells + mismatch


def _diverse(candidates, limit, cells, stopped):
    # Tuples contain verified score, seeded tie-break, board, feature tuple.
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    elite_count = min(len(candidates), max(1, (limit + 3) // 4))
    selected = candidates[:elite_count]
    remaining = candidates[elite_count:]
    nearest = [1001] * len(remaining)
    # Update once per chosen parent: O(pool * limit), not O(pool * limit**2).
    previous = list(selected)
    while remaining and len(selected) < limit and not stopped():
        for chosen in previous:
            for i, candidate in enumerate(remaining):
                if i % 32 == 0 and stopped():
                    return [item[2] for item in (selected + remaining)[:limit]]
                nearest[i] = min(nearest[i], _distance(chosen[3], candidate[3], cells))
        index = max(range(len(remaining)),
                    key=lambda i: (nearest[i], remaining[i][0], remaining[i][1]))
        chosen = remaining.pop(index)
        nearest.pop(index)
        selected.append(chosen)
        previous = [chosen]
    # An expired selection budget still returns validated candidates. Remaining
    # items retain verified-score order if diversity selection was interrupted.
    return [item[2] for item in (selected + remaining)[:limit]]


def cached_parents(problem, database, *, limit=32, seed=0,
                   candidate_limit=256, max_seconds=5.0, should_stop=None):
    """Return up to limit valid, distinct local boards, without modifying cache.

    At most candidate_limit blobs are decoded/validated (16..1024, >= limit).
    Memory is bounded by that pool plus <=10 metadata reservoirs for a 16x16
    board. Each reservoir covers a power-of-two band of deficits from maximum
    score. A metadata scan is linear in cache size and receives at most half
    the remaining cooperative wall budget after a small validated elite preload.
    Partial scans still leave time for candidate validation and selection.
    The best quarter of returned parents are elites within the validated sample;
    the rest maximize minimum diversity from already selected parents.

    A deadline or cancellation returns any already selected/validated elites,
    possibly fewer than limit. Corrupt SQLite/schema errors still propagate.
    Missing files return [] and are never created. Scores stored in the cache
    are sampling hints; they never establish board validity or final rank.
    """
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError('limit must be 1..256')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')
    if type(candidate_limit) is not int or not max(16, limit) <= candidate_limit <= 1024:
        raise ValueError('candidate_limit must be 16..1024 and at least limit')
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError('max_seconds must be finite and nonnegative')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')

    deadline = time.monotonic() + max_seconds
    halted = False
    callback_errors = []
    scan_deadline = None
    scan_interrupted = False

    def stopped():
        nonlocal halted
        if not halted:
            halted = time.monotonic() >= deadline or bool(should_stop and should_stop())
        return halted

    def sql_progress():
        nonlocal scan_interrupted
        try:
            if stopped():
                return 1
            if scan_deadline is not None and time.monotonic() >= scan_deadline:
                scan_interrupted = True
                return 1
            return 0
        except BaseException as exc:
            callback_errors.append(exc)
            return 1

    database = Path(database)
    if stopped() or not database.is_file():
        return []
    edges = grid_edges(problem.size)
    structure = replace(problem, edges=())
    cells = problem.size ** 2
    rng = random.Random(seed)
    bands = {}
    elite = []
    elite_capacity = min(candidate_limit, max(limit, candidate_limit // 4))
    candidates = []
    seen = set()
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro',
                                 uri=True, timeout=min(1.0, max_seconds))) as db:
        db.set_progress_handler(sql_progress, 1000)
        # One read snapshot keeps rowids/placements consistent during imports.
        db.execute('BEGIN')
        loaded_rows = {}
        def load_parent(rowid, rank):
            # Preload and sampled rows share one strict placement-read cap.
            if rowid in loaded_rows or len(loaded_rows) >= candidate_limit or stopped():
                return
            loaded_rows[rowid] = rank
            row = db.execute('SELECT placement FROM boards WHERE rowid=?', (rowid,)).fetchone()
            if not row:
                return
            blob = row[0]
            if not isinstance(blob, bytes) or len(blob) != 2 * cells or blob in seen:
                return
            seen.add(blob)
            board = list(struct.unpack('<' + str(cells) + 'H', blob))
            if not check_solution(structure, board):
                return
            features = _features(problem, board, edges)
            actual_score = len(edges) - features[2].bit_count()
            candidates.append((actual_score, rank, board, features))
        try:
            # Keep a few independently validated fallbacks before the optional
            # full-cache scan. Metadata scores only select candidates here.
            preload = min(8, max(1, (limit + 3) // 4))
            for rowid, _ in db.execute(
                    'SELECT rowid,score FROM boards WHERE placement IS NOT NULL '
                    'ORDER BY score DESC,rowid LIMIT ?', (preload,)):
                if stopped():
                    break
                load_parent(rowid, rng.getrandbits(64))
            now = time.monotonic()
            scan_deadline = now + max(0., deadline - now) * .5
            try:
                for row_number, (rowid, metadata_score) in enumerate(db.execute(
                        'SELECT rowid,score FROM boards WHERE placement IS NOT NULL ORDER BY rowid')):
                    if row_number % 256 == 0:
                        if stopped() or time.monotonic() >= scan_deadline:
                            break
                    if type(metadata_score) is not int or not 0 <= metadata_score <= len(edges):
                        continue
                    rank = rng.getrandbits(64)
                    band = (len(edges) - metadata_score).bit_length()
                    reservoir = bands.setdefault(band, [])
                    sampled = (rank, rowid)
                    if len(reservoir) < candidate_limit:
                        heapq.heappush(reservoir, sampled)
                    elif sampled > reservoir[0]:
                        heapq.heapreplace(reservoir, sampled)
                    top = (metadata_score, rank, rowid)
                    if len(elite) < elite_capacity:
                        heapq.heappush(elite, top)
                    elif top > elite[0]:
                        heapq.heapreplace(elite, top)
            except sqlite3.OperationalError:
                if callback_errors:
                    raise callback_errors[0]
                if not (halted or scan_interrupted):
                    raise
            finally:
                # A scan-only timeout must not poison later bounded point reads.
                scan_deadline = None
            if callback_errors:
                raise callback_errors[0]
            selected_rows = dict(loaded_rows)
            for _, rank, rowid in sorted(elite, reverse=True):
                if len(selected_rows) >= candidate_limit:
                    break
                selected_rows.setdefault(rowid, rank)
            # Round-robin bands prevent the largest score tier from consuming
            # every sample. Within each band the seeded reservoir is uniform.
            pools = [iter(sorted(bands[band], reverse=True)) for band in sorted(bands)]
            while pools and len(selected_rows) < candidate_limit and not stopped():
                active = []
                for pool in pools:
                    sampled = next(pool, None)
                    if sampled is not None:
                        rank, rowid = sampled
                        selected_rows.setdefault(rowid, rank)
                        active.append(pool)
                    if len(selected_rows) >= candidate_limit:
                        break
                pools = active
            for rowid, rank in selected_rows.items():
                if stopped():
                    break
                load_parent(rowid, rank)
        except sqlite3.OperationalError:
            if callback_errors:
                raise callback_errors[0]
            if not halted:
                raise
        finally:
            db.set_progress_handler(None, 0)
    return _diverse(candidates, limit, cells, stopped)
