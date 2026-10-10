"""Offline, bounded imports of public board archives into the shared SQLite cache.

Archive membership is not an authoritative index. Existing index membership and
freshness metadata are preserved; new public IDs are added as inactive records.
Public IDs are opaque. Local placement hashes use uint16 little-endian states.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import uuid
import zlib

from library_cache import SHA_RE, board_bytes
from validator import load_bundle, read_pieces, validate_board

MAX_COMPRESSED = 1024 ** 3
MAX_EXPANDED = 2 * 1024 ** 3
MAX_LINE = 65536
MAX_ROWS = 1_000_000


class ArchiveImportError(ValueError):
    """An archive or existing cache failed import validation."""


class ArchiveCancelled(ArchiveImportError):
    """Cancelled before the cache transaction committed."""


def _check_cancel(callback):
    if callback is not None and callback():
        raise ArchiveCancelled('Archive import cancelled; cache was not changed')


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ArchiveImportError('Duplicate JSON key')
        result[key] = value
    return result


def _constant(value):
    raise ArchiveImportError('Non-finite JSON number: ' + value)


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        _constant(value)
    return result


def _json(line):
    try:
        return json.loads(line.decode('utf-8'), object_pairs_hook=_pairs,
                          parse_constant=_constant, parse_float=_float)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ArchiveImportError('Invalid archive JSON: ' + str(exc)) from exc


def _copy_archive(source, destination, cancel):
    if not source.is_file() or source.stat().st_size > MAX_COMPRESSED:
        raise ArchiveImportError('Archive must be a regular file of at most 1 GiB')
    digest = hashlib.sha256()
    size = 0
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        while True:
            _check_cancel(cancel)
            block = incoming.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > MAX_COMPRESSED:
                raise ArchiveImportError('Compressed archive exceeds 1 GiB')
            outgoing.write(block)
            digest.update(block)
    return digest.hexdigest(), size


def _prepare(snapshot, stage_path, bundle, cancel, progress):
    counts = Counter()
    rows = expanded = 0
    eof = object()
    with closing(sqlite3.connect(stage_path)) as stage, gzip.open(snapshot, 'rb') as stream:
        stage.execute('CREATE TABLE incoming(public_sha TEXT PRIMARY KEY, score INTEGER NOT NULL, local_sha TEXT NOT NULL, placement BLOB NOT NULL)')

        def line():
            nonlocal expanded
            _check_cancel(cancel)
            raw = stream.readline(MAX_LINE + 1)
            expanded += len(raw)
            if len(raw) > MAX_LINE or expanded > MAX_EXPANDED:
                raise ArchiveImportError('Archive exceeds line or expanded size limit')
            if not raw:
                return eof
            if not raw.strip():
                raise ArchiveImportError('Blank archive line')
            return _json(raw)

        header = line()
        if not isinstance(header, dict) or type(header.get('schema')) is not int or header['schema'] != 1:
            raise ArchiveImportError('Unsupported archive header')
        total = header.get('count')
        if type(total) is not int or not 1 <= total <= MAX_ROWS:
            raise ArchiveImportError('Invalid archive count')
        generated = header.get('generated_at')
        try:
            if not isinstance(generated, str) or datetime.fromisoformat(generated.replace('Z', '+00:00')).tzinfo is None:
                raise ValueError('timestamp requires an explicit timezone')
        except ValueError as exc:
            raise ArchiveImportError('Invalid archive timestamp') from exc
        if progress:
            progress({'phase': 'validating', 'validated_boards': 0, 'total_boards': total})
        while (document := line()) is not eof:
            rows += 1
            if rows > total or not isinstance(document, dict):
                raise ArchiveImportError('Invalid record or archive count')
            sha, score, breaks = (document.get(k) for k in ('sha', 'score', 'breaks'))
            if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
                raise ArchiveImportError('Invalid public board identifier')
            if type(score) is not int or not 0 <= score <= 480 or type(breaks) is not int or breaks != 480 - score:
                raise ArchiveImportError('Invalid score or break count')
            board = document.get('board')
            if not isinstance(board, list):
                raise ArchiveImportError('Expected a board array')
            check = validate_board(board, bundle)
            if not check['valid'] or check['score'] != score or check['break_count'] != breaks:
                raise ArchiveImportError(f'Board rules or score fail on line {rows + 1}')
            blob = board_bytes(board)
            local_sha = hashlib.sha256(blob).hexdigest()
            try:
                stage.execute('INSERT INTO incoming VALUES(?,?,?,?)', (sha, score, local_sha, blob))
            except sqlite3.IntegrityError as exc:
                raise ArchiveImportError('Duplicate public board identifier') from exc
            counts[score] += 1
            if rows % 1000 == 0:
                stage.commit()
                if progress:
                    progress({'phase': 'validating', 'validated_boards': rows, 'total_boards': total})
        if rows != total:
            raise ArchiveImportError('Archive count does not match its header')
        stage.commit()
        unique = stage.execute('SELECT COUNT(DISTINCT local_sha) FROM incoming').fetchone()[0]
    if progress:
        progress({'phase': 'validating', 'validated_boards': rows, 'total_boards': total})
    return {'validated_boards': rows, 'total_boards': total, 'unique_placements': unique,
            'duplicate_placements': rows - unique, 'generated_at': generated,
            'expanded_bytes': expanded, 'scores': {str(k): v for k, v in sorted(counts.items(), reverse=True)},
            'gzip_complete': True, 'all_piece_frame_clue_score_checks_passed': True}


def _schema(db):
    # execute, not executescript: the latter would commit our transaction first.
    for sql in (
        'CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)',
        'CREATE TABLE IF NOT EXISTS boards(public_sha TEXT PRIMARY KEY,score INTEGER NOT NULL,has_content INTEGER NOT NULL,active INTEGER NOT NULL DEFAULT 0,local_sha TEXT,placement BLOB,retry_after REAL NOT NULL DEFAULT 0,failures INTEGER NOT NULL DEFAULT 0)',
        'CREATE INDEX IF NOT EXISTS board_lookup ON boards(local_sha)',
        'CREATE INDEX IF NOT EXISTS board_pending ON boards(active,has_content,local_sha,score)',
    ):
        db.execute(sql)


def _merge(cache, stage, private_db, backup, report, bundle, cancel, progress):
    existing = cache.exists()
    target = cache if existing else private_db
    cancelled = False

    def interrupt():
        nonlocal cancelled
        cancelled = bool(cancel is not None and cancel())
        return int(cancelled)

    with closing(sqlite3.connect(target.as_uri() + ('?mode=rw' if existing else '?mode=rwc'),
                                uri=True, timeout=30, isolation_level=None)) as db:
        db.execute('ATTACH DATABASE ? AS archive_stage', (stage.as_uri() + '?mode=ro',))
        try:
            db.execute('BEGIN IMMEDIATE')
            _check_cancel(cancel)
            if existing:
                # Another read connection sees the exact pre-merge state while
                # BEGIN IMMEDIATE excludes concurrent writers. Never back up a
                # connection with its own open write transaction (it can hang).
                try:
                    with closing(sqlite3.connect(cache.as_uri() + '?mode=ro', uri=True)) as source:
                        with closing(sqlite3.connect(backup)) as destination:
                            source.backup(destination, pages=256,
                                          progress=lambda *_: _check_cancel(cancel))
                except BaseException as exc:
                    # An interrupted backup is not reusable evidence. This is
                    # the unique destination created for this import only.
                    try:
                        backup.unlink(missing_ok=True)
                    except OSError as cleanup_error:
                        exc.add_note('Could not remove incomplete backup: ' + str(cleanup_error))
                    raise
            _check_cancel(cancel)
            db.set_progress_handler(interrupt, 1000)
            _schema(db)
            conflict = db.execute('''SELECT old.public_sha FROM boards AS old
                JOIN archive_stage.incoming AS new ON old.public_sha=new.public_sha
                WHERE old.score!=new.score OR (old.local_sha IS NOT NULL AND
                (old.local_sha!=new.local_sha OR old.placement IS NULL OR old.placement!=new.placement))
                OR (old.placement IS NOT NULL AND old.placement!=new.placement) LIMIT 1''').fetchone()
            if conflict:
                raise ArchiveImportError('Archive conflicts with cached public identity: ' + conflict[0])
            added = db.execute('''SELECT COUNT(*) FROM archive_stage.incoming AS new LEFT JOIN boards AS old
                ON old.public_sha=new.public_sha WHERE old.public_sha IS NULL''').fetchone()[0]
            filled = db.execute('''SELECT COUNT(*) FROM archive_stage.incoming AS new JOIN boards AS old
                ON old.public_sha=new.public_sha WHERE old.local_sha IS NULL''').fetchone()[0]
            db.execute('''INSERT INTO boards(public_sha,score,has_content,active,local_sha,placement)
                SELECT public_sha,score,1,0,local_sha,placement FROM archive_stage.incoming WHERE 1
                ON CONFLICT(public_sha) DO UPDATE SET local_sha=excluded.local_sha,
                placement=excluded.placement,retry_after=0,failures=0''')
            base = [[u, r, d, l] for u, d, l, r in bundle.pieces_udlr]
            db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)',
                       ('canonical_faces', json.dumps(base, separators=(',', ':'))))
            provenance = {k: report[k] for k in ('archive_sha256', 'generated_at', 'validated_boards', 'archive_path')}
            provenance['imported_at'] = datetime.now(timezone.utc).isoformat()
            db.execute('INSERT INTO metadata VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                       ('last_archive_import', json.dumps(provenance, separators=(',', ':'))))
            total, cached, indexed, active_cached, known = db.execute('''SELECT COUNT(*),
                COUNT(local_sha),COALESCE(SUM(active),0),COALESCE(SUM(active=1 AND local_sha IS NOT NULL),0),
                COUNT(DISTINCT local_sha) FROM boards''').fetchone()
            if progress:
                progress({'phase': 'merging', 'validated_boards': report['validated_boards'],
                          'total_boards': report['total_boards'], 'ready_to_commit': True})
            _check_cancel(cancel)
            db.set_progress_handler(None, 0)
            db.commit()
        except BaseException as exc:
            db.set_progress_handler(None, 0)
            if db.in_transaction:
                db.rollback()
            if cancelled and isinstance(exc, sqlite3.OperationalError):
                raise ArchiveCancelled('Archive import cancelled; cache was not changed') from exc
            raise
    if not existing:
        _check_cancel(cancel)
        try:
            # Publish an entire closed SQLite file without overwriting a cache
            # concurrently created by another app process. Same filesystem.
            os.link(private_db, cache)
        except FileExistsError:
            return _merge(cache, stage, private_db, backup, report, bundle, cancel, progress)
    return {'new_boards': added + filled, 'new_public_ids': added, 'filled_existing_boards': filled,
            'already_cached': report['validated_boards'] - added - filled,
            'known_exact_boards': known, 'cached_geometries': cached, 'total_public_ids': total,
            'indexed_boards': indexed, 'indexed_cached_geometries': active_cached,
            'backup_path': str(backup) if existing else None}


def import_archive(archive_path, data_dir, *, pieces_path=None, should_cancel=None, on_progress=None,
                   expected_sha256=None):
    """Validate then merge a local gzip JSONL archive; never make network requests.

    data_dir contains library.sqlite3. Callbacks run synchronously; cancellation
    before commit raises ArchiveCancelled. After commit, cancellation is ignored.
    All returned counts are integers and paths are strings (backup may be null).
    new_boards counts newly cached public records, including missing geometries
    filled for existing IDs. Indexed membership and freshness stay unchanged.
    """
    _check_cancel(should_cancel)
    if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not SHA_RE.fullmatch(expected_sha256)):
        raise ArchiveImportError('Expected archive SHA-256 must be 64 lowercase hexadecimal characters')
    bundle = load_bundle()
    if pieces_path is not None and read_pieces(pieces_path) != bundle.pieces_udlr:
        raise ArchiveImportError('Piece definitions differ from the official bundled puzzle')
    directory = Path(data_dir).expanduser().resolve()
    evidence = directory / 'archive-imports'
    evidence.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    snapshot = evidence / (token + '.jsonl.gz.tmp')
    stage = evidence / (token + '.stage.sqlite3')
    private_db = evidence / (token + '.new.sqlite3')
    backup = evidence / ('library-before-' + token + '.sqlite3')
    report = None
    report_path = None
    try:
        digest, size = _copy_archive(Path(archive_path).expanduser().resolve(), snapshot, should_cancel)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ArchiveImportError('Archive SHA-256 differs from the expected download hash')
        report = _prepare(snapshot, stage, bundle, should_cancel, on_progress)
        _check_cancel(should_cancel)
        archive_copy = evidence / ('archive-' + digest + '.jsonl.gz')
        # The validated private snapshot is authoritative. Replacing an existing
        # copy at its content-derived name is safe and keeps provenance reusable.
        os.replace(snapshot, archive_copy)
        report.update(archive_sha256=digest, compressed_bytes=size, archive_path=str(archive_copy),
                      archive_became_active_snapshot=False, network_requests=0)
        if on_progress:
            on_progress({'phase': 'merging', 'validated_boards': report['validated_boards'],
                         'total_boards': report['total_boards'], 'ready_to_commit': False})
        _check_cancel(should_cancel)
        report.update(_merge(directory / 'library.sqlite3', stage, private_db, backup,
                             report, bundle, should_cancel, on_progress))
        report.update(committed=True, completed_at=datetime.now(timezone.utc).isoformat())
        report_path = evidence / ('import-' + token + '.json')
        report['report_path'] = str(report_path)
        return report
    except (gzip.BadGzipFile, EOFError, zlib.error) as exc:
        raise ArchiveImportError('Truncated or corrupt gzip archive') from exc
    finally:
        # Only our unique, explicit temporary paths; never remove other cache
        # files or recurse through a directory supplied by a caller.
        cleanup_warnings = []
        for path in (snapshot, stage, private_db):
            for suffix in ('', '-journal', '-wal', '-shm'):
                temporary = Path(str(path) + suffix)
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    cleanup_warnings.append(f'Could not remove {temporary.name}: {exc}')
        if report is not None and report.get('committed'):
            # The returned dictionary is updated before the pending return.
            # Cleanup and evidence-write failures cannot undo a committed merge.
            if cleanup_warnings:
                report['cleanup_warnings'] = cleanup_warnings
            try:
                report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            except OSError as exc:
                report['report_write_error'] = str(exc)
        elif cleanup_warnings:
            original_error = sys.exc_info()[1]
            if original_error is not None:
                original_error.add_note('; '.join(cleanup_warnings))
