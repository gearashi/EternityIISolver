"""Small offline archive fixtures exercise validation and transactional cache use."""
from contextlib import closing
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import archive_import as imp
from app_paths import resource_root
from library_cache import LibraryMonitor


class ArchiveImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = resource_root() / 'data'
        cls.record = json.loads((cls.data / 'record466.json').read_text(encoding='utf-8'))
        cls.seed_path = next(p for p in (cls.data / 'seeds').glob('*.json') if len(p.stem) == 64)
        cls.seed = json.loads(cls.seed_path.read_text(encoding='utf-8'))
        cls.sha = '2c037e70f7e93518a48733c7aacd096226b7f23728efabd6289088b91c0945cd'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='eternity-archive-')
        self.home = Path(self.temp.name)
        self.cache = self.home / 'library'
        self.row = {'sha': self.sha, 'score': 466, 'breaks': 14, 'board': self.record['board'],
                    'discoverer': 'Jef', 'date': '2026-09-19T11:27:35.041Z'}
        self.other = {'sha': self.seed_path.stem, 'score': 465, 'breaks': 15, 'board': self.seed['board']}

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, rows=None, **changes):
        rows = [self.row] if rows is None else rows
        header = {'schema': 1, 'count': len(rows), 'generated_at': '2026-10-09T23:45:22Z'}
        header.update(changes)
        result = self.home / 'input.gz'
        with gzip.open(result, 'wt', encoding='utf-8') as stream:
            for document in [header, *rows]:
                stream.write(json.dumps(document) + '\n')
        return result

    def initial(self, content=False):
        monitor = LibraryMonitor(self.cache, pieces_path=self.data / 'pieces.txt')
        monitor.ingest_index({'schema': 'eternity2-board-library-index/v2', 'count': 1,
                              'generated_at': '2026-10-10T00:00:00Z',
                              'boards': [{'sha': self.sha, 'score': 466, 'has_content': False}]})
        with monitor._db:
            monitor._put_meta('index_checked_at', '123.0')
        if content:
            monitor.register_known_document(self.record, self.sha)
        return monitor

    def snapshot(self):
        with closing(sqlite3.connect(self.cache / 'library.sqlite3')) as db:
            return (db.execute('SELECT * FROM boards ORDER BY public_sha').fetchall(),
                    db.execute('SELECT * FROM metadata ORDER BY key').fetchall())

    def assert_clean_staging(self):
        imports = self.cache / 'archive-imports'
        self.assertFalse(list(imports.glob('*.tmp')))
        self.assertFalse(list(imports.glob('*.stage.sqlite3*')))
        self.assertFalse(list(imports.glob('*.new.sqlite3*')))

    def test_offline_import_preserves_index_and_updates_existing_reader(self):
        monitor = self.initial()
        try:
            before = self.snapshot()
            source = self.archive([self.row, self.other])
            events = []
            with patch('socket.socket.connect', side_effect=AssertionError('network')), \
                 patch('socket.getaddrinfo', side_effect=AssertionError('DNS')), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('HTTP')):
                report = imp.import_archive(source, self.cache, on_progress=events.append)
            self.assertEqual(report['new_boards'], 2)
            self.assertEqual(report['new_public_ids'], 1)
            self.assertEqual(report['filled_existing_boards'], 1)
            self.assertEqual(report['already_cached'], 0)
            self.assertEqual(report['known_exact_boards'], 2)
            self.assertEqual(report['indexed_boards'], 1)
            self.assertEqual(report['network_requests'], 0)
            self.assertTrue(monitor.is_known(self.record['board']))
            self.assertTrue(monitor.is_known(self.seed['board']))
            self.assertEqual(monitor._db.execute('SELECT active,has_content FROM boards WHERE public_sha=?',
                                               (self.sha,)).fetchone(), (1, 0))
            self.assertEqual(monitor._db.execute('SELECT active FROM boards WHERE public_sha=?',
                                               (self.other['sha'],)).fetchone(), (0,))
            after_meta = dict(self.snapshot()[1])
            self.assertEqual({key: after_meta[key] for key, _ in before[1]}, dict(before[1]))
            with closing(sqlite3.connect(report['backup_path'])) as backup:
                self.assertEqual(backup.execute('SELECT * FROM boards ORDER BY public_sha').fetchall(), before[0])
                self.assertEqual(backup.execute('SELECT * FROM metadata ORDER BY key').fetchall(), before[1])
            self.assertEqual(hashlib.sha256(Path(report['archive_path']).read_bytes()).hexdigest(), report['archive_sha256'])
            self.assertEqual(Path(report['archive_path']).read_bytes(), source.read_bytes())
            self.assertTrue(any(e.get('ready_to_commit') for e in events))
            repeated = imp.import_archive(source, self.cache)
            self.assertEqual(repeated['new_boards'], 0)
            self.assertEqual(repeated['already_cached'], 2)
            json.dumps(report, allow_nan=False)
        finally:
            monitor.close()
        self.assert_clean_staging()

    def test_fresh_cache_allows_duplicate_placements_under_distinct_public_ids(self):
        report = imp.import_archive(self.archive([self.row, dict(self.row, sha='a' * 64)]), self.cache)
        self.assertIsNone(report['backup_path'])
        self.assertEqual(report['validated_boards'], 2)
        self.assertEqual(report['unique_placements'], 1)
        self.assertEqual(report['known_exact_boards'], 1)
        self.assertEqual(report['indexed_boards'], 0)
        monitor = LibraryMonitor(self.cache)
        try:
            self.assertTrue(monitor.is_known(self.record['board']))
            self.assertFalse(monitor.status()['full_metadata_loaded'])
        finally:
            monitor.close()
        self.assert_clean_staging()

    def test_invalid_records_never_create_live_cache(self):
        clue = self.row['board'][:]
        clue[34] = 828
        frame = self.row['board'][:]
        frame[0] ^= 1
        cases = [([dict(self.row, score=480, breaks=0)], {}),
                 ([dict(self.row, board=clue)], {}), ([dict(self.row, board=frame)], {}),
                 ([self.row, self.row], {}), ([self.row], {'count': 2}),
                 ([self.row, None], {'count': 1})]
        for rows, header in cases:
            with self.subTest(rows=len(rows), header=header), self.assertRaises(imp.ArchiveImportError):
                imp.import_archive(self.archive(rows, **header), self.cache)
            self.assertFalse((self.cache / 'library.sqlite3').exists())
            self.assert_clean_staging()

    def test_truncation_crc_duplicate_keys_nonfinite_and_bounds(self):
        source = self.archive()
        original = source.read_bytes()
        crc = bytearray(original)
        crc[-8] ^= 1
        for malformed in (original[:-5], bytes(crc)):
            source.write_bytes(malformed)
            with self.assertRaises(imp.ArchiveImportError):
                imp.import_archive(source, self.cache)
        for raw in ('{"schema":1,"schema":1}\n', '{"value":NaN}\n', '{"value":1e999}\n'):
            source.write_bytes(gzip.compress(raw.encode()))
            with self.assertRaises(imp.ArchiveImportError):
                imp.import_archive(source, self.cache)
        source = self.archive()
        for limit, value in (('MAX_COMPRESSED', 1), ('MAX_EXPANDED', 1), ('MAX_LINE', 1), ('MAX_ROWS', 0)):
            with self.subTest(limit=limit), patch.object(imp, limit, value), self.assertRaises(imp.ArchiveImportError):
                imp.import_archive(source, self.cache)
        self.assertFalse((self.cache / 'library.sqlite3').exists())
        self.assert_clean_staging()

    def test_conflicting_identity_rolls_back_and_keeps_backup(self):
        monitor = self.initial(content=True)
        try:
            original = self.snapshot()
            # A different valid placement claimed under an already cached ID.
            changed = dict(self.other, sha=self.sha)
            with self.assertRaisesRegex(imp.ArchiveImportError, 'conflicts'):
                imp.import_archive(self.archive([dict(self.row, sha='b' * 64), changed]), self.cache)
            self.assertEqual(self.snapshot(), original)
            self.assertEqual(len(list((self.cache / 'archive-imports').glob('library-before-*.sqlite3'))), 1)
        finally:
            monitor.close()
        self.assert_clean_staging()

    def test_cancellation_validation_and_precommit_preserve_cache(self):
        monitor = self.initial()
        try:
            original = self.snapshot()
            for target in ('validating', 'merging', 'ready'):
                cancelled = False

                def progress(event):
                    nonlocal cancelled
                    cancelled = event.get('ready_to_commit', False) if target == 'ready' else event['phase'] == target

                with self.subTest(target=target), self.assertRaises(imp.ArchiveCancelled):
                    imp.import_archive(self.archive([self.row, self.other]), self.cache,
                                       should_cancel=lambda: cancelled, on_progress=progress)
                self.assertEqual(self.snapshot(), original)
                self.assert_clean_staging()
        finally:
            monitor.close()

    def test_cancelled_fresh_merge_publishes_no_database(self):
        cancelled = False

        def progress(event):
            nonlocal cancelled
            cancelled = event.get('ready_to_commit', False)

        with self.assertRaises(imp.ArchiveCancelled):
            imp.import_archive(self.archive(), self.cache, should_cancel=lambda: cancelled, on_progress=progress)
        self.assertFalse((self.cache / 'library.sqlite3').exists())
        self.assert_clean_staging()

    def test_backup_precedes_schema_initialization(self):
        self.cache.mkdir()
        with closing(sqlite3.connect(self.cache / 'library.sqlite3')) as db:
            db.execute('CREATE TABLE old_data(value TEXT)')
            db.execute("INSERT INTO old_data VALUES('preserve')")
            db.commit()
        report = imp.import_archive(self.archive(), self.cache)
        with closing(sqlite3.connect(report['backup_path'])) as backup:
            self.assertEqual(backup.execute('SELECT * FROM old_data').fetchall(), [('preserve',)])
            self.assertEqual(backup.execute("SELECT name FROM sqlite_master WHERE name='boards'").fetchall(), [])

    def test_expected_hash_rejects_before_merge(self):
        source = self.archive()
        for value in ('invalid', '0' * 64):
            with self.subTest(value=value), self.assertRaises(imp.ArchiveImportError):
                imp.import_archive(source, self.cache, expected_sha256=value)
            self.assertFalse((self.cache / 'library.sqlite3').exists())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        result = imp.import_archive(source, self.cache, expected_sha256=digest)
        self.assertEqual(result['archive_sha256'], digest)

    def test_cleanup_failure_after_commit_returns_success_with_warning(self):
        original_unlink = Path.unlink

        def fail_stage_unlink(path, *args, **kwargs):
            if path.name.endswith('.stage.sqlite3'):
                raise PermissionError('injected temporary-file lock')
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, 'unlink', fail_stage_unlink):
            report = imp.import_archive(self.archive(), self.cache)
        self.assertTrue(report['committed'])
        self.assertEqual(report['new_boards'], 1)
        self.assertIn('injected temporary-file lock', report['cleanup_warnings'][0])
        saved = json.loads(Path(report['report_path']).read_text(encoding='utf-8'))
        self.assertEqual(saved['cleanup_warnings'], report['cleanup_warnings'])
        with closing(sqlite3.connect(self.cache / 'library.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT COUNT(local_sha) FROM boards').fetchone(), (1,))

    def test_cleanup_failure_preserves_original_precommit_exception(self):
        original_unlink = Path.unlink

        def fail_stage_unlink(path, *args, **kwargs):
            if path.name.endswith('.stage.sqlite3'):
                raise PermissionError('injected temporary-file lock')
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, 'unlink', fail_stage_unlink):
            with self.assertRaisesRegex(imp.ArchiveImportError, 'Board rules or score') as caught:
                imp.import_archive(self.archive([dict(self.row, score=480, breaks=0)]), self.cache)
        self.assertTrue(any('injected temporary-file lock' in note for note in caught.exception.__notes__))
        self.assertFalse((self.cache / 'library.sqlite3').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
