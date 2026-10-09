"""Offline behavior tests against the independently validated downloaded fixtures."""
import gzip,json,unittest,tempfile
import inspect
from pathlib import Path
from unittest.mock import patch
import library_cache
from library_cache import LibraryMonitor, local_board_hash

ROOT=Path(__file__).resolve().parent/'data' if (Path(__file__).resolve().parent/'data').exists() else Path(__file__).resolve().parents[1]
DOC466=json.loads((ROOT/'record466.json').read_text())
SHA466='2c037e70f7e93518a48733c7aacd096226b7f23728efabd6289088b91c0945cd'
SEED=next(p for p in (ROOT/'seeds' if (ROOT/'seeds').exists() else ROOT/'gpu_design/seeds').glob('*.json') if len(p.stem)==64)
DOC465=json.loads(SEED.read_text())
INDEX={'schema':'eternity2-board-library-index/v2','count':2,'generated_at':'offline-fixture',
 'boards':[{'sha':SHA466,'score':466,'has_content':True},{'sha':SEED.stem,'score':465,'has_content':True}]}

class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='eternity-library-')
        self.path=Path(self.temporary.name)
        self.mon=LibraryMonitor(self.path,pieces_path=ROOT/'pieces.txt' if (ROOT/'pieces.txt').exists() else ROOT/'256pieces.txt')
    def tearDown(self):
        self.mon.stop();self.mon.close()
        self.temporary.cleanup()
    def test_unknown_then_exact_known_and_persist(self):
        self.assertIsNone(self.mon.is_known(DOC466['board']))
        self.mon.ingest_index(INDEX)
        self.assertIsNone(self.mon.is_known(DOC466['board']))
        self.mon.register_known_document(DOC466,SHA466)
        self.assertTrue(self.mon.is_known(DOC466['board']))
        self.assertIsNone(self.mon.is_known(DOC465['board']))
        again=LibraryMonitor(self.path)
        try:self.assertTrue(again.is_known(DOC466['board']))
        finally:again.close()
    def test_local_geometry_registration_and_complete_tiers(self):
        self.mon.ingest_index(INDEX)
        self.assertFalse(self.mon.warm_once())
        self.assertEqual(self.mon.status()['pending_geometries'],2)
        self.mon.register_known_document(DOC466,SHA466)
        self.mon.register_known_document(DOC465,SEED.stem)
        self.assertFalse(self.mon.warm_once())
        self.assertTrue(self.mon.status()['geometry_cache_complete'])
        self.assertTrue(self.mon.is_known(DOC465['board']))
    def test_missing_score_tier_is_absent_snapshot(self):
        self.mon.ingest_index(INDEX)
        changed=DOC466['board'][:];changed[17],changed[18]=changed[18],changed[17]
        self.assertNotIn(self.mon._score(changed),[465,466])
        self.assertFalse(self.mon.is_known(changed))
        old_score=self.mon._score;self.mon._score=lambda board:467
        try:self.assertFalse(self.mon.is_known(changed))
        finally:self.mon._score=old_score
    def test_poll_and_warm_preserve_snapshot_and_historical_timestamp(self):
        self.mon.ingest_index(INDEX,{'ETag':'"fixture"'})
        with self.mon._db:self.mon._put_meta('index_checked_at',123.0)
        before=self.mon.status();changes=self.mon._db.total_changes
        index_bytes=(self.path/'index.json.gz').read_bytes()
        for _ in range(3):
            self.mon.start();self.mon.poll_once();self.assertFalse(self.mon.warm_once())
        after=self.mon.status()
        self.assertEqual(after,before)
        self.assertEqual(self.mon._db.total_changes,changes)
        self.assertEqual((self.path/'index.json.gz').read_bytes(),index_bytes)
        self.assertEqual(after['last_index_check_unix'],123.0)
        self.assertTrue(after['snapshot_stale'])
        self.assertFalse(after['online_freshness_verified'])
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            self.mon.ingest_index({'count':0})
        self.assertEqual(self.mon.status(),before)

    def test_local_index_import_is_not_an_online_check(self):
        self.assertIsNone(self.mon.status()['last_index_check_unix'])
        with patch('library_cache.time.time',return_value=987.0):self.mon.ingest_index(INDEX)
        status=self.mon.status()
        self.assertEqual(status['last_local_index_load_unix'],987.0)
        self.assertIsNone(status['last_index_check_unix'])
        self.assertTrue(status['full_metadata_loaded'])
        self.assertTrue(status['snapshot_stale'])
        self.assertEqual(json.loads(gzip.decompress((self.path/'index.json.gz').read_bytes())),INDEX)
    def test_invalid_board_rejected(self):
        broken=DOC466['board'][:];broken[17]=broken[18]
        with self.assertRaises(ValueError):self.mon.is_known(broken)
        bad=dict(DOC466);bad['score']=480
        with self.assertRaises(ValueError):self.mon.register_known_document(bad,SHA466)
        self.assertEqual(self.mon.status()['known_exact_boards'],0)
    def test_bootstrap_and_compatibility_methods_never_connect_or_start_threads(self):
        bootstrap=self.path/'library-index.json';bootstrap.write_text(json.dumps(INDEX))
        with patch('socket.socket.connect',side_effect=AssertionError('Network connect forbidden')) as connect, \
             patch('socket.create_connection',side_effect=AssertionError('Network connection forbidden')) as create, \
             patch('urllib.request.urlopen',side_effect=AssertionError('URL request forbidden')) as urlopen, \
             patch('threading.Thread.start',side_effect=AssertionError('Background thread forbidden')) as start:
            # Old online timing/settings arguments must not restore networking.
            again=LibraryMonitor(self.path,interval_seconds=.01,request_timeout=.01,detail_interval_seconds=.01)
            try:
                self.assertEqual(again.status()['indexed_boards'],2)
                for _ in range(3):
                    self.assertIsNone(again.start())
                    status=again.poll_once()
                    self.assertFalse(again.warm_once())
                    self.assertTrue(again.stop(timeout=0))
                    self.assertFalse(status['running'])
                    self.assertFalse(status['network_enabled'])
                    self.assertEqual(status['mode'],'offline')
                    self.assertEqual(status['network_requests_this_session'],0)
                    self.assertEqual(status['detail_downloads_this_session'],0)
                    self.assertFalse(status['automatic_polling_enabled'])
                for mocked in (connect,create,urlopen,start):mocked.assert_not_called()
            finally:again.close()

    def test_existing_legacy_sqlite_cache_survives_without_refresh(self):
        self.mon.ingest_index(INDEX);self.mon.register_known_document(DOC466,SHA466)
        with self.mon._db:
            self.mon._db.execute("DELETE FROM metadata WHERE key='index_loaded_at'")
            self.mon._put_meta('index_checked_at',123.0)
        old_rows=self.mon._db.execute('SELECT * FROM boards ORDER BY public_sha').fetchall()
        again=LibraryMonitor(self.path,interval_seconds=1,request_timeout=60,detail_interval_seconds=1)
        try:
            again.start();again.poll_once();again.warm_once()
            self.assertTrue(again.is_known(DOC466['board']))
            self.assertIsNone(again.is_known(DOC465['board']))
            self.assertEqual(again._db.execute('SELECT * FROM boards ORDER BY public_sha').fetchall(),old_rows)
            self.assertEqual(again.status()['last_index_check_unix'],123.0)
            self.assertIsNone(again.status()['last_local_index_load_unix'])
        finally:again.close()

    def test_close_is_idempotent_and_reopened_cache_preserves_boards(self):
        self.mon.ingest_index(INDEX)
        self.mon.register_known_document(DOC466,SHA466)
        before=self.mon.status()
        self.assertTrue(self.mon.stop())
        self.assertEqual(self.mon.status(),before)
        self.mon.close();self.mon.close()
        self.assertTrue(self.mon.stop())
        # Renaming the closed database also exercises Windows handle release.
        database=self.path/'library.sqlite3'
        renamed=self.path/'closed-library.sqlite3'
        database.rename(renamed);renamed.rename(database)
        again=LibraryMonitor(self.path)
        try:
            self.assertTrue(again.is_known(DOC466['board']))
            self.assertEqual(again.status(),before)
        finally:again.close();again.close()

    def test_reader_observes_another_process_cache_commits(self):
        self.mon.ingest_index(INDEX)
        other=LibraryMonitor(self.path)
        try:
            self.assertIsNone(self.mon.is_known(DOC466['board']))
            other.register_known_document(DOC466,SHA466)
            self.assertTrue(self.mon.is_known(DOC466['board']))
            self.assertEqual(self.mon.status()['known_exact_boards'],1)
            other.register_known_document(DOC465,SEED.stem)
            self.assertTrue(self.mon.is_known(DOC465['board']))
            self.assertEqual(self.mon.status()['known_exact_boards'],2)
        finally:other.close()

    def test_invalid_refresh_preserves_cached_scores_and_snapshot(self):
        self.mon.ingest_index(INDEX)
        self.mon.register_known_document(DOC466,SHA466)
        before=self.mon.status()
        raw=(self.path/'index.json.gz').read_bytes()
        bad=json.loads(json.dumps(INDEX));bad['boards'][0]['score']=480
        with self.assertRaisesRegex(ValueError,'validated cached'):
            self.mon.ingest_index(bad)
        self.assertEqual(self.mon.status(),before)
        self.assertEqual((self.path/'index.json.gz').read_bytes(),raw)
        self.assertTrue(self.mon.is_known(DOC466['board']))
        self.assertEqual(list(self.path.glob('index.*.tmp')),[])

    def test_public_identity_cannot_be_rebound(self):
        self.mon.register_known_document(DOC466,SHA466)
        # Force only the claimed score comparison through to isolate identity.
        with self.mon._db:
            self.mon._db.execute('UPDATE boards SET score=? WHERE public_sha=?',(DOC465['score'],SHA466))
        with self.assertRaisesRegex(ValueError,'identity changed'):
            self.mon.register_known_document(DOC465,SHA466)
        self.assertTrue(self.mon.is_known(DOC466['board']))
        self.assertEqual(self.mon.status()['known_exact_boards'],1)

    def test_runtime_has_no_download_implementation_or_endpoint(self):
        self.assertFalse(hasattr(LibraryMonitor,'_request'))
        self.assertFalse(hasattr(LibraryMonitor,'_decode'))
        self.assertFalse(hasattr(LibraryMonitor,'_loop'))
        self.assertFalse(hasattr(library_cache,'API'))
        source=inspect.getsource(library_cache)
        self.assertNotIn('urllib',source)
        self.assertNotIn('https://',source)
        self.assertNotIn('http://',source)

if __name__=='__main__':unittest.main(verbosity=2)
