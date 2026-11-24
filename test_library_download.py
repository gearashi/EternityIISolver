"""Manual archive transfer tests: synthetic gzip responses, no outside network."""
import gzip
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from app_paths import resource_root
from library_download import LibraryArchiveManager, ARCHIVE_HOST, ARCHIVE_PATH
from validator import load_bundle


class Response:
    status = 200
    def __init__(self, body, *, status=200, headers=None):
        self.status = status
        self.body = io.BytesIO(body)
        self.headers = {'Content-Type': 'application/gzip', **(headers or {})}
    def getheader(self, name):
        return self.headers.get(name)
    def read1(self, size):
        return self.body.read(size)


class Connection:
    def __init__(self, response, calls):
        self.response = response
        self.calls = calls
        self.sock = None
        self.closed = False
    def request(self, method, path, body=None, headers=None):
        self.calls.append((method, path, body, headers))
    def getresponse(self):
        return self.response
    def close(self):
        self.closed = True


class ManualArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='eternity-manual-archive-')
        self.path = Path(self.temporary.name)
        bundle = load_bundle()
        self.record = {'sha': '2c037e70f7e93518a48733c7aacd096226b7f23728efabd6289088b91c0945cd',
                       'score': 466, 'breaks': 14, 'board': list(bundle.record_board)}
        self.header = {'schema': 1, 'count': 1, 'generated_at': '2026-10-09T23:45:22Z'}
        self.payload = gzip.compress(('\n'.join(json.dumps(v) for v in (self.header, self.record))+'\n').encode())
        self.manager = LibraryArchiveManager(self.path, pieces_path=resource_root()/'data/pieces.txt')
    def tearDown(self):
        self.assertTrue(self.manager.close())
        self.assertEqual(self.path.resolve().parent, Path(tempfile.gettempdir()).resolve())
        self.temporary.cleanup()
    def run_download(self, response):
        calls = []
        connection = Connection(response, calls)
        with patch('library_download.http.client.HTTPSConnection', return_value=connection) as factory:
            self.assertTrue(self.manager.start_download())
            self.manager._thread.join(8)
            self.assertFalse(self.manager._thread.is_alive())
        factory.assert_called_once_with(ARCHIVE_HOST, timeout=20)
        self.assertTrue(connection.closed)
        return self.manager.status(), calls

    def test_fast_publication_during_merge_is_replayed_after_commit_and_restart(self):
        import exact_worker
        from library_cache import local_board_hash, queue_local_result
        from manual_export import export_best
        from validator import validate_board

        self.manager.close()
        directory = self.path / 'library'
        pieces = resource_root() / 'data/pieces.txt'
        self.manager = LibraryArchiveManager(directory, pieces_path=pieces)
        entered, release = threading.Event(), threading.Event()
        observe = self.manager._progress
        def pause_merge(info):
            observe(info)
            if info.get('merged_boards') == 1:
                entered.set()
                if not release.wait(30):
                    raise RuntimeError('Test did not release archive transaction')
        board = list(self.record['board'])
        board[17] = board[17] // 4 * 4 + (board[17] + 1) % 4
        candidate = {'board': board, 'score': validate_board(board)['score']}
        def publish(**kwargs):
            kwargs['on_board'](candidate)
            return {'outcome': 'timeout'}
        calls = []
        response = Connection(Response(self.payload), calls)
        with patch('library_download.http.client.HTTPSConnection', return_value=response), \
                patch.object(self.manager,'_progress',pause_merge):
            self.assertTrue(self.manager.start_download())
            try:
                self.assertTrue(entered.wait(10))
                with patch.object(exact_worker, 'peak_memory_mb', return_value=64):
                    code = exact_worker.run_task(self.path,engine='fast-score',seconds=20,solver_factory=lambda _:publish)
                self.assertEqual(0,code)
                self.assertEqual(candidate['score'], export_best(self.path)['score'])
                self.assertFalse(self.manager.is_known(board))
                self.assertTrue(list((directory/'pending-local').glob('*.json')))
            finally:
                release.set()
                self.manager._thread.join(15)
        self.assertEqual(1, len(calls))
        self.assertEqual('complete', self.manager.status()['archive']['phase'])
        self.assertTrue(self.manager.is_known(board))
        self.assertFalse(list((directory/'pending-local').glob('*.json')))
        self.manager.close()
        # Simulate shutdown after durable publication, before SQLite acknowledges
        # it. Replaying an already committed result must also be idempotent.
        queue_local_result(directory, dict(candidate,sha256_uint16le=local_board_hash(board)))
        another = board.copy()
        another[18] = another[18] // 4 * 4 + (another[18]+1) % 4
        queue_local_result(directory, {'board':another, 'score':validate_board(another)['score'],
                                       'sha256_uint16le':local_board_hash(another)})
        self.manager = LibraryArchiveManager(directory,pieces_path=pieces)
        self.assertEqual(3, self.manager.status()['known_exact_boards'])
        self.assertTrue(self.manager.is_known(another))
        self.assertFalse(list((directory/'pending-local').glob('*.json')))
    def test_constructor_status_and_legacy_start_do_not_connect(self):
        with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')), \
             patch('threading.Thread.start', side_effect=AssertionError('No automatic jobs')):
            other = LibraryArchiveManager(self.path/'other', pieces_path=resource_root()/'data/pieces.txt')
            try:
                other.start(); other.poll_once(); other.warm_once()
                status = other.status()
                self.assertEqual(status['archive']['phase'], 'idle')
                self.assertFalse(status['network_enabled'])
                self.assertEqual(status['network_requests_this_session'], 0)
                self.assertFalse(status['automatic_polling_enabled'])
            finally: other.close()
    def test_one_get_streams_archive_then_merges_offline_and_reuses_duplicates(self):
        for expected_new, expected_reused in ((1,0),(0,1)):
            status, calls = self.run_download(Response(self.payload))
            self.assertEqual(len(calls), 1)
            method, path, body, headers = calls[0]
            self.assertEqual((method,path,body), ('GET', ARCHIVE_PATH, None))
            self.assertEqual(set(headers), {'Accept','Accept-Encoding','User-Agent'})
            self.assertNotIn('?', path)
            self.assertEqual(status['archive']['phase'], 'complete' if expected_new else 'unchanged', status)
            self.assertEqual(status['archive']['new_boards'], expected_new)
            self.assertEqual(status['archive']['already_cached'], expected_reused)
            self.assertEqual(status['known_exact_boards'], 1)
            self.assertFalse(status['network_enabled'])
            self.assertFalse(status['uploads_enabled'])
            self.assertFalse(status['progress_reporting_enabled'])
            self.assertFalse(status['automatic_polling_enabled'])
            self.assertTrue(Path(status['archive']['archive_path']).is_file())
            self.assertEqual(list(self.path.glob('.archive-download-*')), [])
    def test_redirect_error_and_unexpected_payload_never_retry_or_change_cache(self):
        cases = [Response(b'',status=302,headers={'Location':'https://example.invalid/'}),
                 Response(b'',status=503), Response(b'<html/>',headers={'Content-Type':'text/html'}),
                 Response(self.payload,headers={'Content-Length':str(len(self.payload)+1)}),
                 Response(self.payload,headers={'Content-Encoding':'gzip'}),
                 Response(b'broken gzip'), Response(b'')]
        for response in cases:
            with self.subTest(response=response):
                status, calls = self.run_download(response)
                self.assertEqual(status['archive']['phase'], 'error')
                self.assertEqual(len(calls), 1)
                self.assertEqual(status['known_exact_boards'], 0)
                self.assertTrue(status['archive_can_start'])
                self.assertEqual(list(self.path.glob('.archive-download-*')), [])
    def test_size_limits_are_checked_with_and_without_length(self):
        for headers in ({'Content-Length':'101'}, {}):
            with self.subTest(headers=headers), patch('library_download.MAX_COMPRESSED', 100):
                status,calls = self.run_download(Response(b'x'*101,headers=headers))
                self.assertEqual(status['archive']['phase'], 'error')
                self.assertEqual(status['known_exact_boards'], 0)
                self.assertEqual(len(calls),1)
    def test_busy_request_does_not_queue_a_second_transfer_and_cancel_unwinds(self):
        entered = threading.Event(); release = threading.Event()
        class WaitingResponse(Response):
            def read1(self, size):
                entered.set()
                if not release.wait(5): raise TimeoutError('Test transfer timeout')
                raise OSError('Connection interrupted')
        calls=[]
        connection=Connection(WaitingResponse(self.payload),calls)
        class Sock:
            def shutdown(self,*args):release.set()
            def settimeout(self,*args):pass
        connection.sock=Sock()
        with patch('library_download.http.client.HTTPSConnection',return_value=connection) as factory:
            self.assertTrue(self.manager.start_download())
            self.assertTrue(entered.wait(3))
            self.assertTrue(self.manager.status()['network_enabled'])
            self.assertFalse(self.manager.start_download())
            self.assertTrue(self.manager.cancel())
            self.manager._thread.join(5)
            self.assertFalse(self.manager._thread.is_alive())
            factory.assert_called_once()
        status=self.manager.status()
        self.assertEqual(status['archive']['phase'],'cancelled')
        self.assertEqual(status['known_exact_boards'],0)
        self.assertFalse(status['network_enabled'])
        self.assertFalse(status['archive_can_cancel'])
        self.assertEqual(len(calls),1)
    def test_cancel_during_validation_preserves_cache(self):
        from archive_import import ArchiveCancelled
        entered=threading.Event();release=threading.Event()
        def importing(*args,should_cancel,on_progress,**kwargs):
            on_progress({'phase':'validating','validated_boards':0,'total_boards':1})
            entered.set()
            if not release.wait(5):raise TimeoutError('Test import timeout')
            self.assertTrue(should_cancel())
            raise ArchiveCancelled('Cancelled fixture')
        connection=Connection(Response(self.payload),[])
        with patch('library_download.http.client.HTTPSConnection',return_value=connection), \
             patch('library_download.import_archive',side_effect=importing):
            self.manager.start_download();self.assertTrue(entered.wait(3))
            self.assertFalse(self.manager.status()['network_enabled'])
            self.assertTrue(self.manager.cancel());release.set()
            self.manager._thread.join(5)
        self.assertEqual(self.manager.status()['archive']['phase'],'cancelled')
        self.assertEqual(self.manager.status()['known_exact_boards'],0)
    def test_thread_launch_failure_can_be_retried_and_closed(self):
        with patch('library_download.threading.Thread.start', side_effect=RuntimeError('Thread unavailable')):
            self.assertFalse(self.manager.start_download())
        status = self.manager.status()
        self.assertEqual(status['archive']['phase'], 'error')
        self.assertFalse(status['archive']['running'])
        self.assertTrue(status['archive_can_start'])
        self.assertEqual(status['network_requests_this_session'], 0)
        status, calls = self.run_download(Response(self.payload))
        self.assertEqual(status['archive']['phase'], 'complete')
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.manager.close())

    def test_closed_manager_cannot_start_and_cached_counts_survive_restart(self):
        self.run_download(Response(self.payload))
        self.manager.close()
        with self.assertRaises(RuntimeError):self.manager.start_download()
        again=LibraryArchiveManager(self.path,pieces_path=resource_root()/'data/pieces.txt')
        try:
            self.assertEqual(again.status()['known_exact_boards'],1)
            self.assertEqual(again.status()['network_requests_this_session'],0)
            self.assertEqual(again.status()['archive']['phase'],'complete')
            self.assertEqual(again.status()['archive']['new_boards'],1)
        finally:again.close()

    def test_conditional_refresh_after_restart_does_not_read_or_validate_body(self):
        self.run_download(Response(self.payload, headers={'ETag': '"snapshot-1"'}))
        self.manager.close()
        self.manager = LibraryArchiveManager(self.path, pieces_path=resource_root()/'data/pieces.txt')
        response = Response(b'', status=304)
        with patch.object(response, 'read1', side_effect=AssertionError('304 has no body')), \
             patch('library_download.import_archive', side_effect=AssertionError('already validated')):
            status, calls = self.run_download(response)
        self.assertEqual(calls[0][3]['If-None-Match'], '"snapshot-1"')
        self.assertEqual(status['archive']['phase'], 'unchanged')
        self.assertEqual(status['archive']['bytes_downloaded'], 0)
        self.assertEqual(status['known_exact_boards'], 1)

    def test_failed_refresh_never_marks_the_new_validator_as_merged(self):
        self.run_download(Response(self.payload, headers={'ETag': '"good"'}))
        failed, _ = self.run_download(Response(b'broken gzip', headers={'ETag': '"bad"'}))
        self.assertEqual(failed['archive']['phase'], 'error')
        _, calls = self.run_download(Response(b'', status=304))
        self.assertEqual(calls[0][3]['If-None-Match'], '"good"')
        self.assertEqual(self.manager.status()['known_exact_boards'], 1)

    def test_matching_200_response_headers_skip_the_body(self):
        for generation, headers in enumerate((
                {'ETag': '"identical"'}, {'Last-Modified': 'Tue, 24 Nov 2026 01:02:03 GMT'}), start=1):
            with self.subTest(headers=headers):
                # A new compressed representation initially receives these validators.
                body = gzip.compress(gzip.decompress(self.payload), mtime=generation)
                headers = dict(headers, **{'Content-Length': str(len(body))})
                self.run_download(Response(body, headers=headers))
                response = Response(body, headers=headers)
                with patch.object(response, 'read1', side_effect=AssertionError('unnecessary body transfer')):
                    status, _ = self.run_download(response)
                self.assertEqual(status['archive']['phase'], 'unchanged')
                self.assertEqual(status['archive']['bytes_downloaded'], 0)

    def test_identical_200_body_skips_validation_when_no_http_validator_exists(self):
        self.run_download(Response(self.payload))
        with patch('library_download.import_archive', side_effect=AssertionError('duplicate validation')):
            status, calls = self.run_download(Response(self.payload))
        self.assertEqual(len(calls), 1)
        self.assertEqual(status['archive']['phase'], 'unchanged')
        self.assertEqual(status['archive']['new_boards'], 0)

    def test_rate_limit_reason_survives_restart_and_next_success_clears_it(self):
        status, calls = self.run_download(Response(b'', status=429, headers={'Retry-After': '60'}))
        self.assertEqual(len(calls), 1)
        self.assertIn('HTTP 429', status['archive']['error'])
        self.manager.close()
        self.manager = LibraryArchiveManager(self.path, pieces_path=resource_root()/'data/pieces.txt')
        self.assertIn('after 60 seconds', self.manager.status()['archive']['error'])
        self.run_download(Response(self.payload))
        self.manager.close()
        self.manager = LibraryArchiveManager(self.path, pieces_path=resource_root()/'data/pieces.txt')
        self.assertEqual(self.manager.status()['archive']['phase'], 'complete')
        self.assertIsNone(self.manager.status()['archive']['error'])

    def test_continuing_download_can_outlast_the_original_ten_minute_cutoff(self):
        # Scale the byte milestone to this small real gzip and advance a fake clock.
        from types import SimpleNamespace
        clock = SimpleNamespace(now=0)
        class SlowResponse(Response):
            def read1(self, size):
                clock.now += 10  # steady progress inside the socket's 20-second idle timeout
                return super().read1(min(size, 8))
        with patch('library_download.time', SimpleNamespace(monotonic=lambda: clock.now, time=lambda: 0)), \
             patch('library_download.PROGRESS_BYTES', 32):
            status, calls = self.run_download(SlowResponse(self.payload))
        self.assertGreater(clock.now, 600)
        self.assertEqual(len(calls), 1)
        self.assertEqual(status['archive']['phase'], 'complete', status)
        self.assertEqual(status['known_exact_boards'], 1)

if __name__=='__main__':unittest.main(verbosity=2)
