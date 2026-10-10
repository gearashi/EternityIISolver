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
            self.assertEqual(status['archive']['phase'], 'complete', status)
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
            self.assertEqual(again.status()['archive']['phase'],'idle')
        finally:again.close()

if __name__=='__main__':unittest.main(verbosity=2)
