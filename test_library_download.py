"""Mocked public-library transport checks; no test contacts the project."""
import copy
import gzip
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import library_download
from library_download import LibraryDownloader

DATA = Path(__file__).resolve().parent / 'data'
DOC466 = json.loads((DATA / 'record466.json').read_text(encoding='utf-8'))
SHA466 = '2c037e70f7e93518a48733c7aacd096226b7f23728efabd6289088b91c0945cd'
SEED = next(path for path in (DATA / 'seeds').glob('*.json') if len(path.stem) == 64)
DOC465 = json.loads(SEED.read_text(encoding='utf-8'))
INDEX = {'schema': 'eternity2-board-library-index/v2', 'count': 2, 'generated_at': 'fixture',
         'boards': [{'sha': SEED.stem, 'score': 465, 'has_content': True},
                    {'sha': SHA466, 'score': 466, 'has_content': True}]}


class Response:
    def __init__(self, document=None, *, body=None, status=200, headers=None, on_read=None):
        self.status = status
        self.body = json.dumps(document).encode() if body is None else body
        self.headers = {'content-type': 'application/json', 'content-length': str(len(self.body))}
        self.headers.update({key.lower(): value for key, value in (headers or {}).items()})
        self.offset = 0
        self.on_read = on_read

    def getheader(self, name):
        return self.headers.get(name.lower())

    def read(self, count):
        if self.on_read:
            self.on_read()
        chunk = self.body[self.offset:self.offset+count]
        self.offset += len(chunk)
        return chunk


class FakeSocket:
    def __init__(self):
        self.shutdown_calls = 0
        self.timeout = None

    def shutdown(self, how):
        self.shutdown_calls += 1

    def settimeout(self, timeout):
        self.timeout = timeout


class Connection:
    def __init__(self, transport, host, timeout):
        self.transport = transport
        self.host = host
        self.timeout = timeout
        self.sock = FakeSocket()
        self.closed = False
        self.call = None
        self.response = None

    def request(self, method, path, body=None, headers=None):
        self.call = (method, path, body, dict(headers))
        if not self.transport.responses:
            raise AssertionError('Unexpected extra request')
        self.response = self.transport.responses.pop(0)
        if isinstance(self.response, Exception):
            raise self.response

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class Transport:
    def __init__(self):
        self.responses = []
        self.connections = []

    def __call__(self, host, *, timeout):
        connection = Connection(self, host, timeout)
        self.connections.append(connection)
        return connection


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='eternity-download-test-')
        self.path = Path(self.temporary.name)
        self.transport = Transport()
        self.network = patch('library_download.http.client.HTTPSConnection', self.transport)
        self.network.start()
        self.mon = LibraryDownloader(self.path, pieces_path=DATA/'pieces.txt')

    def tearDown(self):
        self.mon.close()
        self.network.stop()
        self.temporary.cleanup()

    def enable_manual(self):
        # Explicitly enable through the public API, but drive requests manually
        # for deterministic tests without wall-clock sleeps.
        with patch.object(self.mon, '_start_thread'):
            self.mon.start()

    def permit_request(self):
        self.mon._next_request = 0

    def wait_until(self, predicate, seconds=3):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.005)
        self.fail('Timed out waiting for mocked downloader')

    def test_construction_poll_and_warm_remain_offline_until_start(self):
        self.mon.ingest_index(INDEX)
        self.mon.poll_once()
        self.assertFalse(self.mon.warm_once())
        self.assertEqual(self.transport.connections, [])
        status = self.mon.status()
        self.assertEqual(status['mode'], 'offline')
        self.assertFalse(status['network_enabled'])
        self.assertFalse(status['uploads_enabled'])
        self.assertFalse(status['progress_reporting_enabled'])
        with self.assertRaisesRegex(ValueError, 'trusted'):
            LibraryDownloader(self.path/'missing-pieces')

    def test_transport_is_fixed_https_get_without_private_headers_or_proxies(self):
        with self.mon._db:
            self.mon._put_meta('etag', '"fake"\r\nAuthorization: secret')
            self.mon._put_meta('last_modified', 'Cookie: secret')
        self.transport.responses = [Response(INDEX, headers={'Set-Cookie': 'session=untrusted'})]
        self.enable_manual()
        with patch.dict(os.environ, {'HTTPS_PROXY': 'http://foreign.invalid:1234', 'HTTP_PROXY': 'http://foreign.invalid:1234'}):
            self.mon.poll_once()
        connection = self.transport.connections[0]
        method, path, body, headers = connection.call
        self.assertEqual((connection.host, connection.timeout), (library_download.HOST, 20))
        self.assertEqual((method, path, body), ('GET', '/library/index', None))
        self.assertEqual(set(headers), {'Accept', 'Accept-Encoding', 'User-Agent'})
        self.assertTrue(connection.closed)
        status = self.mon.status()
        self.assertEqual(status['network_requests_this_session'], 1)
        self.assertEqual(status['successful_requests_this_session'], 1)
        self.assertTrue(status['online_freshness_verified'])
        self.assertEqual(status['mode'], 'read-only-downloads')
        self.mon.poll_once()
        self.assertFalse(self.mon.warm_once(), 'Details must wait at least one second')
        self.assertEqual(len(self.transport.connections), 1)

    def test_arbitrary_urls_and_malformed_hashes_never_reach_transport(self):
        self.enable_manual()
        for sha in ('A'*64, 'a'*63, 'a'*65, 'https://other.invalid/', '../index', 'a'*64+'?token=x', '\r\n', True):
            with self.subTest(sha=sha), self.assertRaises(ValueError):
                self.mon._fetch(sha)
        self.assertEqual(self.transport.connections, [])

    def test_canonical_pieces_are_checked_before_creating_download_cache(self):
        wrong = self.path/'wrong-pieces.txt'
        rows = (DATA/'pieces.txt').read_text().splitlines()
        first = rows[0].split(); first[0] = str((int(first[0])+1) % 23)
        rows[0] = ' '.join(first)
        wrong.write_text('\n'.join(rows))
        rejected = self.path/'rejected-cache'
        with self.assertRaisesRegex(ValueError, 'trusted official'):
            LibraryDownloader(rejected, pieces_path=wrong)
        self.assertFalse(rejected.exists())
        self.assertEqual(self.transport.connections, [])

    def test_redirects_are_errors_and_are_never_followed(self):
        self.enable_manual()
        for code in (301, 302, 303, 307, 308):
            self.permit_request()
            self.transport.responses.append(Response({}, status=code, headers={'Location': 'https://foreign.invalid/upload'}))
            self.mon.poll_once()
        self.assertEqual(len(self.transport.connections), 5)
        self.assertTrue(all(c.host == library_download.HOST and c.call[1] == '/library/index' and c.closed for c in self.transport.connections))
        self.assertEqual(self.mon.status()['failed_requests_this_session'], 5)
        self.assertFalse(self.mon.status()['full_metadata_loaded'])

    def test_gzip_index_and_only_missing_details_highest_score_first(self):
        index = copy.deepcopy(INDEX)
        index['boards'].append({'sha': 'f'*64, 'score': 480, 'has_content': False})
        index['count'] = 3
        self.transport.responses = [Response(body=gzip.compress(json.dumps(index).encode()), headers={'Content-Encoding': 'gzip'}),
                                    Response(DOC466), Response(DOC465)]
        self.enable_manual()
        self.mon.poll_once()
        self.permit_request(); self.assertTrue(self.mon.warm_once())
        self.permit_request(); self.assertTrue(self.mon.warm_once())
        self.permit_request(); self.assertFalse(self.mon.warm_once())
        self.assertEqual([c.call[1] for c in self.transport.connections],
                         ['/library/index', '/library/board/'+SHA466, '/library/board/'+SEED.stem])
        self.assertTrue(self.mon.is_known(DOC466['board']))
        self.assertTrue(self.mon.is_known(DOC465['board']))
        self.assertEqual(self.mon.status()['detail_downloads_this_session'], 2)
        self.assertTrue(all(c.closed for c in self.transport.connections))

    def test_cached_geometry_is_reused_across_refresh_and_restart(self):
        self.mon.ingest_index(INDEX)
        self.mon.register_known_document(DOC466, SHA466)
        self.mon.close()
        self.mon = LibraryDownloader(self.path, pieces_path=DATA/'pieces.txt')
        self.assertTrue(self.mon.is_known(DOC466['board']))
        self.transport.responses = [Response(INDEX), Response(DOC465)]
        self.enable_manual(); self.mon.poll_once()
        self.permit_request(); self.assertTrue(self.mon.warm_once())
        self.assertEqual(self.transport.connections[-1].call[1], '/library/board/'+SEED.stem)
        self.permit_request(); self.assertFalse(self.mon.warm_once())
        self.assertEqual(len(self.transport.connections), 2)

    def test_failures_back_off_immediately_and_persist_across_restart(self):
        self.transport.responses = [OSError('fixture connection failure')]
        self.enable_manual(); self.mon.poll_once()
        for _ in range(5):
            self.mon.poll_once(); self.mon.warm_once()
        status = self.mon.status()
        self.assertEqual(status['network_requests_this_session'], 1)
        self.assertEqual(status['failed_requests_this_session'], 1)
        self.assertGreater(status['next_request_in_seconds'], 29)
        self.assertIsNotNone(status['last_error'])
        self.mon.stop(0); self.enable_manual(); self.mon.poll_once()
        self.assertEqual(len(self.transport.connections), 1)
        self.mon.close()
        self.mon = LibraryDownloader(self.path, pieces_path=DATA/'pieces.txt')
        self.enable_manual(); self.mon.poll_once()
        self.assertEqual(len(self.transport.connections), 1)

    def test_failed_detail_has_retry_marker_and_keeps_cache_valid(self):
        self.mon.ingest_index(INDEX)
        self.transport.responses = [OSError('fixture detail failure')]
        self.enable_manual()
        self.assertFalse(self.mon.warm_once())
        row = self.mon._db.execute('SELECT retry_after,failures,local_sha FROM boards WHERE public_sha=?', (SHA466,)).fetchone()
        self.assertGreater(row[0], time.time())
        self.assertEqual(row[1:], (1, None))
        self.assertEqual(self.mon.status()['cached_geometries'], 0)

    def test_failed_top_detail_does_not_starve_other_missing_boards(self):
        self.mon.ingest_index(INDEX)
        self.transport.responses = [OSError('broken top detail'), Response(DOC465)]
        self.enable_manual()
        self.assertFalse(self.mon.warm_once())
        self.permit_request()  # Simulate expiration of the shorter global backoff.
        self.assertTrue(self.mon.warm_once())
        self.assertEqual([c.call[1] for c in self.transport.connections],
                         ['/library/board/'+SHA466, '/library/board/'+SEED.stem])
        self.assertTrue(self.mon.is_known(DOC465['board']))
        self.assertIsNone(self.mon.is_known(DOC466['board']))

    def test_connection_setup_failure_is_counted_and_backed_off(self):
        self.enable_manual()
        with patch('library_download.http.client.HTTPSConnection', side_effect=OSError('TLS fixture failure')):
            self.mon.poll_once(); self.mon.poll_once()
        status = self.mon.status()
        self.assertEqual(status['network_requests_this_session'], 1)
        self.assertEqual(status['failed_requests_this_session'], 1)
        self.assertEqual(status['requests_in_flight'], 0)
        self.assertGreater(status['next_request_in_seconds'], 29)

    def test_invalid_or_changed_index_keeps_previous_cache(self):
        self.mon.ingest_index(INDEX); self.mon.register_known_document(DOC466, SHA466)
        saved = (self.path/'index.json.gz').read_bytes()
        bad = copy.deepcopy(INDEX); bad['boards'][1]['score'] = 480
        malformed = copy.deepcopy(INDEX); malformed['boards'][1] = None
        self.transport.responses = [Response(bad), Response(malformed), Response({'schema': 'wrong'})]
        self.enable_manual()
        for _ in range(3):
            self.permit_request(); self.mon.poll_once()
            self.assertEqual((self.path/'index.json.gz').read_bytes(), saved)
            self.assertTrue(self.mon.is_known(DOC466['board']))
        self.assertEqual(self.mon.status()['failed_requests_this_session'], 3)

    def test_invalid_detail_cannot_poison_palette_or_geometry(self):
        self.mon.ingest_index(INDEX)
        base = self.mon._meta('canonical_faces')
        bad = copy.deepcopy(DOC466); bad['board'][17] = bad['board'][18]
        wrong_edges = copy.deepcopy(DOC466); wrong_edges['board_edges'] = 'b'*1024
        self.enable_manual()
        for doc in (bad, wrong_edges, dict(DOC466, score=480)):
            self.permit_request()
            with self.mon._db:
                self.mon._db.execute('UPDATE boards SET retry_after=0')
            self.transport.responses.append(Response(doc))
            self.assertFalse(self.mon.warm_once())
            self.assertEqual(self.mon._meta('canonical_faces'), base)
            self.assertEqual(self.mon.status()['known_exact_boards'], 0)

    def test_response_type_header_and_json_validation(self):
        cases = [Response({}, headers={'Content-Type': 'text/html'}),
                 Response({}, headers={'Content-Encoding': 'br'}),
                 Response({}, headers={'Content-Length': '-1'}),
                 Response({}, headers={'Content-Length': '９'}),
                 Response({}, headers={'Content-Length': str(library_download.INDEX_LIMIT+1)}),
                 Response(body=b'{"a":1,"a":2}'), Response(body=b'{"a":NaN}'),
                 Response(body=b'not JSON'), Response({}, headers={'Content-Length': '99'})]
        self.enable_manual()
        for response in cases:
            self.permit_request(); self.transport.responses.append(response); self.mon.poll_once()
        self.assertEqual(self.mon.status()['failed_requests_this_session'], len(cases))
        self.assertFalse(self.mon.status()['full_metadata_loaded'])
        self.assertTrue(all(c.closed for c in self.transport.connections))

    def test_compressed_and_decompressed_limits_and_bad_gzip(self):
        self.enable_manual()
        bomb = gzip.compress(json.dumps({'padding': 'x'*3000}).encode())
        cases = [Response(body=b' '*513, headers={'Content-Length': None}),
                 Response(body=bomb, headers={'Content-Encoding': 'gzip'}),
                 Response(body=gzip.compress(b'{}')[:-3], headers={'Content-Encoding': 'gzip'}),
                 Response(body=gzip.compress(b'{}')+gzip.compress(b'{}'), headers={'Content-Encoding': 'gzip'})]
        with patch.object(library_download, 'INDEX_LIMIT', 512):
            for response in cases:
                self.permit_request(); self.transport.responses.append(response); self.mon.poll_once()
        self.assertEqual(self.mon.status()['failed_requests_this_session'], 4)
        self.assertFalse(self.mon.status()['full_metadata_loaded'])

    def test_request_deadline_aborts_slow_reads(self):
        now = [10.0]
        def slow_read():
            now[0] += 21
        self.transport.responses = [Response(INDEX, on_read=slow_read)]
        self.enable_manual()
        self.permit_request()
        with patch('library_download.time.monotonic', side_effect=lambda: now[0]):
            self.mon.poll_once()
        self.assertEqual(self.mon.status()['failed_requests_this_session'], 1)
        self.assertTrue(self.transport.connections[0].closed)

    def test_stop_zero_is_immediate_status_stays_responsive_and_close_defers(self):
        entered, release = threading.Event(), threading.Event()
        def blocked_read():
            entered.set()
            if not release.wait(5):
                raise TimeoutError('fixture was not released')
        self.transport.responses = [Response(INDEX, on_read=blocked_read)]
        self.mon.start()
        try:
            self.assertTrue(entered.wait(2))
            began = time.monotonic()
            self.assertFalse(self.mon.stop(timeout=0))
            status = self.mon.status()
            self.assertLess(time.monotonic()-began, .5)
            self.assertFalse(status['network_enabled'])
            self.assertEqual(status['mode'], 'offline')
            self.assertEqual(status['requests_in_flight'], 1)
            self.assertFalse(self.mon.close(timeout=0))
            self.assertFalse(self.mon._closed, 'SQLite must remain open while the request unwinds')
        finally:
            release.set()
        self.wait_until(lambda: self.mon._closed)
        status = self.mon.status()
        self.assertEqual(status['cancelled_requests_this_session'], 1)
        self.assertEqual(status['requests_in_flight'], 0)
        self.assertTrue(self.mon.close())

    def test_disable_reenable_during_request_never_spawns_duplicate_downloads(self):
        entered, release = threading.Event(), threading.Event()
        def blocked_read():
            entered.set()
            if not release.wait(5):
                raise TimeoutError('fixture was not released')
        self.transport.responses = [Response(INDEX, on_read=blocked_read), Response(INDEX)]
        self.mon.start()
        try:
            self.assertTrue(entered.wait(2))
            original_thread = self.mon._thread
            self.mon.stop(timeout=0)
            self.mon.start(); self.mon.start()
            self.assertIs(self.mon._thread, original_thread)
            self.assertEqual(len(self.transport.connections), 1)
        finally:
            release.set()
        self.wait_until(lambda: self.mon.status()['index_downloads_this_session'] == 1)
        self.assertTrue(self.mon.stop())
        self.assertEqual(len(self.transport.connections), 2)
        status = self.mon.status()
        self.assertEqual(status['cancelled_requests_this_session'], 1)
        self.assertEqual(status['failed_requests_this_session'], 0)
        self.assertEqual(status['successful_requests_this_session'], 1)


if __name__ == '__main__':
    unittest.main()
