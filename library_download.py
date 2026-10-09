"""Explicitly enabled, GET-only downloads into the otherwise offline library cache.

Only the fixed public index and public board-document paths are implemented.
This module has no upload, progress-reporting, cookie, proxy or redirect support.
"""
import http.client
import json
import math
import socket
import threading
import time
import zlib

from library_cache import LibraryMonitor, SHA_RE
from validator import load_bundle, read_pieces, validate_board

HOST = 'eternity-control-plane-prod.eternity-cp.workers.dev'
INDEX_LIMIT = 64 * 1024 * 1024
DETAIL_LIMIT = 256 * 1024
READ_CHUNK = 64 * 1024
BACKOFF_SECONDS = 30


class _Cancelled(Exception):
    pass


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON object key')
        result[key] = value
    return result


def _bad_constant(_):
    raise ValueError('Non-finite JSON number')


class LibraryDownloader(LibraryMonitor):
    """Network access starts only at start(); stop() immediately disables it."""
    def __init__(self, data_dir, interval_seconds=900, *, pieces_path=None,
                 request_timeout=20, detail_interval_seconds=1.0):
        if pieces_path is None:
            raise ValueError('Downloads require a trusted canonical pieces_path')
        for value in (interval_seconds, request_timeout, detail_interval_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Download timing settings must be finite and positive')
        if request_timeout > 20:
            raise ValueError('Request timeout must not exceed 20 seconds')
        bundle = load_bundle()
        if read_pieces(pieces_path) != bundle.pieces_udlr:
            raise ValueError('pieces_path differs from the trusted official puzzle')
        super().__init__(data_dir, max(900, interval_seconds), pieces_path=pieces_path,
                         request_timeout=request_timeout, detail_interval_seconds=max(1, detail_interval_seconds))
        self._bundle = bundle
        self._io_lock = threading.Lock()
        self._wake = threading.Event()
        self._enabled = False
        self._close_requested = False
        self._epoch = 0
        self._connection = None
        self._in_flight = 0
        self._requests = self._successes = self._failures = self._cancelled = 0
        self._indexes = self._details = 0
        self._consecutive_failures = 0
        self._next_index = 0.0
        self._last_verified = None
        self._closed_status = None
        try:
            retry = float(self._meta('download_retry_after') or 0)
            if not math.isfinite(retry):
                retry = 0
        except (ValueError, TypeError):
            retry = 0
        self._next_request = time.monotonic() + min(900, max(0, retry-time.time()))

    def _allowed(self, epoch=None):
        with self._lock:
            return (self._enabled and not self._closed and not self._close_requested
                    and (epoch is None or epoch == self._epoch))

    @staticmethod
    def _path(public_sha=None):
        if public_sha is None:
            return '/library/index', INDEX_LIMIT
        if not isinstance(public_sha, str) or not SHA_RE.fullmatch(public_sha):
            raise ValueError('A lowercase 64-hex public board SHA is required')
        return '/library/board/' + public_sha, DETAIL_LIMIT

    def _fetch(self, public_sha=None):
        path, limit = self._path(public_sha)
        with self._lock:
            if not self._allowed():
                raise _Cancelled()
            epoch = self._epoch
            self._in_flight = 1
            self._requests += 1
        deadline = time.monotonic() + self.request_timeout
        connection = None
        try:
            connection = http.client.HTTPSConnection(HOST, timeout=self.request_timeout)
            with self._lock:
                self._connection = connection
            if not self._allowed(epoch):
                raise _Cancelled()
            # HTTPSConnection uses certificate verification and direct sockets;
            # it does not interpret HTTP(S)_PROXY, cookies or redirects.
            connection.request('GET', path, body=None, headers={
                'Accept': 'application/json', 'Accept-Encoding': 'gzip',
                'User-Agent': 'EternityIISolver-library-reader/1.0',
            })
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Library request exceeded its time budget')
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError(f'Library HTTP status {response.status}; redirects are not followed')
            content_type = (response.getheader('Content-Type') or '').split(';', 1)[0].strip().lower()
            if content_type != 'application/json':
                raise ValueError('Library response must be application/json')
            length = response.getheader('Content-Length')
            if length is not None and (not length.isascii() or not length.isdecimal() or int(length) > limit):
                raise ValueError('Invalid or oversized library Content-Length')
            encoding = (response.getheader('Content-Encoding') or 'identity').strip().lower()
            if encoding not in ('identity', 'gzip'):
                raise ValueError('Unsupported library content encoding')
            inflater = zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == 'gzip' else None
            payload = bytearray()
            wire_bytes = 0
            while True:
                if not self._allowed(epoch):
                    raise _Cancelled()
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Library request exceeded its time budget')
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read(min(READ_CHUNK, limit-wire_bytes+1))
                if not chunk:
                    break
                wire_bytes += len(chunk)
                if wire_bytes > limit:
                    raise ValueError('Compressed library response exceeds size limit')
                decoded = inflater.decompress(chunk, limit-len(payload)+1) if inflater else chunk
                payload.extend(decoded)
                if len(payload) > limit or (inflater and inflater.unconsumed_tail):
                    raise ValueError('Decompressed library response exceeds size limit')
            if inflater and (not inflater.eof or inflater.unused_data):
                raise ValueError('Incomplete or concatenated gzip response')
            if length is not None and wire_bytes != int(length):
                raise ValueError('Library Content-Length mismatch')
            if not self._allowed(epoch):
                raise _Cancelled()
            return json.loads(payload.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_bad_constant)
        finally:
            try:
                if connection is not None:
                    connection.close()
            finally:
                with self._lock:
                    if self._connection is connection:
                        self._connection = None
                    self._in_flight = 0

    def _success(self, kind):
        with self._lock, self._db:
            self._successes += 1
            self._consecutive_failures = 0
            self._next_request = time.monotonic() + self.detail_interval_seconds
            self._put_meta('download_retry_after', 0)
            self._last_error = None
            if kind == 'index':
                self._indexes += 1
                self._last_verified = time.time()
                self._put_meta('index_checked_at', self._last_verified)
                self._next_index = time.monotonic() + self.interval_seconds
            else:
                self._details += 1

    def _failure(self, exc, *, public_sha=None, cancelled=False):
        with self._lock:
            if cancelled:
                self._cancelled += 1
                return
            self._failures += 1
            self._consecutive_failures += 1
            delay = min(900, BACKOFF_SECONDS * 2 ** min(5, self._consecutive_failures-1))
            self._next_request = time.monotonic() + delay
            # Avoid echoing remote body data, headers or arbitrary exception text.
            self._last_error = f'Library download failed ({type(exc).__name__}); retry in {delay} seconds'
            with self._db:
                self._put_meta('download_retry_after', time.time()+delay)
                if public_sha is not None:
                    row = self._db.execute('SELECT failures FROM boards WHERE public_sha=?', (public_sha,)).fetchone()
                    failures = max(0, int(row[0])) if row else 0
                    detail_delay = min(3600, 60 * 2 ** min(6, failures))
                    self._db.execute('UPDATE boards SET retry_after=?, failures=failures+1 WHERE public_sha=?',
                                     (time.time()+detail_delay, public_sha))

    def _poll_index(self):
        if not self._io_lock.acquire(blocking=False):
            return False
        try:
            with self._lock:
                if not self._allowed() or time.monotonic() < max(self._next_index, self._next_request):
                    return False
                requests_before = self._requests
                epoch = self._epoch
            try:
                document = self._fetch()
                parsed = self._validate_index(document)
                if not self._allowed(epoch):
                    raise _Cancelled()
                with self._lock:
                    # An immutable public identifier cannot change the score of
                    # geometry already validated and cached under that identifier.
                    known_scores = dict(self._db.execute('SELECT public_sha,score FROM boards WHERE local_sha IS NOT NULL'))
                if any(sha in known_scores and known_scores[sha] != score for sha, score, _ in parsed):
                    raise ValueError('Index changes the score of a cached board')
                self.ingest_index(document)
                self._success('index')
                return True
            except Exception as exc:
                if self._requests > requests_before:
                    self._failure(exc, cancelled=isinstance(exc, _Cancelled) or not self._allowed(epoch))
                return False
        finally:
            self._io_lock.release()
            self._close_if_idle()

    def poll_once(self):
        """Refresh the complete index if explicitly enabled and currently due."""
        self._poll_index()
        return self.status()

    def _validate_detail(self, document, expected_score):
        if not isinstance(document, dict) or type(document.get('score')) is not int:
            raise ValueError('Invalid board document')
        checked = validate_board(document.get('board'), self._bundle)
        if not checked['valid'] or checked['score'] != expected_score or document['score'] != expected_score:
            raise ValueError('Board document fails independent puzzle/score validation')
        letters, pieces = document.get('board_edges'), document.get('board_pieces')
        if not isinstance(letters, str) or len(letters) != 1024 or any(c < 'a' or c > 'w' for c in letters):
            raise ValueError('Invalid board edge encoding')
        if not isinstance(pieces, str) or len(pieces) != 768 or not pieces.isascii() or not pieces.isdecimal():
            raise ValueError('Invalid board piece encoding')
        mapping = {}
        for cell, state in enumerate(document['board']):
            if int(pieces[3*cell:3*cell+3]) != state//4+1:
                raise ValueError('Board piece encoding differs')
            for color, letter in zip(self._bundle.oriented_edges[state], letters[4*cell:4*cell+4]):
                if color in mapping and mapping[color] != letter:
                    raise ValueError('Board edge encoding differs')
                mapping[color] = letter
        if len(mapping) != 23 or len(set(mapping.values())) != 23 or mapping.get(0) != 'a':
            raise ValueError('Invalid board palette')

    def warm_once(self):
        """Download one missing available geometry, highest scores first."""
        if not self._io_lock.acquire(blocking=False):
            return False
        try:
            with self._lock:
                if not self._allowed() or time.monotonic() < self._next_request:
                    return False
                row = self._db.execute('''SELECT public_sha,score FROM boards
                    WHERE active=1 AND has_content=1 AND local_sha IS NULL AND retry_after<=?
                    ORDER BY score DESC,public_sha LIMIT 1''', (time.time(),)).fetchone()
                if row is None:
                    return False
                public_sha, score = row
                requests_before = self._requests
                epoch = self._epoch
            try:
                document = self._fetch(public_sha)
                self._validate_detail(document, score)
                if not self._allowed(epoch):
                    raise _Cancelled()
                self.register_known_document(document, public_sha)
                self._success('detail')
                return True
            except Exception as exc:
                if self._requests > requests_before:
                    self._failure(exc, public_sha=public_sha,
                                  cancelled=isinstance(exc, _Cancelled) or not self._allowed(epoch))
                return False
        finally:
            self._io_lock.release()
            self._close_if_idle()

    def _start_thread(self):
        self._thread = threading.Thread(target=self._loop, name='public-library-reader', daemon=True)
        self._thread.start()

    def start(self):
        with self._lock:
            if self._closed or self._close_requested:
                raise RuntimeError('Library downloader is closed')
            if not self._enabled:
                self._enabled = True
                self._next_index = 0.0
            self._wake.set()
            if self._thread is None or not self._thread.is_alive():
                self._start_thread()

    def _loop(self):
        try:
            while True:
                self._wake.clear()
                if not self._allowed():
                    break
                try:
                    self._poll_index()
                    if self._allowed():
                        self.warm_once()
                except Exception as exc:
                    # A damaged/unwritable local database must not create an
                    # uncontrolled thread-restart or request loop.
                    with self._lock:
                        self._last_error = f'Library cache failure ({type(exc).__name__}); downloads stopped'
                        self._enabled = False
                        self._epoch += 1
                    break
                self._wake.wait(1.0)
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
                    # start() may arrive while this thread is finishing stop().
                    if self._enabled and not self._close_requested and not self._closed:
                        self._start_thread()
            self._close_if_idle()

    def stop(self, timeout=25):
        with self._lock:
            self._enabled = False
            self._epoch += 1
            self._wake.set()
            connection, thread = self._connection, self._thread
        if connection is not None:
            # Only cancel our own socket. DNS/OS connect calls may still finish
            # later; SQLite remains open until that operation has unwound.
            try:
                if connection.sock is not None:
                    connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if thread is not None and thread is not threading.current_thread() and timeout > 0:
            thread.join(timeout=timeout)
        idle = self._io_lock.acquire(blocking=False)
        if idle:
            self._io_lock.release()
        return idle and (thread is None or not thread.is_alive())

    def _close_if_idle(self):
        if not self._close_requested or not self._io_lock.acquire(blocking=False):
            return False
        try:
            with self._lock:
                if not self._closed:
                    self._closed_status = self.status()
                    self._closed_status.update(running=False, network_enabled=False, mode='offline')
                    super().close()
                return True
        finally:
            self._io_lock.release()

    def close(self, timeout=25):
        with self._lock:
            self._close_requested = True
            self._enabled = False
        self.stop(timeout)
        return self._close_if_idle()

    def status(self):
        with self._lock:
            if self._closed:
                return dict(self._closed_status)
            result = super().status()
            fresh = self._last_verified is not None and time.time()-self._last_verified < self.interval_seconds
            result.update(network_enabled=self._enabled, mode='read-only-downloads' if self._enabled else 'offline',
                          running=self._thread is not None and self._thread.is_alive(),
                          automatic_polling_enabled=self._enabled, uploads_enabled=False,
                          progress_reporting_enabled=False, requests_in_flight=self._in_flight,
                          network_requests_this_session=self._requests,
                          successful_requests_this_session=self._successes,
                          failed_requests_this_session=self._failures,
                          cancelled_requests_this_session=self._cancelled,
                          index_downloads_this_session=self._indexes, detail_downloads_this_session=self._details,
                          online_freshness_verified=bool(fresh), snapshot_stale=not fresh if result['full_metadata_loaded'] else None,
                          next_request_in_seconds=max(0, self._next_request-time.monotonic()))
            return result
