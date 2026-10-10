"""Manual, single-request public archive downloads followed by an offline merge.

Constructing this manager never downloads. Each explicit start_download() asks
only for the fixed gzip URL; there is no polling, retry loop or per-board route.
"""
from __future__ import annotations
import hashlib
import http.client
from pathlib import Path
import socket
import threading
import time
import uuid

from archive_import import ArchiveCancelled, import_archive
from library_cache import LibraryMonitor
from validator import load_bundle, read_pieces

ARCHIVE_URL = 'https://stats.eternityathome.org/library/archive'
ARCHIVE_HOST = 'stats.eternityathome.org'
ARCHIVE_PATH = '/library/archive'
MAX_COMPRESSED = 1024 * 1024 * 1024
READ_CHUNK = 128 * 1024
REQUEST_TIMEOUT = 20
DOWNLOAD_DEADLINE = 600


class _DownloadCancelled(Exception):
    pass


class LibraryArchiveManager(LibraryMonitor):
    def __init__(self, data_dir, *, pieces_path=None):
        if pieces_path is None or read_pieces(pieces_path) != load_bundle().pieces_udlr:
            raise ValueError('Archive imports require the trusted canonical puzzle')
        super().__init__(data_dir, pieces_path=pieces_path)
        self._pieces_path = Path(pieces_path)
        self._cancel = threading.Event()
        self._connection = None
        self._active = False
        self._network_active = False
        self._close_requested = False
        self._closed_status = None
        self._requests = 0
        self._job = self._idle_job()

    @staticmethod
    def _idle_job():
        return {'phase': 'idle', 'running': False, 'source_url': ARCHIVE_URL,
                'bytes_downloaded': 0, 'total_bytes': None, 'validated_boards': 0,
                'total_boards': None, 'new_boards': 0, 'already_cached': 0,
                'cancel_requested': False, 'error': None}

    def start_download(self):
        with self._lock:
            if self._close_requested or self._closed:
                raise RuntimeError('Archive manager is closed')
            if self._active:
                return False
            self._cancel.clear()
            self._job = dict(self._idle_job(), phase='downloading', running=True)
            self._last_error = None
            self._active = True
            try:
                self._thread = threading.Thread(target=self._run, name='manual-library-archive', daemon=True)
                self._thread.start()
            except Exception as exc:
                self._thread = None
                self._active = False
                self._last_error = ('Could not start archive job: ' + str(exc))[:500]
                self._job.update(phase='error', running=False, error=self._last_error)
                return False
            return True

    def _interrupt_download(self):
        with self._lock:
            connection = self._connection
        if connection is not None:
            try:
                if connection.sock is not None:
                    connection.sock.shutdown(socket.SHUT_RDWR)
            except (OSError, AttributeError):
                pass
            try:
                connection.close()
            except OSError:
                pass

    def cancel(self):
        with self._lock:
            if not self._active or self._job['phase'] not in ('downloading', 'validating'):
                return False
            self._cancel.set()
            self._job['cancel_requested'] = True
        self._interrupt_download()
        return True

    def _check_cancel(self):
        if self._cancel.is_set():
            raise _DownloadCancelled()

    def _download(self, destination):
        self._check_cancel()
        connection = http.client.HTTPSConnection(ARCHIVE_HOST, timeout=REQUEST_TIMEOUT)
        deadline = time.monotonic() + DOWNLOAD_DEADLINE
        digest = hashlib.sha256()
        with self._lock:
            self._connection = connection
            self._network_active = True
        try:
            self._check_cancel()
            with self._lock:
                self._requests += 1
            # Direct TLS socket: certificate verification, no proxy/cookie/auth
            # handling and no automatic redirects. The body and query are absent.
            connection.request('GET', ARCHIVE_PATH, body=None, headers={
                'Accept': 'application/gzip, application/octet-stream',
                'Accept-Encoding': 'identity',
                'User-Agent': 'EternityIISolver-archive-reader/0.2.2',
            })
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError(f'Archive server returned HTTP {response.status}; no redirect or retry was followed')
            mime = (response.getheader('Content-Type') or '').split(';', 1)[0].strip().lower()
            if mime not in ('application/gzip', 'application/x-gzip', 'application/octet-stream'):
                raise ValueError('Expected a gzip archive response')
            if (response.getheader('Content-Encoding') or 'identity').strip().lower() != 'identity':
                raise ValueError('Unexpected HTTP compression; expected a gzip file')
            length = response.getheader('Content-Length')
            if length is not None:
                if not length.isascii() or not length.isdecimal() or not 0 < int(length) <= MAX_COMPRESSED:
                    raise ValueError('Invalid or oversized archive Content-Length')
                length = int(length)
            with self._lock:
                self._job['total_bytes'] = length
            received = 0
            with destination.open('xb') as output:
                while True:
                    self._check_cancel()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Archive download exceeded ten minutes')
                    if connection.sock is not None:
                        connection.sock.settimeout(min(REQUEST_TIMEOUT, remaining))
                    block = response.read1(min(READ_CHUNK, MAX_COMPRESSED - received + 1))
                    if not block:
                        break
                    received += len(block)
                    if received > MAX_COMPRESSED:
                        raise ValueError('Compressed archive exceeds 1 GiB')
                    output.write(block)
                    digest.update(block)
                    with self._lock:
                        self._job['bytes_downloaded'] = received
            if length is not None and received != length:
                raise ValueError('Archive Content-Length mismatch')
            if received == 0:
                raise ValueError('Archive response is empty')
            self._check_cancel()
            return digest.hexdigest()
        finally:
            connection.close()
            with self._lock:
                self._connection = None
                self._network_active = False

    def _progress(self, value):
        with self._lock:
            phase = value.get('phase')
            if phase in ('validating', 'merging'):
                self._job['phase'] = phase
            for key in ('validated_boards', 'total_boards'):
                count = value.get(key)
                if type(count) is int and count >= 0:
                    self._job[key] = count

    def _run(self):
        pending = self.data_dir / ('.archive-download-' + uuid.uuid4().hex + '.jsonl.gz.part')
        try:
            expected_digest = self._download(pending)
            self._progress({'phase': 'validating'})
            result = import_archive(pending, self.data_dir, pieces_path=self._pieces_path,
                                    should_cancel=self._cancel.is_set, on_progress=self._progress,
                                    expected_sha256=expected_digest)
            if result['archive_sha256'] != expected_digest:
                raise ValueError('Downloaded archive hash changed during import')
            with self._lock:
                self._job.update(phase='complete', validated_boards=result['validated_boards'],
                                 total_boards=result['validated_boards'], new_boards=result['new_boards'],
                                 already_cached=result['already_cached'], generated_at=result['generated_at'],
                                 archive_sha256=result['archive_sha256'], archive_path=result.get('archive_path'),
                                 completed_at=time.time(), error=None)
        except (ArchiveCancelled, _DownloadCancelled):
            with self._lock:
                self._job.update(phase='cancelled', error=None)
        except Exception as exc:
            with self._lock:
                if self._cancel.is_set():
                    self._job.update(phase='cancelled', error=None)
                else:
                    self._last_error = ('Archive update failed: ' + str(exc))[:500]
                    self._job.update(phase='error', error=self._last_error)
        finally:
            try:
                pending.unlink(missing_ok=True)
            except OSError:
                with self._lock:
                    self._job['cleanup_warning'] = 'A temporary archive could not be removed.'
            with self._lock:
                self._active = False
                self._network_active = False
                self._job['running'] = False
            if self._close_requested:
                self._close_cache()

    def status(self):
        with self._lock:
            if self._closed:
                return dict(self._closed_status)
            result = super().status()
            result.update(mode='manual-archive', running=self._active, network_enabled=self._network_active,
                          automatic_polling_enabled=False, uploads_enabled=False, progress_reporting_enabled=False,
                          network_requests_this_session=self._requests, archive=dict(self._job),
                          archive_can_start=not self._active and not self._close_requested,
                          archive_can_cancel=self._active and not self._cancel.is_set()
                              and self._job['phase'] in ('downloading', 'validating'))
            return result

    def _close_cache(self):
        with self._lock:
            if not self._closed:
                self._closed_status = self.status()
                self._closed_status.update(running=False, network_enabled=False, archive_can_start=False, archive_can_cancel=False)
                super().close()

    def close(self, timeout=25):
        with self._lock:
            self._close_requested = True
            self._cancel.set()
            thread = self._thread
        self._interrupt_download()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        if thread is None or not thread.is_alive():
            self._close_cache()
            return True
        # The worker closes the read cache only after its import has unwound.
        return False
