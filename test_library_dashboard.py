"""Manual archive controls with fake jobs: no external requests or real search."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from dashboard_server import create_server


class FakeArchiveManager:
    def __init__(self, *args, **kwargs):
        self.starts = self.cancels = 0
        self.closed = False
        self.phase = 'idle'
        self.error = None
    def start_download(self):
        if self.phase in ('downloading', 'validating', 'merging'): return False
        self.starts += 1
        self.phase = 'downloading'
        return True
    def cancel(self):
        if self.phase not in ('downloading', 'validating'): return False
        self.cancels += 1
        self.phase = 'cancelled'
        return True
    def close(self):
        self.closed = True
    def status(self):
        running = self.phase in ('downloading', 'validating', 'merging')
        return {'mode': 'manual-archive', 'network_enabled': self.phase == 'downloading',
                'indexed_boards': 114084, 'cached_geometries': 120883, 'known_exact_boards': 120883,
                'archive_can_start': not running, 'archive_can_cancel': self.phase in ('downloading', 'validating'),
                'archive': {'phase': self.phase, 'running': running, 'bytes_downloaded': 2 * 1048576,
                            'total_bytes': None, 'validated_boards': 75, 'total_boards': 100,
                            'new_boards': 12, 'already_cached': 88, 'error': self.error,
                            'source_url': 'https://stats.eternityathome.org/library/archive',
                            'generated_at': '2026-10-09T23:45:22Z'}}


class LibraryDashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='eternity-library-ui-')
        self.home = Path(self.tmp.name)
        self.created, self.spawned = [], []
        def factory(*args, **kwargs):
            item = FakeArchiveManager(*args, **kwargs)
            self.created.append(item)
            return item
        self.factory = factory
        def worker(*args, **kwargs):
            self.spawned.append(args)
            raise AssertionError('Search must not start')
        self.server = create_server(self.home, 0, library_factory=factory, worker_factory=worker)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.port}'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=3)
        self.tmp.cleanup()
    def request(self, body, path='/library-archive', origin=None, headers=None):
        options = {'Content-Type': 'application/json', 'Origin': origin or self.base}
        options.update(headers or {})
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=options)
        try:
            with self.opener.open(req, timeout=3) as response: return response.status, json.load(response)
        except urllib.error.HTTPError as exc: return exc.code, json.load(exc)
    def render_status(self, status, *, live=True, event=None):
        node = shutil.which('node')
        if node is None: self.skipTest('Node.js is needed for dashboard JavaScript regression checks')
        script = r'''
const fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync(process.argv[1],'utf8'),source=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],{style:{},value:'',checked:false,hidden:false,disabled:false,textContent:'',listeners:{},addEventListener(kind,fn){this.listeners[kind]=fn;},replaceChildren(){},appendChild(){},setAttribute(){}}]));
for(const [id,value] of [['methodChoice','gpu'],['exactEngine','dfs'],['workerCount','1'],['searchCount','4096'],['backendChoice','auto']])elements.get(id).value=value;
const fixture=JSON.parse(process.argv[2]),calls=[];let resolveFetch;
const context=vm.createContext({document:{getElementById:id=>elements.get(id),querySelectorAll:()=>[]},localStorage:{getItem:()=>null},setInterval(){},AbortSignal:{timeout:()=>undefined},fixture,fetch:(url,options)=>{calls.push({url,...options});return new Promise(resolve=>{resolveFetch=resolve;});}});
vm.runInContext(source.replace('refresh();setInterval(refresh,3000);',''),context);
vm.runInContext('connected=fixture.live;renderStatus(fixture.status,fixture.live);',context);
function snapshot(){const result={};for(const [id,node] of elements)result[id]={text:node.textContent,disabled:node.disabled,hidden:node.hidden,width:node.style.width};return result;}
(async()=>{const result={rendered:snapshot()};if(fixture.event){
 vm.runInContext('refresh=async()=>{};',context);
 const pending=elements.get(fixture.event.cancel?'archiveCancel':'archiveStart').listeners.click();
 result.pending=snapshot();resolveFetch({ok:fixture.event.ok,json:async()=>fixture.event.result});await pending;
 result.after=snapshot();result.calls=calls;
}console.log(JSON.stringify(result));})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        result = subprocess.run([node, '-e', script, str(self.server.resources / 'Dashboard.html'),
                                 json.dumps({'status': status, 'live': live, 'event': event})],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        rendered = json.loads(result.stdout)
        return rendered if event else rendered['rendered']

    def test_boot_shows_cache_without_starting_archive_or_search(self):
        self.assertEqual(len(self.created), 1); self.assertEqual(self.created[0].starts, 0)
        status = self.server.status()
        self.assertEqual(status['library']['known_exact_boards'], 120883)
        self.assertTrue(status['archive_can_start']); self.assertFalse(status['archive_can_cancel'])
        self.assertFalse(status['external_network_enabled']); self.assertNotIn('library_downloads_enabled', status)
        self.assertEqual(self.spawned, [])
    def test_manual_job_duplicate_click_cancel_and_search_stop_are_independent(self):
        runtime = self.home / 'runtime'
        for name in ('STOP', 'best.json', 'checkpoint.npz'): (runtime / name).write_bytes(b'unchanged')
        code, status = self.request({})
        self.assertEqual(code, 202); self.assertTrue(status['archive_request_accepted'])
        self.assertEqual((runtime / 'STOP').read_bytes(), b'unchanged')
        self.assertTrue(status['external_network_enabled']); self.assertTrue(status['can_start'])
        for key in ('solver_network_enabled', 'uploads_enabled', 'progress_reporting_enabled', 'can_stop'): self.assertFalse(status[key])
        self.assertEqual(self.request({})[0], 409); self.assertEqual(self.created[0].starts, 1)
        self.server.stop_worker(); self.assertEqual(self.created[0].phase, 'downloading')
        code, status = self.request({}, '/library-archive/cancel')
        self.assertEqual(code, 202); self.assertFalse(status['external_network_enabled'])
        self.assertEqual(self.request({}, '/library-archive/cancel')[0], 409)
        self.assertFalse((runtime / 'library-downloads.json').exists())
        for name in ('best.json', 'checkpoint.npz'): self.assertEqual((runtime / name).read_bytes(), b'unchanged')
        self.assertTrue((runtime / 'STOP').exists()); self.assertEqual(self.spawned, [])
    def test_actions_reject_extra_data_cross_origin_host_and_old_endpoint(self):
        for endpoint in ('/library-archive', '/library-archive/cancel'):
            for bad in ({'enabled': True}, {'nodes': 1000}, {'url': 'https://example.invalid'}, []):
                self.assertEqual(self.request(bad, endpoint)[0], 400)
            self.assertEqual(self.request({}, endpoint, 'https://foreign.invalid')[0], 403)
            self.assertEqual(self.request({}, endpoint, headers={'Host': 'foreign.invalid'})[0], 403)
            self.assertEqual(self.request({}, endpoint, headers={'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.request({'padding': 'x' * 4096})[0], 413)
        self.assertEqual(self.request({'enabled': True}, '/library-downloads')[0], 404)
        self.assertEqual(self.created[0].starts, 0); self.assertEqual(self.created[0].cancels, 0)
    def test_restart_ignores_old_opt_in_and_does_not_resume_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory); runtime = home / 'runtime'; runtime.mkdir()
            old_setting = runtime / 'library-downloads.json'; old_setting.write_text('{"enabled": true}')
            for attempt in range(2):
                with create_server(home, 0, library_factory=self.factory) as server:
                    manager = self.created[-1]; self.assertEqual(manager.starts, 0)
                    self.assertFalse(server.status()['external_network_enabled'])
                    self.assertTrue(server.status()['archive_can_start'])
                    if attempt == 0: self.assertTrue(server.request_library_archive({})[1])
                self.assertTrue(manager.closed)
            self.assertEqual(old_setting.read_text(), '{"enabled": true}')
    def test_cache_initialization_failure_does_not_disable_search(self):
        def fail(*args, **kwargs): raise OSError('Fixture cache unavailable')
        with tempfile.TemporaryDirectory() as directory:
            with create_server(directory, 0, library_factory=fail) as server:
                status, accepted = server.request_library_archive({})
                self.assertFalse(accepted); self.assertTrue(status['can_start'])
                self.assertFalse(status['external_network_enabled']); self.assertFalse(status['archive_can_start'])
                self.assertIn('Fixture cache unavailable', status['library']['archive']['error'])
        self.assertEqual(self.spawned, [])
    def test_merging_is_offline_and_cannot_cancel(self):
        manager = self.created[0]; manager.phase = 'merging'; status = self.server.status()
        self.assertFalse(status['external_network_enabled'])
        self.assertFalse(status['archive_can_start']); self.assertFalse(status['archive_can_cancel'])
        self.assertEqual(self.request({}, '/library-archive/cancel')[0], 409); self.assertEqual(manager.cancels, 0)
    def test_legacy_worker_warning_survives_separate_archive_status(self):
        (self.home / 'runtime' / 'status.json').write_text(json.dumps({
            'state': 'running', 'pid': os.getpid(), 'library': {'network_enabled': True}}))
        for phase in ('downloading', 'validating', 'complete'):
            with self.subTest(phase=phase):
                self.created[0].phase = phase; status = self.server.status()
                self.assertIn('older worker', status['legacy_worker_warning'])
                self.assertIsNone(status['uploads_enabled']); self.assertIsNone(status['solver_network_enabled'])
                rendered = self.render_status(status)
                self.assertIn('older worker unverified', rendered['footerMode']['text'])
                self.assertIn('Older worker networking remains unverified', rendered['libraryPolicy']['text'])
                self.assertNotIn('uploads disabled', rendered['footerMode']['text'])
        self.assertEqual(self.spawned, [])
    def test_ui_phase_progress_counts_errors_and_disconnect(self):
        manager = self.created[0]
        for phase in ('idle', 'downloading', 'validating', 'merging', 'complete', 'cancelled', 'error'):
            with self.subTest(phase=phase):
                manager.phase = phase; manager.error = '<script>fixture failure</script>' if phase == 'error' else None
                status = self.server.status(); rendered = self.render_status(status)
                self.assertEqual(rendered['known']['text'], '120,883'); self.assertEqual(rendered['indexed']['text'], '114,084')
                self.assertEqual(rendered['archiveStart']['disabled'], not status['archive_can_start'])
                self.assertEqual(rendered['archiveCancel']['disabled'], not status['archive_can_cancel'])
                self.assertFalse(rendered['start']['disabled']); self.assertTrue(rendered['stop']['disabled'])
                self.assertNotIn('of 114,084', rendered['cacheText']['text'])
                if phase == 'downloading':
                    self.assertIn('2 MiB', rendered['archiveStatus']['text']); self.assertIn('unknown', rendered['archiveStatus']['text'])
                    self.assertTrue(rendered['archiveProgress']['hidden'])
                if phase == 'validating':
                    self.assertIn('75 of 100', rendered['archiveStatus']['text']); self.assertEqual(rendered['archiveBar']['width'], '75%')
                if phase == 'complete':
                    self.assertIn('12 new board records cached', rendered['archiveSummary']['text']); self.assertIn('88 records already cached', rendered['archiveSummary']['text'])
                if phase == 'error': self.assertIn(manager.error, rendered['archiveStatus']['text'])
                disconnected = self.render_status(status, live=False)
                self.assertTrue(disconnected['archiveStart']['disabled']); self.assertTrue(disconnected['archiveCancel']['disabled'])
    def test_ui_click_pending_and_failure_do_not_touch_search_controls(self):
        for cancel in (False, True):
            self.created[0].phase = 'downloading' if cancel else 'idle'; before = self.server.status()
            accepted, _ = self.server.request_library_archive({}, cancel=cancel)
            rendered = self.render_status(before, event={'cancel': cancel, 'ok': True, 'result': accepted})
            self.assertTrue(rendered['pending']['archiveStart']['disabled']); self.assertTrue(rendered['pending']['archiveCancel']['disabled'])
            self.assertFalse(rendered['pending']['start']['disabled']); self.assertEqual(len(rendered['calls']), 1)
            call = rendered['calls'][0]
            self.assertEqual(call['url'], '/library-archive/cancel' if cancel else '/library-archive')
            self.assertEqual(call['method'], 'POST'); self.assertEqual(call['body'], '{}')
            self.assertFalse(rendered['after']['start']['disabled'])
        self.created[0].phase = 'idle'
        rendered = self.render_status(self.server.status(), event={'ok': False, 'result': {'error': 'Fixture refused'}})
        self.assertFalse(rendered['after']['archiveStart']['disabled'])
        self.assertIn('Fixture refused', rendered['after']['archiveMessage']['text'])
    def test_starting_exact_worker_cannot_inherit_previous_run_counts(self):
        class StartingWorker:
            pid = 12345
            def poll(self): return None
        path = self.home / 'runtime' / 'status.json'
        prior = {'state': 'stopped', 'pid': os.getpid(), 'method': 'exact', 'exact_engine': 'cp-sat',
                 'branches': 10395682, 'conflicts': 117793, 'max_depth': 88, 'exact_counters_available': True,
                 'exact_phase': 'stopped', 'exact_outcome': 'infeasible', 'exact_conclusion': 'Previous conclusion'}
        path.write_text(json.dumps(prior)); self.server.worker = StartingWorker()
        self.server.worker_status_before = self.server.status_revision()
        for engine in ('sat', 'cp-sat', 'dfs', 'hybrid'):
            with self.subTest(engine=engine):
                self.server.settings.update(method='exact', exact_engine=engine); status = self.server.status()
                self.assertEqual(status['state'], 'starting'); native = engine in ('sat', 'cp-sat')
                self.assertEqual(status['exact_counters_available'], not native)
                self.assertEqual(status['branches'], None if native else 0); self.assertEqual(status['conflicts'], None if native else 0)
                self.assertEqual(status['exact_phase'], 'starting')
                for key in ('max_depth', 'exact_outcome', 'exact_conclusion'): self.assertNotIn(key, status)
                rendered = self.render_status(status)
                self.assertEqual(rendered['proposed']['text'], 'Not yet reported' if native else '0')
                self.assertEqual(rendered['accepted']['text'], 'Not yet reported' if native else '0')
                self.assertEqual(rendered['exactPhase']['text'], 'Starting')
                self.assertFalse(rendered['archiveStart']['disabled'])
        self.assertEqual(json.loads(path.read_text()), prior, 'Previous worker report is read-only')
        self.assertEqual(self.spawned, [])


if __name__ == '__main__': unittest.main()
