"""Local HTTP controls with a fake downloader: no external requests or search."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from dashboard_server import create_server

class FakeDownloader:
    def __init__(self,*args,**kwargs):
        self.enabled=False; self.starts=0; self.stops=0; self.closed=False
    def start(self):
        self.enabled=True; self.starts+=1
    def stop(self,timeout=25):
        self.enabled=False; self.stops+=1
        return True
    def close(self):
        self.closed=True; self.enabled=False
    def status(self):
        return {'mode':'read-only-downloads' if self.enabled else 'offline',
                'network_enabled':self.enabled,'running':self.enabled,
                'indexed_boards':2,'cached_geometries':1,
                'network_requests_this_session':0,'uploads_enabled':False}

class LibraryDashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='eternity-library-ui-')
        self.home=Path(self.tmp.name)
        self.created=[]; self.spawned=[]
        def factory(*args,**kwargs):
            item=FakeDownloader(*args,**kwargs); self.created.append(item); return item
        self.factory=factory
        def worker(*args,**kwargs):
            self.spawned.append(args)
            raise AssertionError('Search must not start')
        self.server=create_server(self.home,0,library_factory=factory,worker_factory=worker)
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
        self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.port}'
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=3)
        self.tmp.cleanup()
    def request(self,body,origin=None):
        req=urllib.request.Request(self.base+'/library-downloads',data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json','Origin':origin or self.base})
        try:
            with self.opener.open(req,timeout=3) as response:return response.status,json.load(response)
        except urllib.error.HTTPError as exc:return exc.code,json.load(exc)
    def test_toggle_is_independent_and_never_starts_search(self):
        runtime=self.home/'runtime'
        for name in ('STOP','best.json','checkpoint.npz'):(runtime/name).write_bytes(b'unchanged')
        self.assertEqual(self.created,[])
        code,status=self.request({'enabled':True})
        self.assertEqual(code,200)
        self.assertTrue(status['library_downloads_enabled'])
        self.assertTrue(status['external_network_enabled'])
        self.assertFalse(status['solver_network_enabled'])
        self.assertFalse(status['uploads_enabled'])
        self.assertFalse(status['progress_reporting_enabled'])
        self.assertTrue(status['can_start'])
        self.assertFalse(status['can_stop'])
        # Stop/save for the puzzle has no effect on the downloader.
        self.server.stop_worker()
        self.assertTrue(self.created[0].enabled)
        code,status=self.request({'enabled':False})
        self.assertEqual(code,200); self.assertFalse(status['external_network_enabled'])
        self.assertFalse(self.created[0].enabled)
        self.assertEqual(json.loads((runtime/'library-downloads.json').read_text()),{'enabled':False})
        for name in ('best.json','checkpoint.npz'):self.assertEqual((runtime/name).read_bytes(),b'unchanged')
        self.assertTrue((runtime/'STOP').exists())
        self.assertEqual(self.spawned,[])
    def test_controls_reject_extra_data_and_cross_origin(self):
        for bad in ({},{'enabled':1},{'enabled':True,'nodes':1000},{'url':'https://example.invalid'}):
            self.assertEqual(self.request(bad)[0],400)
        self.assertEqual(self.request({'enabled':True},'https://foreign.invalid')[0],403)
        self.assertEqual(self.created,[])
    def test_restart_respects_persisted_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory)
            with create_server(home,0,library_factory=self.factory) as first:
                first.set_library_downloads({'enabled':True})
            self.assertTrue(self.created[-1].closed)
            with create_server(home,0,library_factory=self.factory) as second:
                self.assertTrue(second.status()['library_downloads_enabled'])
                self.assertTrue(self.created[-1].enabled)
                second.set_library_downloads({'enabled':False})
            count=len(self.created)
            with create_server(home,0,library_factory=self.factory) as third:
                self.assertFalse(third.status()['library_downloads_enabled'])
            self.assertEqual(len(self.created),count)
    def test_cache_failure_does_not_disable_search_controls(self):
        def fail(*args,**kwargs):raise OSError('Fixture cache unavailable')
        self.server.library_factory=fail
        code,status=self.request({'enabled':True})
        self.assertEqual(code,200)
        self.assertTrue(status['can_start']);self.assertFalse(status['external_network_enabled'])
        self.assertIn('Fixture cache unavailable',status['library']['last_error'])
        self.assertEqual(self.spawned,[])

if __name__=='__main__':unittest.main()
