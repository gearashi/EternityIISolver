"""Local HTTP controls with a fake downloader: no external requests or search."""
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
    def render_status(self, status):
        node=shutil.which('node')
        if node is None:self.skipTest('Node.js is needed for this dashboard JavaScript regression')
        script=r'''
const fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync(process.argv[1],'utf8');
const source=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],{style:{},value:'',checked:false,hidden:false,disabled:false,textContent:'',addEventListener(){},replaceChildren(){},appendChild(){},setAttribute(){}}]));
for(const [id,value] of [['methodChoice','gpu'],['exactEngine','dfs'],['workerCount','1'],['searchCount','4096'],['backendChoice','auto']])elements.get(id).value=value;
const context=vm.createContext({document:{getElementById:id=>elements.get(id),querySelectorAll:()=>[]},localStorage:{getItem:()=>null},setInterval(){},fixture:JSON.parse(process.argv[2])});
vm.runInContext(source.replace('refresh();setInterval(refresh,3000);',''),context);
vm.runInContext('renderStatus(fixture,true);',context);
const result={};for(const id of ['footerMode','libraryPolicy','legacyWarning','throughput','proposed','accepted','exactPhase'])result[id]=elements.get(id).textContent;
console.log(JSON.stringify(result));
'''
        result=subprocess.run([node,'-e',script,str(self.server.resources/'Dashboard.html'),json.dumps(status)],
                              capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)
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
    def test_legacy_worker_warning_survives_separate_downloader_status(self):
        (self.home/'runtime'/'status.json').write_text(json.dumps({
            'state':'running','pid':os.getpid(),'library':{'network_enabled':True},
        }))
        for enabled in (True,False):
            with self.subTest(enabled=enabled):
                code,status=self.request({'enabled':enabled})
                self.assertEqual(code,200)
                self.assertEqual(status['library']['mode'],'read-only-downloads' if enabled else 'offline')
                self.assertIn('older worker',status['legacy_worker_warning'])
                self.assertIsNone(status['uploads_enabled'])
                self.assertIsNone(status['solver_network_enabled'])
                rendered=self.render_status(status)
                self.assertIn('older worker unverified',rendered['footerMode'])
                self.assertIn('Older worker networking remains unverified',rendered['libraryPolicy'])
                self.assertNotIn('uploads disabled',rendered['footerMode'])
        self.assertEqual(self.spawned,[])
    def test_starting_exact_worker_cannot_inherit_previous_run_counts(self):
        class StartingWorker:
            pid=12345
            def poll(self):return None
        path=self.home/'runtime'/'status.json'
        prior={'state':'stopped','pid':os.getpid(),'method':'exact','exact_engine':'cp-sat',
               'branches':10395682,'conflicts':117793,'max_depth':88,'exact_counters_available':True,
               'exact_phase':'stopped','exact_outcome':'infeasible','exact_conclusion':'Previous conclusion'}
        path.write_text(json.dumps(prior))
        self.server.worker=StartingWorker()
        self.server.worker_status_before=self.server.status_revision()
        for engine in ('sat','cp-sat','dfs','hybrid'):
            with self.subTest(engine=engine):
                self.server.settings.update(method='exact',exact_engine=engine)
                status=self.server.status()
                self.assertEqual(status['state'],'starting')
                native=engine in ('sat','cp-sat')
                self.assertEqual(status['exact_counters_available'],not native)
                self.assertEqual(status['branches'],None if native else 0)
                self.assertEqual(status['conflicts'],None if native else 0)
                self.assertEqual(status['exact_phase'],'starting')
                for key in ('max_depth','exact_outcome','exact_conclusion'):self.assertNotIn(key,status)
                rendered=self.render_status(status)
                self.assertEqual(rendered['proposed'],'Not yet reported' if native else '0')
                self.assertEqual(rendered['accepted'],'Not yet reported' if native else '0')
                self.assertEqual(rendered['exactPhase'],'Starting')
        self.assertEqual(json.loads(path.read_text()),prior,'The previous saved worker report is read-only')
        self.assertEqual(self.spawned,[])

if __name__=='__main__':unittest.main()
