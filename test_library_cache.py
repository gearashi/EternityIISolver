"""Offline behavior tests against the independently validated downloaded fixtures."""
import gzip,json,shutil,time,unittest,uuid,tempfile
from pathlib import Path
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
        self.mon.stop();self.mon._db.close()
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
        finally:again._db.close()
    def test_lazy_geometry_and_complete_tiers(self):
        self.mon.ingest_index(INDEX)
        fixtures={SHA466:DOC466,SEED.stem:DOC465}
        self.mon._request=lambda url,headers=None:(200,{'Content-Encoding':'gzip'},gzip.compress(json.dumps(fixtures[url.rsplit('/',1)[-1]]).encode()))
        self.assertTrue(self.mon.warm_once());self.assertTrue(self.mon.warm_once())
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
    def test_etag_and_bad_update_preserve_snapshot(self):
        self.mon.ingest_index(INDEX,{'ETag':'"fixture"'})
        calls=[]
        def fetch(url,headers=None):calls.append(headers);return 304,{},b''
        self.mon._request=fetch;self.mon.poll_once()
        self.assertEqual(calls[0]['If-None-Match'],'"fixture"')
        self.mon._request=lambda *a:(200,{},b'{"count":0}')
        self.mon.poll_once();self.assertEqual(self.mon.status()['indexed_boards'],2)
        self.assertIn('Unsupported',self.mon.status()['last_error'])
    def test_invalid_board_rejected(self):
        broken=DOC466['board'][:];broken[17]=broken[18]
        with self.assertRaises(ValueError):self.mon.is_known(broken)
        bad=dict(DOC466);bad['score']=480
        with self.assertRaises(ValueError):self.mon.register_known_document(bad,SHA466)
        self.assertEqual(self.mon.status()['known_exact_boards'],0)
    def test_bootstrap_and_background_stop(self):
        bootstrap=self.path/'library-index.json';bootstrap.write_text(json.dumps(INDEX))
        again=LibraryMonitor(self.path)
        fixtures={SHA466:DOC466,SEED.stem:DOC465}
        def fetch(url,headers=None):
            return (304,{},b'') if url.endswith('/index') else (200,{},json.dumps(fixtures[url.rsplit('/',1)[-1]]).encode())
        again._request=fetch
        try:
            self.assertEqual(again.status()['indexed_boards'],2)
            again.start();time.sleep(.15);self.assertTrue(again.stop())
            self.assertFalse(again.status()['running'])
        finally:again._db.close()

if __name__=='__main__':unittest.main(verbosity=2)
