"""Hybrid lifecycle with injected sampler; no GPU, network or canonical state."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import exact_worker
from validator import load_bundle

class HybridWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bundle=load_bundle()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='eternity-hybrid-worker-')
        self.addCleanup(self.temp.cleanup)
        self.home=Path(self.temp.name);self.runtime=self.home/'runtime';self.runtime.mkdir()
        self.calls=0
    def sampler(self,*args,**kwargs):
        self.calls+=1
        return {'boards':[list(self.bundle.record_board)],'scores':[466],
                'generated_boards':4096,'boards_per_second':12345,'elapsed_seconds':.33,
                'backend':'fixture','device':'fixture','compile_seconds':.01}
    def solver(self,problem,seed,checkpoint=None,hints=None,**kwargs):
        self.assertEqual(hints,[list(self.bundle.record_board)])
        return {'outcome':'timeout','board':None,'checkpoint':{'problem_hash':exact_worker.problem_hash(problem),'seed':seed}}
    def run_worker(self,**kwargs):
        with patch.object(exact_worker,'peak_memory_mb',return_value=64):
            return exact_worker.run_task(self.home,engine='hybrid',solver_factory=lambda _:self.solver,
                                        sampler_factory=self.sampler,**kwargs)
    def status(self):return json.loads((self.runtime/'status.json').read_text())
    def test_hints_frozen_and_checkpoints_separate(self):
        gpu=self.runtime/'checkpoint.npz';gpu.write_bytes(b'preserved GPU fixture')
        dfs=self.runtime/'exact-dfs-checkpoint.json';dfs.write_text('preserved plain DFS fixture')
        self.assertEqual(self.run_worker(),0)
        snapshot=(self.runtime/'hybrid-hints.json').read_bytes()
        self.assertEqual(self.run_worker(),0)
        self.assertEqual(self.calls,1);self.assertTrue(self.status()['resumed'])
        self.assertEqual((self.runtime/'hybrid-hints.json').read_bytes(),snapshot)
        self.assertEqual(gpu.read_bytes(),b'preserved GPU fixture')
        self.assertEqual(dfs.read_text(),'preserved plain DFS fixture')
        self.assertEqual(self.status()['hybrid_hint_count'],1)
        self.assertEqual(self.status()['best_score'],466)
        self.assertFalse(self.status()['external_network_enabled'])
    def test_corrupt_hints_rejected_then_explicit_fresh_archives(self):
        self.assertEqual(self.run_worker(),0)
        path=self.runtime/'hybrid-hints.json';value=json.loads(path.read_text());value['scores']=[0];path.write_text(json.dumps(value))
        changed=path.read_bytes();self.assertEqual(self.run_worker(),1)
        self.assertEqual(path.read_bytes(),changed);self.assertEqual(self.calls,1)
        self.assertEqual(self.run_worker(fresh_exact=True),0)
        self.assertEqual(self.calls,2);self.assertTrue(list(self.runtime.glob('hybrid-hints.archived-*.json')))
    def test_saved_checkpoint_without_hints_cannot_resample(self):
        (self.runtime/'exact-hybrid-dfs-checkpoint.json').write_text('{}')
        self.assertEqual(self.run_worker(),1);self.assertEqual(self.calls,0)
    def test_stop_during_sampling_preserves_stop_without_frontier(self):
        original=self.sampler
        def stopping(*args,**kwargs):
            result=original(*args,**kwargs)
            (self.runtime/'STOP').write_text('Stop requested')
            return result
        self.sampler=stopping
        self.assertEqual(self.run_worker(),0)
        self.assertEqual(self.status()['state'],'stopped')
        self.assertFalse((self.runtime/'hybrid-hints.json').exists())
    def test_preserved_stop_skips_sampler(self):
        (self.runtime/'STOP').write_text('Stop requested')
        self.assertEqual(self.run_worker(preserve_stop=True),0);self.assertEqual(self.calls,0)
if __name__=='__main__':unittest.main()
