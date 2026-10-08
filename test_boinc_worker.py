"""Bounded worker contract tests using an injected engine; no GPU or network."""
import itertools
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import boinc_worker as worker
from validator import load_bundle, validate_board


class Array:
    def __init__(self, value):
        self.value = np.asarray(value).copy()
    def get(self):
        return self.value.copy()
    def __getitem__(self, key):
        return Array(self.value[key])


class FakeEngine:
    def __init__(self, bundle, replicas, seed, backend, cache_dir, corrupt=False):
        self.bundle = bundle; self.n = replicas; self.backend = 'cuda' if backend == 'auto' else backend
        self.device = 'injected test engine: no GPU'; self.calls = []; self.corrupt = corrupt
    def initialize(self, seeds):
        self.boards = Array(np.tile(np.asarray(seeds[0], np.int16), (self.n, 1)).T)
        self.bestboards = Array(self.boards.get())
        score = validate_board(seeds[0], self.bundle)['score']
        self.scores = Array(np.full(self.n, score, np.int32))
        self.bestscores = Array(np.full(self.n, score, np.int32))
        self.counters = Array(np.zeros(2 * self.n, np.uint64))
    def cool(self, elapsed):
        pass
    def step(self, count):
        self.calls.append(count)
        self.counters.value[:self.n] += count
        if self.corrupt:
            self.bestscores.value[0] = 480
        return count * 10.0
    def verify_device_scores(self, indices):
        for index in indices:
            report = validate_board(self.boards.value[:, index], self.bundle)
            assert report['valid'] and report['score'] == int(self.scores.value[index])
        return len(indices)
    def checkpoint(self, path):
        np.savez_compressed(path, **{key:getattr(self,key).get() for key in ('boards','bestboards','scores','bestscores','counters')})
    def resume(self, path):
        with np.load(path, allow_pickle=False) as archive:
            for key in ('boards','bestboards','scores','bestscores','counters'):
                setattr(self, key, Array(archive[key]))
        return True


class Factory:
    def __init__(self, corrupt=False):
        self.instances = []; self.corrupt = corrupt
    def __call__(self, *args, **kwargs):
        instance = FakeEngine(*args, **kwargs, corrupt=self.corrupt)
        self.instances.append(instance)
        return instance


class WorkunitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='eternity-worker-test-')
        self.directory = Path(self.temporary.name)
        self.input = self.directory / 'workunit.json'
        self.output = self.directory / 'result.json'
        self.checkpoint = self.directory / 'checkpoint.npz'
        self.progress = self.directory / 'fraction_done.txt'
        self.document = {'schema':worker.WORKUNIT_SCHEMA,'workunit_id':'test-001',
                         'pieces_sha256':self.bundle.metadata['input_sha256']['pieces.txt'],
                         'replicas':32,'seed':20261008,'steps_per_replica':9}
        self.write()
        self.factory = Factory()

    def tearDown(self):
        self.temporary.cleanup()

    def write(self):
        self.input.write_text(json.dumps(self.document, indent=2), encoding='utf-8')

    def run_worker(self, **kwargs):
        options = {'standalone_diagnostic':True,'engine_factory':self.factory}
        options.update(kwargs)
        return worker.run_workunit(self.input,self.output,self.checkpoint,self.progress,**options)

    def test_zero_step_validates_without_gpu(self):
        self.document['steps_per_replica'] = 0; self.write()
        code, result = self.run_worker()
        self.assertEqual(code, 0); self.assertEqual(self.factory.instances, [])
        self.assertEqual(result['score'], 466); self.assertFalse(result['complete'])
        self.assertFalse(result['exhausted']); self.assertTrue(result['budget_complete'])
        self.assertEqual(float(self.progress.read_text()), 1.0)
        self.assertEqual(result['input_sha256'], worker.canonical_input_hash(self.document))

    def test_exact_step_budget_and_embedded_checkpoint(self):
        code, result = self.run_worker()
        self.assertEqual(code, 0)
        self.assertEqual(self.factory.instances[0].calls, [1,2,4,2])
        self.assertEqual(result['completed_steps_per_replica'], 9)
        self.assertEqual(result['proposed_moves'], 9 * 32)
        self.assertEqual(result['mode'], 'five-clue')
        with np.load(self.checkpoint, allow_pickle=False) as archive:
            metadata = worker.strict_json(archive[worker.METADATA_KEY].item())
        self.assertEqual(metadata['input_sha256'], result['input_sha256'])
        self.assertEqual(metadata['completed_steps_per_replica'], 9)
        self.assertEqual(list(self.directory.glob('*.tmp*')), [])

    def test_interrupt_resume_does_not_repeat_budget(self):
        stop = lambda: bool(self.factory.instances and sum(self.factory.instances[-1].calls) >= 3)
        code, partial = self.run_worker(stop_requested=stop)
        self.assertEqual(code, 75); self.assertEqual(partial['status'], 'interrupted')
        self.assertEqual(partial['completed_steps_per_replica'], 3)
        self.assertFalse(partial['budget_complete'])
        first_active = partial['elapsed_seconds']
        code, result = self.run_worker()
        self.assertEqual(code, 0); self.assertTrue(result['resumed'])
        self.assertEqual(sum(self.factory.instances[-1].calls), 6)
        self.assertEqual(result['completed_steps_per_replica'], 9)
        self.assertEqual(result['proposed_moves'], 9 * 32)
        self.assertAlmostEqual(result['elapsed_seconds'], first_active + .06)
        code, repeated = self.run_worker()
        self.assertEqual(code, 0); self.assertEqual(self.factory.instances[-1].calls, [])
        self.assertEqual(repeated['proposed_moves'], result['proposed_moves'])

    def test_suspension_wall_clock_does_not_consume_steps(self):
        with patch('boinc_worker.time.monotonic', side_effect=itertools.count(0,3600).__next__):
            code, result = self.run_worker()
        self.assertEqual(code, 0)
        self.assertEqual(result['completed_steps_per_replica'], 9)
        self.assertAlmostEqual(result['elapsed_seconds'], .09)

    def test_foreign_checkpoint_rejected_before_gpu_or_output_change(self):
        self.run_worker(); original = self.output.read_bytes(); count = len(self.factory.instances)
        self.document['workunit_id'] = 'another-unit'; self.write()
        with self.assertRaisesRegex(worker.WorkunitError, 'different workunit'):
            self.run_worker()
        self.assertEqual(len(self.factory.instances), count)
        self.assertEqual(self.output.read_bytes(), original)

    def test_checkpoint_bound_to_all_input_fields(self):
        self.run_worker(); self.document['seed'] += 1; self.write()
        with self.assertRaisesRegex(worker.WorkunitError, 'different workunit'):
            self.run_worker()

    def test_unbound_checkpoint_rejected(self):
        np.savez_compressed(self.checkpoint, boards=np.zeros((256,32),np.int16))
        with self.assertRaisesRegex(worker.WorkunitError, 'not bound'):
            self.run_worker()
        self.assertEqual(self.factory.instances, [])

    def test_overbudget_metadata_rejected(self):
        self.run_worker()
        with np.load(self.checkpoint, allow_pickle=False) as archive:
            arrays = {key:archive[key].copy() for key in archive.files}
        metadata = json.loads(arrays[worker.METADATA_KEY].item())
        metadata['completed_steps_per_replica'] = 10
        arrays[worker.METADATA_KEY] = np.asarray(json.dumps(metadata))
        np.savez_compressed(self.checkpoint, **arrays)
        with self.assertRaisesRegex(worker.WorkunitError, 'checkpoint completed steps'):
            self.run_worker()

    def test_input_size_limit_matches_server(self):
        self.document['steps_per_replica'] = 0
        text = json.dumps(self.document)
        self.input.write_bytes((text + ' ' * (worker.MAX_JSON_BYTES - len(text))).encode('utf-8'))
        code, _ = self.run_worker()
        self.assertEqual(code, 0)
        original = self.output.read_bytes()
        with self.input.open('ab') as handle:
            handle.write(b' ')
        with self.assertRaisesRegex(worker.WorkunitError, 'exceeds 131072 bytes'):
            self.run_worker()
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(self.factory.instances, [])

    def test_non_npz_checkpoint_is_invalid_input(self):
        with self.checkpoint.open('wb') as handle:
            np.save(handle, np.asarray([1, 2, 3]))
        with self.assertRaisesRegex(worker.WorkunitError, 'must be an NPZ archive'):
            self.run_worker()
        with patch('sys.stderr', new_callable=io.StringIO) as stderr:
            code = worker.main(['--input', str(self.input), '--output', str(self.output),
                                '--checkpoint', str(self.checkpoint), '--progress', str(self.progress),
                                '--standalone-diagnostic'], engine_factory=self.factory)
        self.assertEqual(code, 2)
        self.assertIn('Invalid workunit or checkpoint', stderr.getvalue())
        self.assertEqual(self.factory.instances, [])
        self.assertFalse(self.output.exists())

    def test_bad_checkpoint_archive_is_invalid_input(self):
        for data in (b'', b'PK\x03\x04truncated zip', b'not a numpy archive'):
            with self.subTest(data=data):
                self.checkpoint.write_bytes(data)
                with self.assertRaisesRegex(worker.WorkunitError, 'Malformed checkpoint archive'):
                    self.run_worker()
        self.assertEqual(self.factory.instances, [])
        self.assertFalse(self.output.exists())

    def test_duplicate_keys_and_nonfinite_json_rejected(self):
        for text in ('{"schema":1,"schema":2}', '{"seconds":NaN}'):
            with self.subTest(text=text), self.assertRaises(worker.WorkunitError):
                worker.strict_json(text)

    def test_strict_schema_rejects_unknown_or_bad_values(self):
        original = self.document.copy()
        for changed in ({'instructions':'fetch a website'}, {'seconds':30}, {'replicas':True},
                        {'steps_per_replica':-1}, {'steps_per_replica':10**9+1}, {'seed':2**32},
                        {'pieces_sha256':'0'*64}, {'encoding':'mirror-pieces'}):
            with self.subTest(changed=changed):
                self.document = {**original, **changed}; self.write()
                with self.assertRaises(worker.WorkunitError):
                    self.run_worker()
        self.assertEqual(self.factory.instances, [])

    def test_illegal_seed_is_rejected(self):
        self.document['board'] = list(self.bundle.record_board)
        self.document['board'][10] = self.document['board'][11]
        self.write()
        with self.assertRaisesRegex(worker.WorkunitError, 'starting board is illegal'):
            self.run_worker()

    def test_scheduled_mode_needs_explicit_cuda_allocation(self):
        for options in ({}, {'backend':'cuda'}, {'backend':'opencl','cuda_device':0}):
            with self.subTest(options=options), self.assertRaises(worker.WorkunitError):
                self.run_worker(standalone_diagnostic=False, **options)
        selected = []
        code, result = self.run_worker(standalone_diagnostic=False, backend='cuda', cuda_device=3,
                                       device_binder=selected.append)
        self.assertEqual(code, 0); self.assertEqual(selected, [3])
        self.assertEqual(result['allocation']['cuda_ordinal'], 3)
        self.assertEqual(result['allocation']['mode'], 'boinc-wrapper')

    def test_backend_conflict_is_rejected(self):
        self.document['backend'] = 'opencl'; self.write()
        with self.assertRaisesRegex(worker.WorkunitError, 'conflicts'):
            self.run_worker(backend='cuda')

    def test_false_gpu_score_is_not_returned(self):
        self.factory = Factory(corrupt=True)
        with self.assertRaisesRegex(RuntimeError, 'independent CPU validation'):
            self.run_worker()
        self.assertFalse(self.output.exists())

    def test_input_output_alias_rejected(self):
        with self.assertRaisesRegex(worker.WorkunitError, 'must be different'):
            worker.run_workunit(self.input,self.input,self.checkpoint,self.progress,
                                standalone_diagnostic=True,engine_factory=self.factory)

    def test_formatting_does_not_change_binding(self):
        self.run_worker()
        self.input.write_text(json.dumps(self.document,sort_keys=True,separators=(',',':')),encoding='utf-8')
        code, result = self.run_worker()
        self.assertEqual(code, 0); self.assertTrue(result['resumed'])


if __name__ == '__main__':
    unittest.main()
