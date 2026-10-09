"""Bounded CPU tests for sampler selection/validation; no GPU initialization."""
import unittest
from unittest.mock import patch
import numpy as np

import hybrid_sampling as sampling
from kernel_port import to_opencl
from validator import load_bundle, validate_board


class HybridSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def test_tables_allow_fresh_permutations_and_preserve_all_five_clues(self):
        base, cells, offsets, fixed, masks, faces = sampling._layout(self.bundle)
        self.assertEqual(list(np.diff(offsets)), [4, 56, 191])
        self.assertEqual(set(np.flatnonzero(fixed)), set(self.bundle.fixed_clues))
        self.assertEqual(len(set(cells)), 251)
        rng = np.random.default_rng(29)
        for _ in range(4):
            board = base.copy()
            for left, right in zip(offsets[:-1], offsets[1:]):
                positions = cells[left:right]
                board[positions] = board[rng.permutation(positions)]
            for cell in cells:
                piece = int(board[cell]) // 4
                legal = [r for r in range(4) if int(masks[cell, piece]) & (1 << r)]
                board[cell] = 4 * piece + rng.choice(legal)
            self.assertTrue(validate_board(board, self.bundle)['valid'])
            self.assertTrue(np.any(board != base))

    def test_selector_is_bounded_and_has_top_and_random_current_banks(self):
        scores = np.arange(4096, dtype=np.int32) % 100
        first = sampling._selection(scores, scores[::-1], np.random.default_rng(5))
        second = sampling._selection(scores, scores[::-1], np.random.default_rng(5))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        indices, banks, values = first
        self.assertEqual(len(indices), 64)
        self.assertEqual(list(banks), [1] * 32 + [0] * 32)
        self.assertEqual(len(set(indices[32:])), 32)
        np.testing.assert_array_equal(values[:32], scores[indices[:32]])
        self.assertTrue(np.all(values[:32] == 99))

    def test_screen_deduplicates_and_rechecks_claimed_duplicate_scores(self):
        board = list(self.bundle.record_board)
        result = sampling._screen([board, board], [466, 466], self.bundle)
        self.assertEqual(result['selected_boards'], 2)
        self.assertEqual(result['cpu_validated_boards'], 1)
        self.assertEqual(result['duplicate_candidates'], 1)
        self.assertEqual(result['boards'], [board])
        with self.assertRaisesRegex(RuntimeError, 'score'):
            sampling._screen([board, board], [466, 467], self.bundle)

    def test_screen_rejects_illegal_and_malformed_candidates(self):
        board = list(self.bundle.record_board)
        invalid = board.copy(); invalid[0] = invalid[1]
        for boards, scores in (([invalid], [466]), ([board[:-1]], [466]), ([board], [466.5])):
            with self.subTest(scores=scores), self.assertRaises(RuntimeError):
                sampling._screen(boards, scores, self.bundle)
        with self.assertRaises(ValueError):
            sampling._screen([board] * 65, [466] * 65, self.bundle)

    def test_zero_and_stopped_calls_do_not_initialize_gpu(self):
        with patch.object(sampling, '_Device', side_effect=AssertionError('GPU should not initialize')):
            self.assertEqual(sampling.sample(self.bundle, seconds=0)['generated_boards'], 0)
            self.assertEqual(sampling.sample(self.bundle, should_stop=lambda: True)['outcome'], 'stopped')

    def test_fake_device_counts_full_boards_and_includes_transfer_screen_time(self):
        clock = [100.0]
        board = list(self.bundle.record_board)
        class FakeDevice:
            backend = 'fixture'; device = 'CPU-only fake'; details = {}
            compile_seconds = 1.0; initialization_seconds = 0.5
            def __init__(self, *args): pass
            def generate(self): clock[0] += 0.01; return 10.0
            def selected(self): clock[0] += 0.01; return [board, board], [466, 466]
            def close(self): pass
        with patch.object(sampling, '_Device', FakeDevice), \
             patch.object(sampling.time, 'perf_counter', side_effect=lambda: clock[0]):
            result = sampling.sample(self.bundle, seconds=0.025, replicas=32)
        self.assertEqual(result['generated_boards'], 96)
        self.assertEqual(result['batches'], 3)
        self.assertAlmostEqual(result['elapsed_seconds'], 0.04)
        self.assertAlmostEqual(result['boards_per_second'], 2400)
        self.assertEqual(result['retained_unique_boards'], 1)
        self.assertEqual(result['duplicate_candidates'], 1)
        self.assertEqual(result['compile_seconds'], 1)

    def test_kernel_port_covers_both_kernels_and_private_rng(self):
        source = to_opencl(sampling.KERNEL_SOURCE)
        self.assertEqual(source.count('__kernel void '), 2)
        self.assertIn('__private unsigned int *state', source)
        self.assertNotIn('blockIdx', source)

    def test_generation_error_and_post_init_stop_release_device(self):
        closed = []
        stopped = [False]
        class FakeDevice:
            backend = 'fixture'; device = 'CPU-only fake'; details = {}
            compile_seconds = 0; initialization_seconds = 0
            def __init__(self, *args): pass
            def generate(self): raise RuntimeError('Fixture kernel failure')
            def close(self): closed.append(True)
        with patch.object(sampling, '_Device', FakeDevice), self.assertRaisesRegex(RuntimeError, 'Fixture kernel failure'):
            sampling.sample(self.bundle, seconds=1)
        self.assertEqual(len(closed), 1)
        class StoppedDevice(FakeDevice):
            def __init__(self, *args): stopped[0] = True
        with patch.object(sampling, '_Device', StoppedDevice):
            result = sampling.sample(self.bundle, seconds=1, should_stop=lambda: stopped[0])
        self.assertEqual(result['outcome'], 'stopped')
        self.assertEqual(result['generated_boards'], 0)
        self.assertEqual(len(closed), 2)

    def test_invalid_parameters_fail_without_gpu(self):
        with patch.object(sampling, '_Device', side_effect=AssertionError('GPU should not initialize')):
            for kwargs in ({'replicas':31}, {'replicas':32769}, {'seconds':float('inf')},
                           {'seed':-1}, {'allow_opencl_cpu':True, 'backend':'auto'}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    sampling.sample(self.bundle, **kwargs)


if __name__ == '__main__':
    unittest.main()
