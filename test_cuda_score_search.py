"""Offline coverage for the CUDA score worker, board publication, and fresh-state lifecycle."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import cuda_score_search
import dashboard_server
import launcher
from manual_export import export_best
from validator import load_bundle, validate_board


class CudaScoreWorkerTests(unittest.TestCase):
    def test_bounded_launch_stop_and_score_ladder_with_mock_device(self):
        import numpy as np
        import time
        from exact_model import generated_problem
        from fast_search import score_board
        problem, solved = generated_problem(4, 3, 19)
        board = list(solved)
        board[5] ^= 1
        score = score_board(problem, board)
        self.assertLess(score, 24)
        profile = {'breaks': (0, 0, 0, 0)}
        candidate = MagicMock()
        candidate.board.__getitem__.return_value.get.return_value = np.array(board)
        candidate.broken.__getitem__.return_value.get.return_value = 24-score
        candidate.step.return_value = (np.array([2]), 15)
        candidate.nodes.sum.return_value.get.return_value = 99
        candidate.deepest.max.return_value.get.return_value = 16
        runtime = Mock()
        runtime.cuda.runtime.getDeviceProperties.return_value = {'name': b'test GPU'}
        with patch('gpu_backends.open_cuda', return_value=runtime), \
             patch.object(cuda_score_search, 'Search', return_value=candidate):
            publish = Mock()
            progress = Mock()
            result = cuda_score_search.run_ladder(problem, profile, 27, 1, time.perf_counter(), 60,
                Path('.'), publish, Mock(side_effect=[False, True]), progress)
        self.assertEqual([call.args[0] for call in candidate.reset.call_args_list], [4, 24-score-1])
        self.assertEqual(candidate.step.call_count, 1)
        self.assertEqual(publish.call_args.args[0]['board'], board)
        self.assertEqual((result['outcome'], result['branches'], result['complete_boards']),
                         ('stopped', 99, 1))

    def test_launcher_publishes_candidates_and_new_run_starts_fresh(self):
        bundle = load_bundle()
        boards = []
        for cell in (68, 100):
            board = list(bundle.record_board)
            board[cell] ^= 1
            checked = validate_board(board, bundle)
            self.assertTrue(checked['valid'])
            self.assertLess(checked['score'], 466)
            boards.append({'board': board, 'score': checked['score']})

        def ladder(problem, profile, seed, lanes, started, seconds, cache_dir, publish,
                   should_stop, on_progress):
            self.assertEqual((problem.size, lanes), (16, 2048))
            self.assertFalse(should_stop())
            if seed == 1:
                for candidate in boards:
                    publish(candidate)
            on_progress({'phase': 'searching', 'branches': 123, 'gpu': 'fixture GPU'})
            return {'outcome': 'timeout', 'branches': 123}

        with tempfile.TemporaryDirectory() as tmp, patch.object(cuda_score_search, 'run_ladder', ladder):
            home = Path(tmp)
            for seed in (1, 2):
                args = ['exact', '--engine', 'cuda-fast-score', '--seed', str(seed), '--seconds', '30',
                        '--state-dir', str(home)]
                self.assertEqual(launcher.main(args), 0)
                with dashboard_server.DashboardServer(home, 0) as server:
                    status = server.status()
                    self.assertFalse(status['resumed'])
                    self.assertEqual(status['gpu_lanes'], 2048)
                    self.assertEqual(status['gpu'], 'fixture GPU')
                    self.assertEqual(status['branches'], 123)
                    self.assertEqual(status['best_score'], 466)
                    self.assertEqual(status['library']['known_exact_boards'], len(boards))
                    for item in boards:
                        self.assertTrue(server.library.is_known(item['board']))
                    expected = max(item['score'] for item in boards) if seed == 1 else 466
                    self.assertEqual(validate_board(server.board()['board'], bundle)['score'], expected)
                    exported = json.loads(Path(export_best(home)['json_path']).read_text())
                    self.assertEqual(exported['score'], expected)
                    self.assertEqual(exported['source_kind'], 'run_best' if seed == 1 else 'runtime_best')

    def test_stop_and_expired_budget_do_not_initialize_cuda(self):
        no_launch = Mock(side_effect=AssertionError('unexpected GPU access'))
        with patch.object(cuda_score_search, 'run_ladder', no_launch):
            for seconds, stop, expected in ((60, True, 'stopped'), (0, False, 'timeout')):
                result = cuda_score_search.solve(None, seconds, 2, 1, lambda:stop, None, None, None)
                self.assertEqual(result['outcome'], expected)
        no_launch.assert_not_called()

    def test_dashboard_and_launcher_cuda_settings(self):
        settings = {'method': 'exact', 'exact_engine': 'cuda-fast-score', 'backend': 'cuda'}
        self.assertEqual(dashboard_server.validate_settings(settings)['workers'], 1)
        args = launcher.parser().parse_args(['run', '--method', 'exact', '--exact-engine', 'cuda-fast-score'])
        self.assertIn('cuda-fast-score', launcher.run_arguments(args))
        for change in ({'workers': 4}, {'backend': 'opencl'}):
            with self.assertRaises(ValueError):
                dashboard_server.validate_settings(dict(settings, **change))
