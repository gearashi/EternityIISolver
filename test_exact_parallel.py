"""Tiny offline correctness and persistence tests for disjoint DFS workers."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import exact_dfs
import exact_parallel
from exact_model import check_solution, generated_problem
from test_exact_dfs import cyclic_unsat


def _cancel_worker(problem, seconds, seed, checkpoint, options):
    # A bounded synthetic worker makes cancellation deterministic even on a
    # machine that could finish a tiny real search before the first poll.
    import time
    deadline = time.monotonic() + min(seconds, 1)
    while time.monotonic() < deadline and not exact_parallel._CHILD_STOP.is_set():
        time.sleep(.005)
    return exact_parallel._run_ticket(problem, 0, seed, checkpoint, options)

class ParallelDfsTests(unittest.TestCase):
    def test_partitions_cover_each_tiny_solution_once(self):
        problem, answer = generated_problem(2, 2, 23, clue_count=1)
        options = exact_dfs.normalize_options()
        queue = exact_parallel._new_queue(problem, 12, options)
        tasks = [exact_parallel._ticket_problem(problem, queue['partition'], t['state']) for t in queue['tickets']]
        self.assertGreaterEqual(len(tasks), 2)
        self.assertEqual(len({t['id'] for t in queue['tickets']}), len(tasks))
        self.assertEqual(sum(check_solution(task, answer) for task in tasks), 1)
        count = 0
        for board in itertools.product(*problem.domains):
            if check_solution(problem, board):
                count += 1
                self.assertEqual(sum(check_solution(task, board) for task in tasks), 1)
        self.assertGreater(count, 0)
        for task in tasks:
            self.assertEqual(task.fixed, problem.fixed)
            self.assertEqual(task.edges, problem.edges)
            self.assertEqual(task.faces, problem.faces)

    def test_spawned_workers_find_valid_tiny_solution(self):
        problem, _ = generated_problem(2, 3, 7, clue_count=1)
        progress = []
        result = exact_parallel.solve(problem, seconds=3, workers=2, seed=4, on_progress=progress.append)
        self.assertEqual(result['outcome'], 'solved')
        self.assertTrue(check_solution(problem, result['board']))
        self.assertTrue(result['solution_verified'])
        self.assertIsNone(result['queue_path'])
        self.assertTrue(progress)
        self.assertEqual(progress[-1]['outcome'], 'solved')
        for item in progress:
            self.assertLessEqual(len(item['active_ticket_ids']), 2)
            self.assertEqual(len(item['active_ticket_ids']), len(set(item['active_ticket_ids'])))

    def test_fresh_and_restored_exhaustion_have_different_scope(self):
        problem = cyclic_unsat()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'queue.json'
            first = exact_parallel.solve(problem, seconds=3, workers=2, queue_path=path)
            self.assertEqual(first['outcome'], 'infeasible')
            self.assertTrue(first['coverage_verified'])
            self.assertEqual(first['tickets_completed'], first['tickets_total'])
            before = path.read_bytes()
            with patch.object(exact_parallel, 'ProcessPoolExecutor', side_effect=AssertionError('Completed tickets must not be relaunched')):
                second = exact_parallel.solve(problem, seconds=0, workers=2, queue_path=path)
            self.assertEqual(second['outcome'], 'infeasible')
            self.assertFalse(second['coverage_verified'])
            self.assertEqual(second['proof_scope'], 'remaining_checkpoint_frontier')
            self.assertEqual(path.read_bytes(), before)

    def test_zero_budget_persists_and_resumes_without_lost_tickets(self):
        problem, _ = generated_problem(2, 2, 11)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'nested' / 'queue.json'
            first = exact_parallel.solve(problem, seconds=0, workers=1, queue_path=path)
            self.assertEqual(first['outcome'], 'timeout')
            queue = json.loads(path.read_text())
            self.assertEqual(len(queue['tickets']), first['tickets_total'])
            self.assertTrue(all(t['status'] == 'pending' for t in queue['tickets']))
            second = exact_parallel.solve(problem, seconds=3, workers=2, queue_path=path)
            self.assertEqual(second['outcome'], 'solved')
            self.assertTrue(check_solution(problem, second['board']))
            self.assertTrue(second['resumed'])
            self.assertEqual(second['proof_scope'], 'remaining_checkpoint_frontier')

    def test_structurally_valid_worker_checkpoint_resumes(self):
        problem, _ = generated_problem(2, 2, 18)
        options = exact_dfs.normalize_options({'gac': True})
        queue = exact_parallel._new_queue(problem, 5, options)
        ticket = queue['tickets'][0]
        task = exact_parallel._ticket_problem(problem, queue['partition'], ticket['state'])
        partial = exact_parallel._run_ticket(task, 0, exact_parallel._ticket_seed(5, 0), None, options)
        self.assertEqual(partial['outcome'], 'timeout')
        ticket['checkpoint'] = partial['checkpoint']
        ticket['slices'] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'queue.json'
            exact_parallel._save(path, queue)
            result = exact_parallel.solve(problem, seconds=3, seed=5, workers=2, options={'gac': True}, queue_path=path)
            self.assertEqual(result['outcome'], 'solved')
            self.assertTrue(result['checkpoint_slices_resumed'])
            self.assertFalse(result['coverage_verified'])
            self.assertTrue(check_solution(problem, result['board']))

    def test_stop_before_spawn_keeps_all_tickets(self):
        problem, _ = generated_problem(2, 2, 8)
        with patch.object(exact_parallel, 'ProcessPoolExecutor', side_effect=AssertionError('Must not spawn after stop')):
            result = exact_parallel.solve(problem, seconds=3, workers=2, should_stop=lambda: True)
        self.assertEqual(result['outcome'], 'stopped')
        self.assertEqual(result['tickets_completed'], 0)
        self.assertTrue(all(t['status'] == 'pending' for t in result['queue']['tickets']))

    def test_child_memory_stop_is_not_infeasibility(self):
        problem, _ = generated_problem(2, 2, 8)
        with patch('search_resources.peak_memory_mb', return_value=513):
            result = exact_parallel._run_ticket(problem, 1, 0, None, exact_dfs.normalize_options())
        self.assertEqual(result['outcome'], 'stopped')
        self.assertEqual(result['resource_interruption'], 'memory_limit')
        self.assertFalse(result['coverage_verified'])
        self.assertIsNotNone(result['checkpoint'])

    def test_queue_corruption_and_changed_options_preserve_file(self):
        problem, _ = generated_problem(2, 2, 8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'queue.json'
            exact_parallel.solve(problem, seconds=0, queue_path=path)
            original = path.read_bytes()
            for overrides in ({'seed': 2}, {'options': {'gac': True}}):
                with self.assertRaises(ValueError):
                    exact_parallel.solve(problem, seconds=0, queue_path=path, **overrides)
                self.assertEqual(path.read_bytes(), original)
            value = json.loads(original)
            value['tickets'].pop()
            path.write_text(json.dumps(exact_parallel._seal(value)))
            changed = path.read_bytes()
            with self.assertRaises(ValueError):
                exact_parallel.solve(problem, seconds=0, queue_path=path)
            self.assertEqual(path.read_bytes(), changed)

    def test_invalid_checkpoint_rejected_before_spawn(self):
        problem, _ = generated_problem(2, 2, 8)
        queue = exact_parallel._new_queue(problem, 0, exact_dfs.normalize_options())
        queue['tickets'][0]['checkpoint'] = {'version': 2}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'queue.json'
            exact_parallel._save(path, queue)
            with patch.object(exact_parallel, 'ProcessPoolExecutor', side_effect=AssertionError('Invalid checkpoint must not spawn')):
                with self.assertRaises(ValueError):
                    exact_parallel.solve(problem, seconds=1, queue_path=path)

    def test_queue_lock_prevents_concurrent_ownership(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'queue.json'
            with exact_parallel._queue_lock(path):
                with self.assertRaises(RuntimeError):
                    with exact_parallel._queue_lock(path):
                        pass
            with exact_parallel._queue_lock(path):
                pass

    def test_atomic_save_failure_preserves_previous_queue(self):
        problem, _ = generated_problem(2, 2, 9)
        queue = exact_parallel._new_queue(problem, 0, exact_dfs.normalize_options())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'queue.json'
            exact_parallel._save(path, queue)
            original = path.read_bytes()
            with patch.object(exact_parallel.os, 'replace', side_effect=OSError('synthetic failure')):
                with self.assertRaises(OSError):
                    exact_parallel._save(path, queue)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_invalid_resources_rejected(self):
        problem, _ = generated_problem(2, 2, 9)
        for args in ({'workers': 0}, {'workers': 5}, {'workers': True},
                     {'seconds': -1}, {'seconds': float('nan')}, {'seconds': True}, {'seed': True}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                exact_parallel.solve(problem, **args)

    def test_cancel_active_workers_preserves_checkpoints(self):
        problem, _ = generated_problem(2, 2, 19)
        checks = 0
        def stop():
            nonlocal checks
            checks += 1
            return checks >= 3
        with patch.object(exact_parallel, '_run_ticket', _cancel_worker):
            result = exact_parallel.solve(problem, seconds=3, workers=2, should_stop=stop)
        self.assertEqual(result['outcome'], 'stopped')
        self.assertFalse(result['coverage_verified'])
        self.assertEqual(result['active_ticket_ids'], [])
        tickets = result['queue']['tickets']
        self.assertTrue(any(t['checkpoint'] is not None for t in tickets))
        self.assertTrue(all(t['status'] == 'pending' for t in tickets))
        self.assertEqual(result['branches'], sum(t['counters']['branches'] for t in tickets))

    def test_invalid_callbacks_do_not_create_queue_or_lock(self):
        problem, _ = generated_problem(2, 2, 9)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'queue.json'
            for kwargs in ({'should_stop': False}, {'on_progress': []}, {'workers': 5}):
                with self.assertRaises(ValueError):
                    exact_parallel.solve(problem, queue_path=path, **kwargs)
                self.assertEqual(list(path.parent.iterdir()), [])
    def test_saved_solution_revalidated_without_workers(self):
        problem, answer = generated_problem(2, 2, 9)
        queue = exact_parallel._new_queue(problem, 0, exact_dfs.normalize_options())
        ticket = next(t for t in queue['tickets'] if t['state'] == answer[queue['partition']['cell']])
        ticket.update(status='solved', board=answer)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'queue.json'
            exact_parallel._save(path, queue)
            with patch.object(exact_parallel, 'ProcessPoolExecutor', side_effect=AssertionError('Saved solution needs no worker')):
                result = exact_parallel.solve(problem, seconds=0, queue_path=path)
            self.assertEqual(result['outcome'], 'solved')
            self.assertTrue(result['solution_verified'])
            ticket['board'] = [0] * 4
            exact_parallel._save(path, queue)
            with self.assertRaises(ValueError):
                exact_parallel.solve(problem, seconds=0, queue_path=path)


if __name__ == '__main__':
    unittest.main()

