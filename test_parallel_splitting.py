"""Finite frontier-coverage and ownership tests for dynamic DFS splitting."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import exact_dfs
import exact_parallel
from exact_model import check_solution
from test_dfs_improvements import all_solutions, corner_problem, dual_problem


def interrupted(problem, seed=0, options=None, branches=1):
    stop = [False]
    def observe(snapshot):
        stop[0] = snapshot['counters']['branches'] >= branches
    result = exact_dfs.solve(problem, seconds=2, seed=seed, options=options,
                             should_stop=lambda: stop[0], on_checkpoint=observe, checkpoint_interval=0)
    if result['outcome'] != 'stopped':
        raise AssertionError('Fixture did not yield its frontier')
    return result


def split_fixture():
    problem = corner_problem(((1, 1),) * 4, edges=())
    options = exact_dfs.normalize_options()
    queue = exact_parallel._new_queue(problem, 7, options)
    ticket = queue['tickets'][0]
    task = exact_parallel._ticket_problem(problem, queue['partition'], ticket['state'])
    result = interrupted(task, exact_parallel._ticket_seed(7, 0))
    ticket['checkpoint'] = result['checkpoint']
    ticket['slices'] = 1
    ticket['counters'] = {key: result[key] for key in ticket['counters']}
    return problem, queue, task


class DynamicSplitTests(unittest.TestCase):
    def test_frontier_cell_piece_and_edge_paths_partition_all_tiny_solutions(self):
        cases = [(corner_problem(((1, 1),) * 4), {}, 'cell'),
                 (dual_problem(), {'dual_branch': True}, 'piece'),
                 (corner_problem(), {'edge_branch': True}, 'edge')]
        for problem, options, kind in cases:
            result = interrupted(problem, seed=17, options=options)
            cp = result['checkpoint']
            self.assertEqual(cp['stack'][0]['kind'], kind)
            paths = exact_parallel._frontier_paths(cp)
            tasks = [exact_parallel._ticket_problem(problem, {'cell': None}, None, path) for path in paths]
            self.assertGreater(len(tasks), 1)
            for board in all_solutions(problem):
                self.assertEqual(sum(check_solution(task, board) for task in tasks), 1)
            for task in tasks:
                for board in all_solutions(task):
                    self.assertTrue(check_solution(problem, board))

    def test_split_tree_validates_and_preserves_counters_and_complete_coverage(self):
        problem, queue, task = split_fixture()
        original_counts = dict(queue['tickets'][0]['counters'])
        indices = exact_parallel._try_split(queue, 0, problem)
        self.assertGreater(len(indices), 1)
        self.assertEqual(queue['tickets'][0]['status'], 'split')
        self.assertEqual(queue['tickets'][0]['counters'], original_counts)
        children = [queue['tickets'][i] for i in indices]
        self.assertTrue(all(not any(t['counters'].values()) for t in children))
        tasks = [exact_parallel._ticket_problem(problem, queue['partition'], t['state'], t['path']) for t in children]
        for board in all_solutions(task):
            self.assertEqual(sum(check_solution(child, board) for child in tasks), 1)
        expected = exact_parallel._new_queue(problem, 7, queue['options'])
        checked = exact_parallel._validate_queue(exact_parallel._seal(queue), expected, problem)
        self.assertEqual(checked, queue)
        self.assertEqual(exact_parallel._try_split(queue, 0, problem), [])  # Parent cannot be scheduled or split twice.

    def test_missing_duplicate_broadened_and_reparented_children_rejected(self):
        problem, queue, _ = split_fixture()
        children = exact_parallel._try_split(queue, 0, problem)
        expected = exact_parallel._new_queue(problem, 7, queue['options'])
        for mode in ('missing', 'duplicate', 'broadened', 'reparented', 'counter', 'root-owned'):
            changed = json.loads(json.dumps(queue))
            if mode == 'missing':
                changed['tickets'][0]['children'].pop()
            elif mode == 'duplicate':
                changed['tickets'][0]['children'][1] = children[0]
            elif mode == 'broadened':
                changed['tickets'][children[0]]['path'] = []
            elif mode == 'reparented':
                changed['tickets'][children[0]]['parent'] = 1
            elif mode == 'root-owned':
                changed['tickets'][0]['children'][0] = 1
            else:
                changed['tickets'][0]['counters']['branches'] += 1
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                exact_parallel._validate_queue(exact_parallel._seal(changed), expected, problem)

    def test_semantically_incomplete_checkpoint_cannot_be_split_or_loaded(self):
        from test_exact_dfs import reseal
        problem, queue, _ = split_fixture()
        cp = queue['tickets'][0]['checkpoint']
        cp['stack'][0]['candidates'].pop()
        queue['tickets'][0]['checkpoint'] = reseal(cp)
        original = json.dumps(queue, sort_keys=True)
        with self.assertRaisesRegex(ValueError, 'does not partition'):
            exact_parallel._try_split(queue, 0, problem)
        self.assertEqual(json.dumps(queue, sort_keys=True), original)
        # Manufacture a structurally matching split tree to ensure loading
        # replays its basis too; valid checksum/child paths are insufficient.
        with patch.object(exact_parallel, '_semantic_checkpoint'):
            children = exact_parallel._try_split(queue, 0, problem)
        self.assertTrue(children)
        with self.assertRaisesRegex(ValueError, 'does not partition'):
            exact_parallel._validate_queue(exact_parallel._seal(queue),
                exact_parallel._new_queue(problem, 7, queue['options']), problem)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'queue.json'
            exact_parallel._save(path, queue)
            original = path.read_bytes()
            with patch.object(exact_parallel, 'ProcessPoolExecutor', side_effect=AssertionError('Do not spawn')):
                with self.assertRaisesRegex(ValueError, 'does not partition'):
                    exact_parallel.solve(problem, seconds=3, seed=7, queue_path=path)
            self.assertEqual(path.read_bytes(), original)

    def test_split_validation_budget_or_cancel_preserves_parent_and_file(self):
        problem, queue, _ = split_fixture()
        original = json.dumps(queue, sort_keys=True)
        for kwargs, outcome in (({'seconds': 0}, 'timeout'),
                                ({'should_stop': lambda: True}, 'stopped')):
            with self.assertRaises(exact_dfs._Interrupted) as caught:
                exact_parallel._try_split(queue, 0, problem, **kwargs)
            self.assertEqual(caught.exception.outcome, outcome)
            self.assertEqual(json.dumps(queue, sort_keys=True), original)
        exact_parallel._try_split(queue, 0, problem)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'queue.json'
            exact_parallel._save(path, queue)
            before = path.read_bytes()
            with patch.object(exact_parallel, 'ProcessPoolExecutor', side_effect=AssertionError('Do not spawn')):
                result = exact_parallel.solve(problem, seconds=0, seed=7, queue_path=path)
            self.assertEqual(result['outcome'], 'timeout')
            self.assertEqual(path.read_bytes(), before)

    def test_caps_skip_split_without_losing_frontier(self):
        for cap_name, cap_value in (('MAX_TICKETS', 4), ('MAX_SPLIT_FANOUT', 1), ('MAX_QUEUE_BYTES', 1)):
            problem, queue, _ = split_fixture()
            before = json.dumps(queue, sort_keys=True)
            with patch.object(exact_parallel, cap_name, cap_value):
                self.assertEqual(exact_parallel._try_split(queue, 0, problem), [])
            self.assertEqual(json.dumps(queue, sort_keys=True), before)

    def test_idle_workers_split_and_solve_real_remaining_regions(self):
        problem, queue, task = split_fixture()
        # The recorded exhaustion of other root tickets is explicitly treated
        # as untrusted history on resume. Only the saved root frontier matters.
        for ticket in queue['tickets'][1:]:
            ticket.update(status='infeasible', coverage_verified=False)
        original_branches = queue['tickets'][0]['counters']['branches']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'queue.json'
            exact_parallel._save(path, queue)
            result = exact_parallel.solve(problem, seconds=3, workers=2, seed=7, queue_path=path)
            self.assertEqual(result['outcome'], 'solved')
            self.assertTrue(check_solution(task, result['board']))
            self.assertGreater(result['split_tickets'], 0)
            self.assertFalse(result['coverage_verified'])
            self.assertEqual(result['proof_scope'], 'remaining_checkpoint_frontier')
            saved = json.loads(path.read_text())
            self.assertEqual(saved['tickets'][0]['counters']['branches'], original_branches)
            self.assertEqual(result['branches'], sum(t['counters']['branches'] for t in saved['tickets']))
            exact_parallel._validate_queue(saved, exact_parallel._new_queue(problem, 7, queue['options']), problem)
            again = exact_parallel.solve(problem, seconds=3, workers=2, seed=7, queue_path=path)
            self.assertEqual(again['board'], result['board'])
            self.assertEqual(again['branches'], result['branches'])

    def test_stop_after_split_retains_children_for_resume(self):
        problem, queue, task = split_fixture()
        children = exact_parallel._try_split(queue, 0, problem)
        for ticket in queue['tickets'][1:4]:
            ticket.update(status='infeasible', coverage_verified=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'queue.json'
            exact_parallel._save(path, queue)
            original = path.read_bytes()
            stopped = exact_parallel.solve(problem, seconds=3, workers=2, seed=7, queue_path=path, should_stop=lambda: True)
            self.assertEqual(stopped['outcome'], 'stopped')
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(stopped['tickets_total'], len(queue['tickets']) - 1)
            self.assertEqual(sum(t['status'] == 'pending' for t in queue['tickets']), len(children))
            result = exact_parallel.solve(problem, seconds=3, workers=2, seed=7, queue_path=path)
            self.assertEqual(result['outcome'], 'solved')
            self.assertTrue(check_solution(task, result['board']))

    def test_legacy_queue_is_preserved_and_rejected(self):
        problem, queue, _ = split_fixture()
        queue['version'] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'queue.json'
            exact_parallel._save(path, queue)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                exact_parallel.solve(problem, seconds=0, seed=7, queue_path=path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
