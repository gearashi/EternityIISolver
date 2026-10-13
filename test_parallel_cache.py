"""Bounded per-run queue setup cache equivalence and input-isolation tests."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import exact_dfs
import exact_parallel as parallel
from exact_model import ExactProblem, generated_problem, problem_hash
from test_parallel_splitting import split_fixture


def mutable_problem():
    problem, board = generated_problem(3, 4, 71)
    return ExactProblem(problem.size, [list(face) for face in problem.faces],
                        dict(problem.fixed), [list(domain) for domain in problem.domains]), board


class QueueSetupCacheTests(unittest.TestCase):
    def test_cached_tasks_equal_uncached_for_every_decision_kind(self):
        problem, _ = generated_problem(3, 4, 9)
        cache = parallel._TaskCache(problem)
        root = cache.partition['states'][0]
        paths = ((), (('cell', 0, problem.domains[0][0]),),
                 (('piece', 0, 3),), (('edge', 0, 1),),
                 (('cell', 0, problem.domains[0][0]), ('edge', 1, 2)))
        for path in paths:
            expected = parallel._ticket_problem(problem, cache.partition, root, path)
            got = cache.task(root, path)
            self.assertEqual(got, expected)
            self.assertEqual(cache.identity(root, path), problem_hash(expected))
            self.assertIs(cache.task(root, [list(decision) for decision in path]), got)
        task = cache.task(root)
        for cell, domain in enumerate(task.domains):
            if cell != cache.partition['cell']:
                self.assertIs(domain, cache.base_domains[cell])

    def test_snapshot_detaches_all_mutable_problem_containers(self):
        problem, _ = mutable_problem()
        expected = parallel._snapshot_problem(problem)
        cache = parallel._TaskCache(problem)
        state = cache.partition['states'][0]
        problem.faces[0][1] = 255
        problem.domains[0].clear()
        problem.fixed[0] = 0
        self.assertEqual(cache.problem, expected)
        self.assertEqual(cache.task(state), parallel._ticket_problem(expected, cache.partition, state))
        self.assertEqual(cache.root_identity, problem_hash(expected))

    def test_solve_snapshot_precedes_user_callbacks(self):
        problem, _ = mutable_problem()
        identity = problem_hash(problem)
        def mutate_and_stop():
            problem.fixed[0] = 10000
            problem.domains[0].clear()
            problem.faces[0][0] = 255
            return True
        with patch.object(parallel, 'ProcessPoolExecutor', side_effect=AssertionError('No worker after stop')):
            answer = parallel.solve(problem, seconds=2, should_stop=mutate_and_stop)
        self.assertEqual(answer['outcome'], 'stopped')
        self.assertEqual(answer['problem_hash'], identity)
        self.assertEqual(answer['queue']['problem_hash'], identity)

    def test_repeated_work_uses_one_domain_build_and_one_hash_per_task(self):
        problem, _ = generated_problem(3, 4, 1)
        original_domains, original_hash = parallel.initial_domains, parallel.problem_hash
        with patch.object(parallel, 'initial_domains', wraps=original_domains) as domains, \
             patch.object(parallel, 'problem_hash', wraps=original_hash) as hashes:
            cache = parallel._TaskCache(problem)
            state = cache.partition['states'][0]
            first = cache.task(state)
            for _ in range(8):
                self.assertIs(cache.task(state), first)
                self.assertEqual(cache.identity(state), original_hash(first))
            self.assertEqual(domains.call_count, 1)
            self.assertEqual(hashes.call_count, 2)  # Root identity + one restricted identity.
            self.assertEqual(cache.task_builds, 1)
            self.assertEqual(cache.hash_builds, 1)

    def test_lru_caps_evict_only_recomputable_setup(self):
        problem, _ = generated_problem(3, 4, 5)
        with patch.object(parallel, 'MAX_CACHED_TASKS', 2), patch.object(parallel, 'MAX_TICKETS', 3):
            cache = parallel._TaskCache(problem)
            state = cache.partition['states'][0]
            remembered = []
            for color in range(1, 7):
                path = (('edge', 0, color),)
                remembered.append((path, cache.identity(state, path)))
                self.assertLessEqual(len(cache.tasks), 2)
                self.assertLessEqual(len(cache.identities), 3)
            self.assertGreater(cache.task_builds, len(cache.tasks))
            for path, identity in remembered:
                self.assertEqual(cache.identity(state, path), identity)

    def test_queue_validation_and_splitting_are_identical_with_cache(self):
        problem, queue, _ = split_fixture()
        cache = parallel._TaskCache(problem)
        expected = parallel._new_queue(problem, 7, queue['options'])
        cached_expected = parallel._new_queue(cache.problem, 7, queue['options'], task_cache=cache)
        self.assertEqual(cached_expected, expected)
        cached_queue = json.loads(json.dumps(queue))
        first = parallel._try_split(queue, 0, problem)
        second = parallel._try_split(cached_queue, 0, cache.problem, task_cache=cache)
        self.assertEqual(first, second)
        self.assertEqual(queue, cached_queue)
        checked = parallel._validate_queue(parallel._seal(queue), expected, problem)
        cached_checked = parallel._validate_queue(parallel._seal(cached_queue), cached_expected,
                                                   cache.problem, task_cache=cache)
        self.assertEqual(checked, cached_checked)
        self.assertGreater(cache.task_hits, 0)

    def test_cached_queue_does_not_skip_semantic_corruption_validation(self):
        from test_exact_dfs import reseal
        problem, queue, _ = split_fixture()
        cache = parallel._TaskCache(problem)
        parallel._semantic_checkpoint(queue, 0, cache.problem, task_cache=cache)
        queue['tickets'][0]['checkpoint']['stack'][0]['candidates'].pop()
        queue['tickets'][0]['checkpoint'] = reseal(queue['tickets'][0]['checkpoint'])
        with self.assertRaisesRegex(ValueError, 'does not partition'):
            parallel._try_split(queue, 0, cache.problem, task_cache=cache)

    def test_foreign_cache_and_invalid_mutable_paths_are_rejected(self):
        problem, _ = generated_problem(3, 4, 1)
        cache = parallel._TaskCache(problem)
        state = cache.partition['states'][0]
        with self.assertRaises(ValueError):
            parallel._resolve_task(problem, cache.partition, state, (), cache)
        for path in ([('cell', True, 0)], [('piece', 0, False)], [[[], 0, 0]], [('edge', 999, 0)]):
            with self.assertRaises(ValueError):
                cache.task(state, path)


if __name__ == '__main__':
    unittest.main()
