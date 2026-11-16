"""Regression coverage for the independent-completion edge watch."""
import unittest
from unittest.mock import patch

from exact_dfs import _Search, solve
from exact_model import check_solution, generated_problem, make_problem


def partial_problem(edges):
    face = (0, 1, 1, 0)
    rotations = tuple(tuple(face[(s-r) % 4] for s in range(4)) for r in range(4))
    return make_problem(2, rotations * 4, edges=edges)


def at_node(problem, uncertain):
    return [sum(1 << state for state in domain
                if state // 4 == cell or cell in uncertain and state // 4 == (cell + 1) % 4)
            for cell, domain in enumerate(problem.domains)]


class IndependentEdgeWatchTests(unittest.TestCase):
    def test_watch_is_rechecked_across_siblings_and_partial_edges(self):
        p = partial_problem(((0, 1, 1, 3), (2, 3, 1, 3)))
        s = _Search(p, 4, 7, None, None, None, None, 5)
        s.build()
        matching = list(range(4))
        with patch.object(s, 'valid_matching', wraps=s.valid_matching) as valid:
            self.assertIsNone(s.independent_completion(at_node(p, {0, 1}), matching))
            self.assertIsNone(s.independent_completion(at_node(p, {0, 1}), [-1] * 4))
            self.assertEqual(valid.call_count, 1)
            self.assertIsNone(s.independent_completion(at_node(p, {2, 3}), matching))
            self.assertIsNone(s.independent_completion(at_node(p, {1, 2}), [-1] * 4))
            answer = s.independent_completion(at_node(p, {1, 2}), matching)
            self.assertEqual(valid.call_count, 4)
        self.assertTrue(check_solution(p, answer))
        self.assertEqual((s.independent_checks, s.independent_completions), (5, 1))
        q = partial_problem(())
        t = _Search(q, 4, 7, None, None, None, None, 5)
        t.build()
        self.assertTrue(check_solution(q, t.independent_completion(at_node(q, range(4)), matching)))

    def test_restore_bypasses_the_watch_and_keeps_the_existing_checkpoint_format(self):
        p = partial_problem(((0, 1, 1, 3),))
        s = _Search(p, 4, 7, None, None, None, None, 5)
        s.build()
        domains = at_node(p, {0, 1})
        s.independent_completion(domains, list(range(4)))
        s.restoring = True
        with patch.object(s, 'valid_matching', wraps=s.valid_matching) as valid:
            self.assertIsNone(s.independent_completion(domains, list(range(4))))
            self.assertEqual(valid.call_count, 1)
        self.assertNotIn('independent_blocker', s.checkpoint())

        problem, _ = generated_problem(4, 3, 0)
        options = {'independent_completion': True, 'probe': True, 'probe_max_candidates': 1}
        stop = False
        def on_save(checkpoint):
            nonlocal stop
            stop = checkpoint['counters']['branches'] >= 2
        args = dict(seed=20261007, seconds=5, options=options)
        interrupted = solve(problem, on_checkpoint=on_save, checkpoint_interval=0,
                            should_stop=lambda: stop, **args)
        self.assertEqual(interrupted['outcome'], 'stopped')
        resumed = solve(problem, checkpoint=interrupted['checkpoint'], **args)
        fresh = solve(problem, **args)
        self.assertEqual((resumed['board'], resumed['branches'], resumed['independent_completions']),
                         (fresh['board'], fresh['branches'], fresh['independent_completions']))
        self.assertTrue(check_solution(problem, resumed['board']))


if __name__ == '__main__':
    unittest.main()
