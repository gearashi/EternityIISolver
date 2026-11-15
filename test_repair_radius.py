"""Independent exhaustive radius controls on tiny physical-piece puzzles."""
from dataclasses import replace
import itertools
import unittest

from exact_model import generated_problem
from exact_optimize import evaluate_board
from repair_radius import search


class SmallRadiusTests(unittest.TestCase):
    def test_all_legal_parents_and_targets(self):
        # Independent oracle lists *all* legal boards and filters by Hamming distance.
        # A high-color negative and a low-color tie/positive population both occur.
        improving_two_only = negatives = positives = 0
        for seed in (7, 28, 51, 172):
            problem, solution = generated_problem(2, 5, seed)
            for fixed in ({}, {0: solution[0]}):
                task = replace(problem, fixed=fixed)
                legal = [(board, check['score']) for board in itertools.product(*task.domains)
                         if (check := evaluate_board(task, board))['valid']]
                for board, score in legal:
                    for radius in (1, 2):
                        possible = any(points > score and sum(a != b for a,b in zip(candidate,board)) <= radius
                                       for candidate, points in legal)
                        found = search(task, board, radius=radius)
                        self.assertEqual(possible, found['improved'])
                        if found['improved']:
                            positives += 1
                            improving_two_only += int(found['max_exhausted_radius'] == 1)
                            checked = evaluate_board(task, found['board'])
                            self.assertTrue(checked['valid'])
                            self.assertEqual(found['score'], checked['score'])
                        else:
                            negatives += 1
                            self.assertEqual(radius, found['max_exhausted_radius'])
        self.assertGreater(min(negatives, positives, improving_two_only), 0)

    def test_rotations_and_restricted_domains(self):
        problem, answer = generated_problem(4, 4, 188)
        active = (5,6,9,10)
        ds = tuple(tuple(s for s in domain if (cell in active and s//4 == answer[cell]//4)
                         or s == answer[cell]) for cell, domain in enumerate(problem.domains))
        p = replace(problem, domains=ds)
        # Full enumeration includes adjacent and nonadjacent two-rotation interactions.
        boards = list(itertools.product(*ds))
        scores = [evaluate_board(p, b)['score'] for b in boards]
        for b, points in zip(boards[::7], scores[::7]):
            expected = any(q > points and sum(l != r for l,r in zip(b, candidate)) <= 2
                           for candidate,q in zip(boards,scores))
            self.assertEqual(expected, search(p,b)['improved'])

    def test_interruption_and_retry(self):
        problem, solution = generated_problem(3, 4, 2)
        parent = solution.copy()
        parent[4] = (parent[4]//4)*4 + (parent[4]+1)%4
        first = search(problem, parent, max_work=0)
        self.assertEqual(('work_limit',0,parent), (first['termination'],first['max_exhausted_radius'],first['board']))
        self.assertEqual('stopped', search(problem, parent, should_stop=lambda: True)['termination'])
        self.assertEqual(search(problem,parent),search(problem,parent))
        self.assertTrue(search(problem,parent)['improved'])

    def test_rejects_partial_edges_invalid_reference_and_caps(self):
        problem, solution = generated_problem(3, 4, 2)
        for args in ({'radius':True},{'radius':3},{'max_work':-1}):
            with self.assertRaises(ValueError):
                search(problem, solution, **args)
        with self.assertRaises(ValueError):
            search(replace(problem,edges=()),solution)
        with self.assertRaises(ValueError):
            search(problem, [0]*9)
