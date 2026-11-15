from dataclasses import replace
import itertools
import unittest

from exact_model import generated_problem
from exact_optimize import evaluate_board
from repair_radius import search as search_pairs
from repair_triples import search


def cycle_only():
    p, solution = generated_problem(5, 20, 38)
    cells = (6, 8, 17)
    seed = list(solution)
    for c, origin in zip(cells, cells[1:] + cells[:1]):
        seed[c] = solution[origin]
    domains = tuple(tuple(sorted({s,t})) for s,t in zip(seed,solution))
    return replace(p, domains=domains), seed


class TripleControls(unittest.TestCase):
    def test_exhaustive_corners_and_small_connected_domains(self):
        observed = set()
        cases = [cycle_only()[0]]
        for seed in (117, 118, 201):
            problem, board = generated_problem(2, 5, seed)
            cases.append(problem)
        problem, board = generated_problem(4, 7, 330)
        cells = (5,6,9)
        pool = {board[c]//4 for c in cells}
        choices = tuple(tuple(s for s in domain if s == board[c] or
                        (c in cells and s//4 in pool and s%4 in (0,2)))
                        for c,domain in enumerate(problem.domains))
        cases.append(replace(problem, domains=choices))
        for p in cases:
            valid = [(b,ans['score']) for b in itertools.product(*p.domains)
                     if (ans:=evaluate_board(p,b))['valid']]
            for b,start in valid:
                possible = any(score>start and sum(x!=y for x,y in zip(candidate,b))<=3
                               for candidate,score in valid)
                r=search(p,b)
                self.assertEqual(possible,r['improved'])
                if possible:
                    self.assertEqual(r['score'], evaluate_board(p,r['board'])['score'])
                    observed.add(sum(x!=y for x,y in zip(b,r['board'])))
                else:
                    self.assertEqual(3,r['max_exhausted_radius'])
                    observed.add(0)
        self.assertIn(3,observed)
        self.assertIn(0,observed)

    def test_bound_after_pair_exhaustion_is_partial(self):
        p, parent = cycle_only()
        pair = search_pairs(p, parent)
        self.assertEqual('exhausted',pair['termination'])
        result = search(p, parent, max_work=pair['work'])
        self.assertEqual(('work_limit', 2), (result['termination'],result['max_exhausted_radius']))
        self.assertEqual(parent,result['board'])
        self.assertTrue(search(p,parent)['improved'])
