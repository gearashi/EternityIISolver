from dataclasses import replace
from pathlib import Path
import unittest

from exact_model import generated_problem
from exact_optimize import evaluate_board
from repair_native_radius import search


def planted_cycle(size, cells):
    problem, solution = generated_problem(size, 12, 142)
    parent = list(solution)
    for index, cell in enumerate(cells):
        parent[cell] = solution[cells[(index + 1) % len(cells)]]
    # The distinct-piece constraint permits exactly the parent or the whole
    # cycle. All neighboring exterior tiles stay fixed.
    domains = tuple(tuple(dict.fromkeys(pair)) for pair in zip(parent, solution))
    return replace(problem, domains=domains), parent, solution


class NativeRadiusTransferCycle(unittest.TestCase):
    def test_in_scope_cycle_recovered_without_external_binary(self):
        problem, parent, target = planted_cycle(5, (6, 8, 16, 18))
        result = search(problem, parent, radius=4, helper=Path(__file__)/'missing')
        self.assertEqual(result['termination'], 'improved')
        self.assertEqual(result['board'], target)
        self.assertEqual(evaluate_board(problem, result['board'])['score'], 40)
        self.assertEqual(result['native_phases'][-1]['witness_distance'], 4)
        self.assertEqual(result['max_exhausted_radius'], 3)

    def test_out_of_scope_improvement_is_not_returned(self):
        problem, parent, target = planted_cycle(6, (7, 9, 16, 19, 21))
        result = search(problem, parent, radius=4, helper=__file__)
        self.assertEqual(result['termination'], 'transfer_bound_unavailable')
        self.assertEqual(result['board'], parent)
        self.assertEqual(result['native_phases'][-1]['witness_distance'], 5)
        self.assertEqual(result['max_exhausted_radius'], 3)
