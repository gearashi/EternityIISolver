"""Independent scalar and legacy differentials for support-mask optimization."""
import random
import unittest
from unittest.mock import patch

import exact_dfs
from exact_model import check_solution, generated_problem
from test_exact_dfs import cyclic_unsat


def legacy_supported(search, source, source_side, target_side):
    allowed = 0
    for color, states in search.colors[source_side].items():
        support = source & states
        if support:
            candidates = search.colors[target_side].get(color, 0)
            pieces = search.pieces(support)
            if pieces & (pieces - 1) == 0:
                candidates &= ~(15 * pieces)
            allowed |= candidates
    return allowed


def scalar_supported(search, source, source_side, target_side):
    states = [state for state in range(len(search.faces)) if source & (1 << state)]
    result = 0
    for candidate, face in enumerate(search.faces):
        if any(search.faces[state][source_side] == face[target_side] and state // 4 != candidate // 4
               for state in states):
            result |= 1 << candidate
    return result


class SupportedMaskTests(unittest.TestCase):
    def test_single_piece_test_matches_scalar_projection_for_broad_masks(self):
        rng = random.Random(6042)
        masks = [0, 1, 15, 16, 17, 1 << 1023, 15 << 1020, (1 << 1024) - 1]
        for piece in range(256):
            masks.extend(orientation_bits << (4 * piece) for orientation_bits in range(1, 16))
        masks += [rng.getrandbits(rng.randrange(1, 1025)) for _ in range(4000)]
        for mask in masks:
            expected_pieces = {state // 4 for state in exact_dfs._bits(mask)}
            candidate = 0 if not mask else 15 << ((mask.bit_length() - 1) & ~3)
            actual = candidate if mask and mask & ~candidate == 0 else 0
            expected = 15 << (4 * next(iter(expected_pieces))) if len(expected_pieces) == 1 else 0
            self.assertEqual(actual, expected)

    def test_supported_matches_independent_scalar_oracle(self):
        problem, _ = generated_problem(3, 5, 608)
        search = exact_dfs._Search(problem, 10, 1, None, None, None, None, 5)
        search.build()
        rng = random.Random(93)
        masks = [0, 1, 15, 1 << 35, 15 << 32, (1 << 36) - 1]
        masks += [rng.getrandbits(36) for _ in range(64)]
        masks += [rng.randrange(1, 16) << (4 * piece) for piece in range(9)]
        for mask in masks:
            for source_side in range(4):
                for target_side in range(4):
                    actual = search.supported(mask, source_side, target_side)
                    self.assertEqual(actual, scalar_supported(search, mask, source_side, target_side))
                    self.assertEqual(actual, legacy_supported(search, mask, source_side, target_side))

    def test_full_width_random_domains_match_legacy_support(self):
        problem, _ = generated_problem(16, 22, 990)
        search = exact_dfs._Search(problem, 10, 1, None, None, None, None, 5)
        search.build()
        rng = random.Random(3773)
        masks = [0, 1 << 1023, 15 << 1020, (1 << 1024) - 1]
        masks += [rng.getrandbits(1024) for _ in range(256)]
        masks += [rng.randrange(1, 16) << (4 * piece) for piece in range(256)]
        for mask in masks:
            source_side, target_side = rng.randrange(4), rng.randrange(4)
            self.assertEqual(search.supported(mask, source_side, target_side),
                             legacy_supported(search, mask, source_side, target_side))

    def test_small_search_order_results_and_adaptive_policy_are_identical(self):
        options_list = ({}, {'gac': True, 'extra_rules': {}, 'probe': True, 'adaptive': True})
        problems = [cyclic_unsat()]
        problems += [generated_problem(3, 2, seed)[0] for seed in range(12)]
        for problem in problems:
            for options in options_list:
                current = exact_dfs.solve(problem, seconds=2, seed=17, options=options)
                with patch.object(exact_dfs._Search, 'supported', legacy_supported):
                    legacy = exact_dfs.solve(problem, seconds=2, seed=17, options=options)
                self.assertIn(current['outcome'], ('solved', 'infeasible'))
                for key in ('outcome', 'board', 'branches', 'conflicts', 'max_depth', 'adaptive_policy'):
                    self.assertEqual(current[key], legacy[key], key)
                if current['board'] is not None:
                    self.assertTrue(check_solution(problem, current['board']))


if __name__ == '__main__':
    unittest.main()
