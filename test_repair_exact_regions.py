"""Finite independent enumeration oracles; no live cache, GPU or network."""
from dataclasses import replace
import itertools
import unittest

from exact_model import generated_problem, grid_edges, make_problem
from exact_optimize import evaluate_board
from repair_exact_regions import RegionFailureCache, patch_lower_bound, region_signature, solve_region
from test_exact_sat import tiny_problem


def legal_boards(p):
    return [(e['score'], list(b)) for b in itertools.product(*p.domains)
            if (e := evaluate_board(p, b))['valid']]


def scramble(p, board):
    result = list(board)
    groups = {}
    for c, s in enumerate(board):
        if c not in p.fixed: groups.setdefault(sum(color == 0 for color in p.faces[s]), []).append(c)
    for cells in groups.values():
        pieces = [result[c] // 4 for c in cells]
        for c, piece in zip(cells, pieces[1:] + pieces[:1]):
            result[c] = next(s for s in p.domains[c] if s // 4 == piece)
    return result


class ExactRegionTests(unittest.TestCase):
    def test_target_decisions_match_all_tiny_legal_completions(self):
        for impossible in (False, True):
            p = tiny_problem(impossible)
            boards = legal_boards(p)
            reference = min(boards)[1]
            for cells in ([0, 1], [0, 1, 2], [0, 3], [0, 1, 2, 3]):
                optimum = max(score for score, b in boards if all(b[c] == reference[c] for c in range(4) if c not in cells))
                for target in (optimum, optimum + 1):
                    for algorithm, ordering in (('frontier', 'mrv'), ('dfs', 'natural'), ('dfs', 'mrv'), ('dfs', 'failure')):
                        with self.subTest(impossible=impossible, cells=cells, target=target, algorithm=algorithm, ordering=ordering):
                            result = solve_region(p, reference, cells, target_score=target, algorithm=algorithm, ordering=ordering)
                            self.assertEqual(result['target_reached'], optimum >= target)
                            self.assertEqual(result['proven_unreachable'], optimum < target)
                            self.assertTrue(evaluate_board(p, result['board'])['valid'])
                            self.assertEqual(evaluate_board(p, result['board'])['score'], result['score'])
                            self.assertTrue(all(result['board'][c] == reference[c] for c in range(4) if c not in cells))

    def test_cache_reuses_proof_across_internal_permutations(self):
        p = tiny_problem(True)
        boards = legal_boards(p); cache = RegionFailureCache(2)
        first = solve_region(p, boards[0][1], range(4), target_score=4, cache=cache)
        self.assertTrue(first['proven_unreachable']); self.assertGreater(first['work'], 0)
        second = solve_region(p, boards[-1][1], range(4), target_score=4, cache=cache)
        self.assertTrue(second['cache_hit']); self.assertTrue(second['proven_unreachable'])
        self.assertEqual(second['work'], 0); self.assertEqual(len(cache), 1)
        second['board'][0] = -1
        self.assertTrue(evaluate_board(p, solve_region(p, boards[0][1], range(4), target_score=4, cache=cache)['board'])['valid'])

    def test_cache_keys_bind_target_pool_domain_clues_and_frozen_complement(self):
        p = tiny_problem(True); board = min(legal_boards(p))[1]
        key = region_signature(p, board, [0, 1], 4)
        self.assertNotEqual(key, region_signature(p, board, [0, 1], 3))
        alternative = next(b for _, b in legal_boards(p) if b[2:] != board[2:])
        self.assertNotEqual(key, region_signature(p, alternative, [0, 1], 4))
        q = replace(p, fixed={2: board[2]})
        self.assertNotEqual(key, region_signature(q, board, [0, 1], 4))
        q = replace(p, domains=tuple((s,) if c == 0 else d for c, (s, d) in enumerate(zip(board, p.domains))))
        self.assertNotEqual(key, region_signature(q, board, [0, 1], 4))
        self.assertNotEqual(key, region_signature(p, board, [0, 2], 4))

    def test_limits_and_cancellation_never_cache_or_claim_absence(self):
        p = tiny_problem(True); board = min(legal_boards(p))[1]
        for algorithm in ('dfs', 'frontier'):
            for kwargs in ({'max_work': 0}, {'should_stop': lambda: True}):
                cache = RegionFailureCache()
                result = solve_region(p, board, range(4), target_score=4, algorithm=algorithm, cache=cache, **kwargs)
                self.assertFalse(result['proven_unreachable']); self.assertFalse(result['complete']); self.assertEqual(len(cache), 0)
                self.assertIn(result['termination'], ('work_limit', 'stopped'))
        p, answer = generated_problem(3, 2, 18)
        board = scramble(p, answer)
        result = solve_region(p, board, range(9), max_states=1)
        self.assertEqual(result['termination'], 'state_limit'); self.assertFalse(result['proven_unreachable'])

    def test_three_by_three_region_with_clues_and_domains(self):
        p, answer = generated_problem(3, 3, 41, clue_count=1)
        board = scramble(p, answer); cells = [c for c in range(9) if c not in p.fixed]
        for algorithm in ('dfs', 'frontier'):
            r = solve_region(p, board, cells, target_score=12, max_work=50000, algorithm=algorithm)
            self.assertTrue(r['target_reached'], r)
            self.assertTrue(all(r['board'][c] == s for c, s in p.fixed.items()))
            self.assertEqual(r['score'], 12)
        with self.assertRaises(ValueError): solve_region(p, board, range(9))

    def test_optional_patch_bounds_preserve_oracle_and_count_their_work(self):
        for impossible in (False, True):
            p = tiny_problem(impossible); boards = legal_boards(p)
            reference = min(boards)[1]; optimum = max(s for s, _ in boards)
            for algorithm in ('dfs', 'frontier'):
                for target in (optimum, optimum + 1):
                    r = solve_region(p, reference, range(4), target_score=target,
                                     algorithm=algorithm, patch_bounds=True)
                    self.assertEqual(r['target_reached'], target <= optimum)
                    self.assertEqual(r['proven_unreachable'], target > optimum)
                    self.assertEqual(r['search_work'], r['candidate_placements'] + r['bound_assignments'])
                    self.assertLessEqual(r['work'], 50000)
        p, answer = generated_problem(3, 3, 41, clue_count=1)
        r = solve_region(p, scramble(p, answer), [c for c in range(9) if c not in p.fixed],
                         target_score=12, algorithm='dfs', patch_bounds=True, max_work=4)
        self.assertFalse(r['proven_unreachable']); self.assertLessEqual(r['work'], 4)

    def test_interrupted_prefix_has_no_negative_proof_or_cache(self):
        p = tiny_problem(True); board = min(legal_boards(p))[1]
        for algorithm in ('dfs', 'frontier'):
            calls = [0]
            def stopping():
                calls[0] += 1
                return calls[0] >= 5
            cache = RegionFailureCache()
            r = solve_region(p, board, range(4), target_score=4, algorithm=algorithm,
                             patch_bounds=True, should_stop=stopping, cache=cache)
            self.assertEqual(r['termination'], 'stopped')
            self.assertFalse(r['proven_unreachable']); self.assertEqual(len(cache), 0)

    def test_empty_region_and_validated_reference_short_circuit(self):
        p = tiny_problem(True); score, board = min(legal_boards(p))
        no = solve_region(p, board, [], target_score=score + 1)
        self.assertTrue(no['proven_unreachable']); self.assertEqual(no['work'], 0)
        yes = solve_region(p, board, range(4), target_score=score)
        self.assertTrue(yes['target_reached']); self.assertEqual(yes['work'], 0)

    def test_argument_guards(self):
        p = tiny_problem(); board = min(legal_boards(p))[1]
        for kwargs in ({'max_work': -1}, {'max_work': True}, {'max_states': 0}, {'algorithm': 'no'},
                       {'ordering': 'no'}, {'patch_bounds': 1}, {'target_score': True}, {'should_stop': 1}, {'cache': {}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): solve_region(p, board, range(4), **kwargs)
        for cells in ([0, 0], [-1], [True], [4]):
            with self.assertRaises(ValueError): solve_region(p, board, cells)
        with self.assertRaises(ValueError): solve_region(replace(p, edges=()), board, range(4))
        with self.assertRaises(ValueError): solve_region(p, [-1] + board[1:], range(4))
        with self.assertRaises(ValueError): RegionFailureCache(0)

    def test_failure_ordering_is_deterministic_and_reduces_fixture_work(self):
        p, answer = generated_problem(3, 3, 41, clue_count=1)
        board = scramble(p, answer); cells = [c for c in range(9) if c not in p.fixed]
        plain = solve_region(p, board, cells, target_score=12, algorithm='dfs', ordering='natural')
        informed = solve_region(p, board, cells, target_score=12, algorithm='dfs', ordering='failure')
        repeated = solve_region(p, board, cells, target_score=12, algorithm='dfs', ordering='failure')
        self.assertTrue(plain['target_reached']); self.assertTrue(informed['target_reached'])
        self.assertEqual(informed, repeated)
        self.assertLess(informed['work'], plain['work'])


class PatchBoundTests(unittest.TestCase):
    def test_bound_never_exceeds_exhaustive_full_puzzle_optimum(self):
        for seed in range(3):
            p, _ = generated_problem(3, 2, seed)
            optimum = min(12 - score for score, _ in legal_boards(p))
            for cap in (0, 20000):
                r = patch_lower_bound(p, max_work=cap)
                self.assertTrue(r['complete']); self.assertLessEqual(r['lower_bound'], optimum)
                indices = [e for g in r['groups'] for e in g['edges']]
                self.assertEqual(sorted(indices), list(range(12)))
                self.assertLessEqual(r['work'], cap)

    def test_frozen_mismatched_board_bound_is_exact(self):
        p, answer = generated_problem(3, 4, 6)
        board = scramble(p, answer)
        q = make_problem(3, p.faces, domains=[(s,) for s in board])
        r = patch_lower_bound(q)
        self.assertEqual(r['lower_bound'], 12 - evaluate_board(q, board)['score'])
        self.assertTrue(any(g['kind'] == '2x3-relaxation' for g in r['groups']))

    def test_local_inventory_contradiction_and_domain_guard(self):
        p, answer = generated_problem(3, 2, 5)
        # Repeat the same two legal corner pieces in three corner cells: the
        # global domain set is impossible; unsupported patches need not detect it.
        q = replace(p, domains=((),) + p.domains[1:])
        self.assertTrue(patch_lower_bound(q)['structurally_infeasible'])
        with self.assertRaises(ValueError): patch_lower_bound(p, domains=[(999,)] * 9)
        with self.assertRaises(ValueError): patch_lower_bound(p, max_work=True)

    def test_stopped_bound_returns_no_partial_certificate(self):
        p, _ = generated_problem(3, 2, 8)
        r = patch_lower_bound(p, should_stop=lambda: True)
        self.assertFalse(r['complete']); self.assertIsNone(r['lower_bound']); self.assertEqual(r['groups'], [])


if __name__ == '__main__': unittest.main()
