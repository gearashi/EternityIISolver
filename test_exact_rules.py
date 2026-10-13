"""Bounded synthetic/oracle checks for optional exact rules; no EII search."""
import itertools
import random
import unittest

from exact_model import ExactProblem, check_solution, generated_problem, make_problem
from exact_rules import AdditionalRules, DEFAULT_OPTIONS, normalize_options


def bitdomains(problem):
    return [sum(1 << state for state in domain) for domain in problem.domains]


def only(name, **caps):
    result = {key: False for key, value in DEFAULT_OPTIONS.items() if type(value) is bool}
    result[name] = True
    result.update(caps)
    return result


def rotations(base):
    return [tuple(base[(side - rot) % 4] for side in range(4)) for rot in range(4)]


def even_weight_problem():
    # Every piece has 0 or 2 copies of color 1, but ten copies overall.
    # Every type can occupy either parity on this 4x4 grid.
    bases = [(0, 1, 1, 0)] * 4 + [(0, 2, 2, 2)] * 8
    bases += [(1, 1, 2, 2)] + [(2, 2, 2, 2)] * 3
    return make_problem(4, [face for base in bases for face in rotations(base)])


def loop_problem():
    # Isolated loop, with no frame claim: each edge has support but the four
    # binary orientations enforce A=B, B=C, D=A, C!=D.
    bases = [(2, 1, 1, 2), (2, 2, 1, 1), (1, 2, 2, 1), (1, 2, 2, 1)]
    return ExactProblem(2, tuple(face for base in bases for face in rotations(base)), {},
                        ((0, 2), (4, 6), (8, 10), (12, 14)))


class AdditionalRuleTests(unittest.TestCase):
    def test_strict_options(self):
        self.assertEqual(normalize_options(), DEFAULT_OPTIONS)
        for bad in ({'unknown': True}, {'hall': 1}, {'block_pairs': True},
                    {'block_pairs': -1}, {'flow_colors': 1000001}, []):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_options(bad)
        self.assertEqual(normalize_options({'block_pairs': 0})['block_pairs'], 0)

    def test_every_rule_preserves_every_tiny_solution(self):
        variants = [None] + [only(name) for name in ('hall', 'color_balance', 'blocks', 'color_flow', 'directional_balance')]
        for seed in range(6):
            original, answer = generated_problem(2, 3, seed)
            rng = random.Random(seed)
            restrictions = [original.domains, tuple(tuple(s for s in d if s == answer[c] or rng.randrange(2))
                             for c, d in enumerate(original.domains))]
            for supplied in restrictions:
                problem = ExactProblem(2, original.faces, original.fixed, supplied)
                solutions = [board for board in itertools.product(*problem.domains) if check_solution(problem, board)]
                self.assertTrue(solutions)
                for options in variants:
                    rules = AdditionalRules(problem, options=options)
                    domains = bitdomains(problem)
                    before = domains.copy()
                    consistent, changed = rules.apply(domains)
                    self.assertTrue(consistent, (seed, options))
                    self.assertEqual(changed, {c for c in range(4) if domains[c] != before[c]})
                    for board in solutions:
                        self.assertTrue(all(domains[c] & (1 << state) for c, state in enumerate(board)), (seed, options, board))

    def test_identical_hall_reserves_physical_pieces_not_rotations(self):
        problem, _ = generated_problem(2, 2, 4)
        domains = [0xff, 0xff, 0xffff, 0xffff]
        rules = AdditionalRules(problem, options=only('hall'))
        self.assertEqual(rules.apply(domains), (True, {2, 3}))
        self.assertEqual(domains, [0xff, 0xff, 0xff00, 0xff00])
        self.assertFalse(rules.apply([0xff, 0xff, 0xff, 0xffff])[0])

    def test_cardinality_dp_detects_even_weight_gap(self):
        problem = even_weight_problem()
        rules = AdditionalRules(problem, options=only('color_balance'))
        self.assertFalse(rules.apply(bitdomains(problem))[0])
        self.assertGreater(rules.counters['balance_work'], 0)
        bounded = AdditionalRules(problem, options=only('color_balance', balance_work=0))
        self.assertTrue(bounded.apply(bitdomains(problem))[0])
        self.assertGreater(bounded.counters['balance_skips'], 0)

    def test_regional_and_noninvariant_guards(self):
        full = even_weight_problem()
        regional = ExactProblem(full.size, full.faces, full.fixed, full.domains, ())
        for name in ('color_balance', 'directional_balance', 'color_flow'):
            rules = AdditionalRules(regional, options=only(name))
            self.assertTrue(rules.apply(bitdomains(regional))[0])
        faces = list(full.faces)
        faces[0] = (0, 2, 2, 0)
        noninvariant = make_problem(4, faces)
        for name in ('color_balance', 'color_flow'):
            rules = AdditionalRules(noninvariant, options=only(name))
            self.assertFalse(rules.invariant)
            self.assertTrue(rules.apply(bitdomains(noninvariant))[0])

    def test_unframed_input_skips_board_identities(self):
        problem = loop_problem()
        for name in ('color_balance', 'directional_balance', 'color_flow'):
            rules = AdditionalRules(problem, options=only(name))
            self.assertTrue(rules.apply(bitdomains(problem))[0])

    def test_loop_join_proves_joint_incompatibility(self):
        problem = loop_problem()
        # Check the independent scalar edge supports first.
        for a, b, sa, sb in ((0, 1, 1, 3), (0, 2, 2, 0), (1, 3, 2, 0), (2, 3, 1, 3)):
            for state in problem.domains[a]:
                self.assertTrue(any(problem.faces[state][sa] == problem.faces[other][sb] for other in problem.domains[b]))
        rules = AdditionalRules(problem, options=only('blocks'))
        self.assertFalse(rules.apply(bitdomains(problem))[0])
        self.assertGreater(rules.counters['block_pairs'], 0)

    def test_loop_is_skipped_if_an_edge_is_not_required(self):
        problem = loop_problem()
        regional = ExactProblem(2, problem.faces, {}, problem.domains, ((0, 1, 1, 3),))
        rules = AdditionalRules(regional, options=only('blocks'))
        self.assertTrue(rules.apply(bitdomains(regional))[0])
        self.assertEqual(rules.counters['block_pairs'], 0)

    def test_caps_skip_before_support_enumeration(self):
        problem = loop_problem()
        for caps in ({'block_pairs': 0}, {'block_total_pairs': 0}):
            domains = bitdomains(problem)
            before = domains.copy()
            rules = AdditionalRules(problem, options=only('blocks', **caps))
            self.assertEqual(rules.apply(domains), (True, set()))
            self.assertEqual(domains, before)
            self.assertEqual(rules.counters['block_pairs'], 0)

    def test_interruption_mid_block_never_deletes_partial_support(self):
        problem = loop_problem()
        rules = AdditionalRules(problem, options=only('blocks'))
        calls = 0
        class Stop(Exception):
            pass
        def check():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise Stop()
        rules.check = check
        domains = bitdomains(problem)
        before = domains.copy()
        with self.assertRaises(Stop):
            rules._blocks(domains, set())
        self.assertEqual(domains, before)

    def test_flow_matches_exhaustive_edge_subsets(self):
        problem, _ = generated_problem(2, 2, 1)
        rules = AdditionalRules(problem, options=only('color_flow'))
        all_edges = [(0, 1), (0, 2), (3, 1), (3, 2)]
        rng = random.Random(91)
        for case in range(100):
            edges = [edge for edge in all_edges if rng.randrange(2)]
            degree = []
            for _ in range(4):
                lo = rng.randrange(3)
                degree.append((lo, rng.randrange(lo, 3)))
            target = rng.randrange(5)
            expected = False
            for flags in itertools.product((0, 1), repeat=len(edges)):
                if sum(flags) != target:
                    continue
                values = [0] * 4
                for (a, b), chosen in zip(edges, flags):
                    values[a] += chosen
                    values[b] += chosen
                if all(lo <= value <= hi for value, (lo, hi) in zip(values, degree)):
                    expected = True
            self.assertEqual(rules._bounded_flow(degree, edges, target), expected, case)

    def test_options_and_domains_determine_pruning_without_history(self):
        problem, _ = generated_problem(2, 3, 2)
        rules = AdditionalRules(problem)
        one, two = bitdomains(problem), bitdomains(problem)
        self.assertEqual(rules.apply(one), rules.apply(two))
        self.assertEqual(one, two)

    def test_invalid_domains_rejected(self):
        problem, _ = generated_problem(2, 2, 1)
        rules = AdditionalRules(problem)
        for bad in ([1], [-1]*4, [True]*4, [1 << 16]*4):
            with self.assertRaises(ValueError):
                rules.apply(bad)


if __name__ == '__main__':
    unittest.main()
