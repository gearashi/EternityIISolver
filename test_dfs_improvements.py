"""Bounded independent oracles for optional exact DFS improvements.

Only generated <=4x4 fixtures are used; every solve has a two-second cap.
These are correctness checks, not Eternity II performance evidence.
"""
import itertools
import json
import random
import unittest
from unittest.mock import patch

import exact_dfs
from exact_model import ExactProblem, check_solution, generated_problem, make_problem
from test_exact_dfs import cyclic_unsat, exhaustive, oracle_domains, reseal


def search_for(problem, options=None):
    search = exact_dfs._Search(problem, 2, 17, None, None, None, None, 5, options=options)
    return search, search.build()


def corner_problem(pairs=((1, 1), (1, 2), (2, 1), (2, 2)), edges=None):
    faces = []
    for a, b in pairs:
        base = (0, a, b, 0)
        faces.extend(tuple(base[(side - r) % 4] for side in range(4)) for r in range(4))
    return make_problem(2, faces, edges=edges)


def dual_problem():
    source = corner_problem(((1, 1),) * 4, edges=())
    domains = tuple(tuple(s for s in d if c < 2 or s // 4 != 0) for c, d in enumerate(source.domains))
    return make_problem(2, source.faces, domains=domains, edges=())


def all_solutions(problem):
    # Scalar complete-tuple enumeration, independent of DFS and its matching.
    return [board for board in itertools.product(*oracle_domains(problem)) if check_solution(problem, board)]


class DfsImprovementsTests(unittest.TestCase):
    def test_scc_deletes_tight_hall_set_edges_missing_from_baseline(self):
        source = corner_problem(((1, 1),) * 4, edges=())
        domains = tuple(tuple(s for s in d if c >= 2 or s // 4 < 2) for c, d in enumerate(source.domains))
        problem = make_problem(2, source.faces, domains=domains, edges=())
        baseline, old = search_for(problem)
        self.assertIsNotNone(baseline.propagate(old, range(4), None))
        enhanced, new = search_for(problem, {'gac': True})
        self.assertIsNotNone(enhanced.propagate(new, range(4), None))
        self.assertEqual(old[:2], new[:2])
        for cell in (2, 3):
            self.assertEqual({s // 4 for s in exact_dfs._bits(new[cell])}, {2, 3})
            self.assertEqual({s // 4 for s in exact_dfs._bits(old[cell])}, {0, 1, 2, 3})
        self.assertEqual(enhanced.gac_removed, 4)

    def test_scc_support_equals_exhaustive_perfect_matchings(self):
        problem = corner_problem(((1, 1),) * 4, edges=())
        search, initial = search_for(problem)
        rng = random.Random(1251)
        for case in range(48):
            graph = [{p for p in range(4) if rng.random() < .6} for _ in range(4)]
            supports = [set() for _ in range(4)]
            matchings = [p for p in itertools.permutations(range(4)) if all(p[c] in graph[c] for c in range(4))]
            domains = [sum(1 << s for s in exact_dfs._bits(initial[c]) if s // 4 in graph[c]) for c in range(4)]
            matched = search.matching(domains, None)
            with self.subTest(case=case):
                self.assertEqual(matched is None, not matchings)
                if not matchings:
                    continue
                for matching in matchings:
                    for cell, piece in enumerate(matching):
                        supports[cell].add(piece)
                search.filter_matching(domains, matched)
                self.assertEqual([{s // 4 for s in exact_dfs._bits(d)} for d in domains], supports)

    def test_placeholder_matching_skips_scc_and_completion(self):
        problem = corner_problem(((1, 1),) * 4, edges=())
        search, domains = search_for(problem, {'gac': True, 'independent_completion': True})
        before = domains.copy()
        self.assertEqual(search.filter_matching(domains, [-1] * 4), set())
        self.assertIsNone(search.independent_completion(domains, [-1] * 4))
        self.assertEqual(domains, before)
        with patch.object(exact_dfs._Search, 'matching', return_value=[-1] * 4):
            result = exact_dfs.solve(problem, seconds=2, options={'gac': True, 'independent_completion': True})
        self.assertEqual(result['outcome'], 'solved')
        self.assertGreater(result['branches'], 0)
        self.assertTrue(check_solution(problem, result['board']))

    def test_independent_cells_complete_without_branching(self):
        source, board = generated_problem(3, 1, 291)
        fixed = {c: board[c] for c in range(9) if (c // 3 + c % 3) % 2 == 0}
        problem = make_problem(3, source.faces, fixed)
        result = exact_dfs.solve(problem, seconds=2, options={'independent_completion': True})
        self.assertEqual(result['outcome'], 'solved')
        self.assertEqual(result['branches'], 0)
        self.assertEqual(result['independent_completions'], 1)
        self.assertTrue(check_solution(problem, result['board']))
        adjacent = corner_problem(((1, 1),) * 4)
        search, domains = search_for(adjacent)
        matched = search.propagate(domains, range(4), None)
        self.assertIsNone(search.independent_completion(domains, matched))

    def test_cell_piece_and_edge_partitions_cover_solutions_once(self):
        cases = [(corner_problem(((1, 1),) * 4), {}, 'cell'),
                 (dual_problem(), {'dual_branch': True}, 'piece'),
                 (corner_problem(), {'edge_branch': True}, 'edge')]
        for problem, options, expected_kind in cases:
            search, domains = search_for(problem, options)
            self.assertIsNotNone(search.propagate(domains, range(4), None))
            kind, target, choices = search.choose_partition(domains)
            with self.subTest(kind=kind):
                self.assertEqual(kind, expected_kind)
                solutions = all_solutions(problem)
                self.assertTrue(solutions)
                children = [search.apply_choice(domains, kind, target, choice)[0] for choice in choices]
                for board in solutions:
                    self.assertEqual(sum(all(d & (1 << s) for d, s in zip(child, board)) for child in children), 1)
                result = exact_dfs.solve(problem, seconds=2, seed=17, options=options)
                self.assertEqual(result['outcome'], 'solved')
                self.assertGreater(result[kind + '_branches'], 0)

    def test_each_option_agrees_with_independent_tiny_oracle(self):
        configurations = [{name: True} for name in ('gac', 'independent_completion', 'dual_branch', 'edge_branch', 'probe')]
        configurations.append(dict(gac=True, independent_completion=True, dual_branch=True, edge_branch=True, probe=True))
        rng = random.Random(611)
        cases = [cyclic_unsat(), dual_problem(), corner_problem()]
        for seed in range(16):
            source, known = generated_problem(2, 3, seed)
            domains = tuple(tuple(s for s in d if rng.random() < .6 or (seed % 2 == 0 and s == known[c])) for c, d in enumerate(source.domains))
            cases.append(ExactProblem(2, source.faces, {}, domains))
        for index, problem in enumerate(cases):
            expected = 'solved' if exhaustive(problem) is not None else 'infeasible'
            for options in configurations:
                with self.subTest(case=index, options=options):
                    result = exact_dfs.solve(problem, seconds=2, options=options)
                    self.assertEqual(result['outcome'], expected)
                    if result['board'] is not None:
                        self.assertTrue(check_solution(problem, result['board']))

    def test_probes_prove_only_failed_values_and_do_not_mutate_input(self):
        problem = cyclic_unsat()
        search, domains = search_for(problem, {'probe': True})
        self.assertIsNotNone(search.propagate(domains, range(4), None))
        before = problem.domains
        result = exact_dfs.solve(problem, seconds=2, options={'probe': True})
        self.assertEqual(result['outcome'], 'infeasible')
        self.assertEqual(result['branches'], 0)
        self.assertGreater(result['probe_rejections'], 0)
        self.assertEqual(problem.domains, before)
        limited = exact_dfs.solve(corner_problem(), seconds=2, options={'probe': True, 'probe_max_candidates': 1, 'probe_max_depth': 0})
        self.assertEqual(limited['probe_checks'], 1)
        self.assertTrue(check_solution(corner_problem(), limited['board']))

    def test_optional_frontiers_resume_with_identical_decisions(self):
        cases = [(dual_problem(), {'dual_branch': True}),
                 (corner_problem(), {'edge_branch': True}),
                 (corner_problem(((1, 1),) * 4), {'gac': True, 'probe': True})]
        for problem, options in cases:
            fresh = exact_dfs.solve(problem, seconds=2, seed=17, options=options)
            checkpoint = None
            for _ in range(32):
                old_count = 0 if checkpoint is None else checkpoint['counters']['branches']
                stop = [False]
                def observe(snapshot):
                    stop[0] = snapshot['counters']['branches'] > old_count
                result = exact_dfs.solve(problem, seconds=2, seed=17, options=options, checkpoint=checkpoint,
                                         should_stop=lambda: stop[0], on_checkpoint=observe, checkpoint_interval=0)
                if result['outcome'] != 'stopped':
                    break
                checkpoint = json.loads(json.dumps(result['checkpoint']))
            else:
                self.fail('Tiny frontier did not terminate')
            with self.subTest(options=options):
                self.assertIsNotNone(checkpoint)
                self.assertEqual(result['outcome'], fresh['outcome'])
                self.assertEqual(result['board'], fresh['board'])
                self.assertEqual(result['branches'], fresh['branches'])
                self.assertEqual(result['conflicts'], fresh['conflicts'])

    def test_stop_during_probe_repeats_entire_pending_node(self):
        problem = corner_problem()
        options = {'probe': True, 'gac': True}
        fresh = exact_dfs.solve(problem, seconds=2, seed=17, options=options)
        original = exact_dfs._Search.check
        def stop_in_probe(search):
            if search.probe_checks > 0:
                raise exact_dfs._Interrupted('stopped')
            return original(search)
        with patch.object(exact_dfs._Search, 'check', stop_in_probe):
            interrupted = exact_dfs.solve(problem, seconds=2, seed=17, options=options)
        self.assertEqual(interrupted['outcome'], 'stopped')
        self.assertEqual(interrupted['checkpoint']['stack'], [{'phase': 'pending'}])
        resumed = exact_dfs.solve(problem, seconds=2, seed=17, options=options, checkpoint=interrupted['checkpoint'])
        self.assertEqual(resumed['board'], fresh['board'])
        self.assertEqual(resumed['branches'], fresh['branches'])

    def test_settings_are_bound_and_legacy_is_rejected_without_mutation(self):
        problem = corner_problem()
        saved = exact_dfs.solve(problem, seconds=0, options={'gac': True})['checkpoint']
        before = json.dumps(saved, sort_keys=True)
        with self.assertRaisesRegex(ValueError, 'options hash'):
            exact_dfs.solve(problem, seconds=2, checkpoint=saved)
        self.assertEqual(json.dumps(saved, sort_keys=True), before)
        legacy = reseal(dict(saved, version=2))
        with self.assertRaisesRegex(ValueError, 'Unsupported DFS checkpoint version.*preserve'):
            exact_dfs.solve(problem, seconds=2, checkpoint=legacy, options={'gac': True})
        result = exact_dfs.solve(problem, seconds=2, checkpoint=saved, options={'gac': True})
        self.assertEqual(result['outcome'], 'solved')
        for options in ({'gac': 1}, {'unknown': True}, {'probe_max_candidates': True},
                        {'probe_max_depth': -1}, {'probe_max_domain': 65}, []):
            with self.subTest(options=options), self.assertRaises(ValueError):
                exact_dfs.solve(problem, seconds=2, options=options)

    def test_additional_rules_integration_and_checkpoint_identity(self):
        options = dict(gac=True, dual_branch=True, edge_branch=True, probe=True, extra_rules={})
        for problem in (corner_problem(), cyclic_unsat(), dual_problem()):
            expected = 'solved' if exhaustive(problem) is not None else 'infeasible'
            result = exact_dfs.solve(problem, seconds=2, options=options)
            self.assertEqual(result['outcome'], expected)
            self.assertGreater(result['rule_counters_session']['checks'], 0)
            self.assertTrue(result['rules_version'])
        problem = corner_problem(((1, 1),) * 4)
        stop = [False]
        def observe(snapshot):
            stop[0] = snapshot['counters']['branches'] > 0
        saved = exact_dfs.solve(problem, seconds=2, seed=17, options=options,
                                should_stop=lambda: stop[0], on_checkpoint=observe, checkpoint_interval=0)['checkpoint']
        self.assertIsNotNone(saved)
        fresh = exact_dfs.solve(problem, seconds=2, seed=17, options=options)
        resumed = exact_dfs.solve(problem, seconds=2, seed=17, options=options, checkpoint=saved)
        self.assertEqual(resumed['board'], fresh['board'])
        self.assertEqual(resumed['branches'], fresh['branches'])
        changed = dict(options, extra_rules={'hall': False})
        with self.assertRaisesRegex(ValueError, 'options hash'):
            exact_dfs.solve(problem, seconds=2, seed=17, options=changed, checkpoint=saved)

    def test_regional_edges_remain_optional_with_every_filter(self):
        source, answer = generated_problem(2, 3, 831)
        faces = list(source.faces)
        piece, orientation = divmod(answer[0], 4)
        base = list(faces[4 * piece])
        base[(1 - orientation) % 4] = 99
        faces[4 * piece:4 * piece + 4] = [tuple(base[(s - r) % 4] for s in range(4)) for r in range(4)]
        problem = make_problem(2, faces, fixed=dict(enumerate(answer)), edges=((2, 3, 1, 3),))
        self.assertTrue(check_solution(problem, answer))
        options = dict(gac=True, independent_completion=True, dual_branch=True, edge_branch=True, probe=True, extra_rules={})
        result = exact_dfs.solve(problem, seconds=2, options=options)
        self.assertEqual(result['outcome'], 'solved')
        self.assertEqual(result['board'], answer)


if __name__ == '__main__':
    unittest.main()
