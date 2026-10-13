"""Finite correctness and deterministic replay checks for adaptive pruning."""
import json
import random
import unittest
from unittest.mock import patch

import exact_dfs
from exact_model import ExactProblem, check_solution, generated_problem
from test_exact_dfs import cyclic_unsat, exhaustive, reseal
from test_dfs_improvements import corner_problem, dual_problem, search_for

OPTIONS = dict(gac=True, dual_branch=True, edge_branch=True, probe=True,
               extra_rules={}, adaptive=True)


class AdaptiveDfsTests(unittest.TestCase):
    def test_adaptive_matches_exhaustive_tiny_oracle(self):
        rng = random.Random(2812)
        cases = [cyclic_unsat(), dual_problem(), corner_problem()]
        for seed in range(24):
            source, known = generated_problem(2, 3, seed)
            domains = tuple(tuple(s for s in domain if rng.random() < .6 or
                                 (seed % 2 == 0 and s == known[cell]))
                            for cell, domain in enumerate(source.domains))
            cases.append(ExactProblem(2, source.faces, {}, domains))
        for index, problem in enumerate(cases):
            with self.subTest(case=index):
                result = exact_dfs.solve(problem, seconds=2, options=OPTIONS)
                self.assertEqual(result['outcome'], 'solved' if exhaustive(problem) is not None else 'infeasible')
                if result['board'] is not None:
                    self.assertTrue(check_solution(problem, result['board']))

    def test_unproductive_filters_back_off_and_are_sampled_again(self):
        problem = corner_problem(((1, 1),) * 4, edges=())
        search, domains = search_for(problem, {'gac': True, 'adaptive': True})
        for _ in range(32):
            self.assertIsNotNone(search.propagate(domains.copy(), range(4), None))
        policy = search.policy['gac']
        self.assertGreater(policy['calls'], 1)
        self.assertGreater(policy['skips'], policy['calls'])
        self.assertGreater(policy['work'], 0)
        self.assertEqual(search.matching_checks, 32)  # Mandatory feasibility never skipped.
        self.assertEqual(policy['removed'], 0)
        self.assertLessEqual(policy['interval'], 16)

    def test_identical_policy_and_decisions_after_repeated_resume(self):
        for problem in (corner_problem(((1, 1),) * 4), dual_problem()):
            fresh = exact_dfs.solve(problem, seconds=2, seed=17, options=OPTIONS)
            checkpoint = None
            for _ in range(32):
                old_count = 0 if checkpoint is None else checkpoint['counters']['branches']
                stop = [False]
                def observe(snapshot):
                    stop[0] = snapshot['counters']['branches'] > old_count
                result = exact_dfs.solve(problem, seconds=2, seed=17, options=OPTIONS, checkpoint=checkpoint,
                                         should_stop=lambda: stop[0], on_checkpoint=observe, checkpoint_interval=0)
                if result['outcome'] != 'stopped':
                    break
                checkpoint = json.loads(json.dumps(result['checkpoint']))
            else:
                self.fail('Tiny adaptive search did not finish')
            self.assertIsNotNone(checkpoint)
            for key in ('outcome', 'board', 'branches', 'conflicts', 'adaptive_policy'):
                self.assertEqual(result[key], fresh[key], key)

    def test_interrupted_filter_rolls_back_uncommitted_policy(self):
        problem = corner_problem(((1, 1),) * 4)
        fresh = exact_dfs.solve(problem, seconds=2, seed=17, options=OPTIONS)
        original = exact_dfs._Search.check
        def interrupt(search):
            if search.policy['gac']['calls'] > 0:
                raise exact_dfs._Interrupted('stopped')
            return original(search)
        with patch.object(exact_dfs._Search, 'check', interrupt):
            partial = exact_dfs.solve(problem, seconds=2, seed=17, options=OPTIONS)
        self.assertEqual(partial['outcome'], 'stopped')
        self.assertEqual(partial['checkpoint']['adaptive_policy'], exact_dfs._new_policy())
        result = exact_dfs.solve(problem, seconds=2, seed=17, options=OPTIONS, checkpoint=partial['checkpoint'])
        self.assertEqual(result['board'], fresh['board'])
        self.assertEqual(result['adaptive_policy'], fresh['adaptive_policy'])

    def test_wall_clock_does_not_change_policy_or_branch_choices(self):
        problem = corner_problem(((1, 1),) * 4)
        fresh = exact_dfs.solve(problem, seconds=2, seed=7, options=OPTIONS)
        with patch('exact_dfs.time.perf_counter', return_value=100):
            frozen = exact_dfs.solve(problem, seconds=2, seed=7, options=OPTIONS)
        for key in ('board', 'branches', 'adaptive_policy'):
            self.assertEqual(frozen[key], fresh[key])

    def test_rule_diagnostic_history_cannot_change_work_policy(self):
        problem = corner_problem(((1, 1),) * 4)
        policies = []
        for offset in (0, 63, 1023):
            search, domains = search_for(problem, {'adaptive': True, 'extra_rules': {}})
            search.rules.counters['block_pairs'] = offset
            self.assertIsNotNone(search.propagate(domains, range(4), None))
            policies.append(search.policy)
        self.assertEqual(policies[0], policies[1])
        self.assertEqual(policies[0], policies[2])

    def test_policy_schema_and_option_identity_are_checked(self):
        problem = corner_problem()
        cp = exact_dfs.solve(problem, seconds=0, options=OPTIONS)['checkpoint']
        before = json.dumps(cp, sort_keys=True)
        changed = json.loads(before)
        changed['adaptive_policy']['gac']['cooldown'] = 17
        with self.assertRaisesRegex(ValueError, 'scheduling interval'):
            exact_dfs.solve(problem, seconds=0, options=OPTIONS, checkpoint=reseal(changed))
        with self.assertRaisesRegex(ValueError, 'options hash'):
            exact_dfs.solve(problem, seconds=0, options=dict(OPTIONS, adaptive=False), checkpoint=cp)
        with self.assertRaisesRegex(ValueError, 'Unsupported DFS checkpoint'):
            exact_dfs.solve(problem, seconds=0, options=OPTIONS, checkpoint=reseal(dict(cp, version=3)))
        self.assertEqual(json.dumps(cp, sort_keys=True), before)


if __name__ == '__main__':
    unittest.main()
