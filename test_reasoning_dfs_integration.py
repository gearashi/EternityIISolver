"""Integration contracts for optional DFS reasoning; bounded synthetic fixtures only.

These tests exercise the real helper algorithms. The only injected behavior is
moving cancellation/deadline to a precise boundary after a helper has returned.
No production puzzle, saved runtime, GPU, network, or persistent checkpoint I/O.
"""
import json
import unittest
from unittest.mock import patch

import exact_dfs
import exact_rules
import residual_components
from exact_model import check_solution, generated_problem
from test_residual_components import binary_components


LEGACY_DFS = dict(gac=False, independent_completion=False, dual_branch=False,
                  edge_branch=False, adaptive=False, probe=False, probe_max_candidates=8,
                  probe_max_domain=8, probe_max_depth=8, extra_rules=None)
LEGACY_RULES = dict(hall=True, color_balance=True, directional_balance=True,
                    blocks=True, color_flow=True, balance_work=12000, block_pairs=1024,
                    block_total_pairs=8192, flow_colors=2, flow_max_edges=160)


def joint_options(**overrides):
    # Isolate the new necessary condition from the other optional filters.
    rules = dict(hall=False, color_balance=False, directional_balance=False,
                 blocks=False, color_flow=False, joint_colors=True,
                 joint_width=2, joint_work=12000, joint_states=5000)
    rules.update(overrides)
    return {'extra_rules': rules}


class ReasoningDfsIntegrationTests(unittest.TestCase):
    def test_disabled_extensions_preserve_legacy_normalized_options_and_hash(self):
        self.assertEqual(exact_dfs.normalize_options(), LEGACY_DFS)
        self.assertEqual(exact_rules.normalize_options(), LEGACY_RULES)
        disabled = dict(component_completion=False, component_work=0, component_max_cells=1)
        self.assertEqual(exact_dfs.normalize_options(disabled), LEGACY_DFS)
        disabled_rules = dict(joint_colors=False, joint_width=3, joint_work=0, joint_states=1)
        self.assertEqual(exact_rules.normalize_options(disabled_rules), LEGACY_RULES)
        problem, _ = generated_problem(2, 3, 451)
        for before, after in ((None, disabled),
                              ({'extra_rules': {}}, dict(disabled, extra_rules=disabled_rules))):
            old = exact_dfs.solve(problem, seconds=0, options=before)
            new = exact_dfs.solve(problem, seconds=0, options=after)
            self.assertEqual(old['options_hash'], new['options_hash'])
            # A saved old frontier accepts explicit disabled extension settings.
            resumed = exact_dfs.solve(problem, seconds=2, options=after, checkpoint=old['checkpoint'])
            self.assertEqual(resumed['outcome'], 'solved')
            self.assertTrue(check_solution(problem, resumed['board']))

    def test_enabled_options_are_normalized_and_bound_to_checkpoint(self):
        options = joint_options(joint_width=3, joint_work=91, joint_states=19)
        options.update(component_completion=True, component_work=77, component_max_cells=4)
        normalized = exact_dfs.normalize_options(options)
        for key in ('component_completion', 'component_work', 'component_max_cells'):
            self.assertEqual(normalized[key], options[key])
        for key in ('joint_colors', 'joint_width', 'joint_work', 'joint_states'):
            self.assertEqual(normalized['extra_rules'][key], options['extra_rules'][key])
        self.assertEqual(exact_dfs.normalize_options(normalized), normalized)
        problem, _ = generated_problem(2, 3, 451)
        saved = exact_dfs.solve(problem, seconds=0, options=options)['checkpoint']
        variants = [dict(options, component_completion=False), dict(options, component_work=78),
                    dict(options, extra_rules=dict(options['extra_rules'], joint_width=2)),
                    dict(options, extra_rules=dict(options['extra_rules'], joint_colors=False))]
        for changed in variants:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'options hash mismatch'):
                exact_dfs.solve(problem, seconds=2, options=changed, checkpoint=saved)

    def test_real_joint_checks_preserve_known_solutions(self):
        for width in (2, 3):
            problem, _ = generated_problem(3, 4, 451, clue_count=1)
            result = exact_dfs.solve(problem, seconds=2, seed=7,
                                     options=joint_options(joint_width=width))
            self.assertEqual(result['outcome'], 'solved')
            self.assertTrue(check_solution(problem, result['board']))
            counters = result['rule_counters_session']
            self.assertGreater(counters['joint_checks'], 0)
            self.assertGreater(counters['joint_work'], 0)
            self.assertGreater(counters['joint_setup_work'], 0)

    def test_joint_work_limit_is_unknown_and_does_not_prune(self):
        problem, _ = generated_problem(3, 3, 59, clue_count=1)
        before = (problem.domains, problem.faces, dict(problem.fixed))
        result = exact_dfs.solve(problem, seconds=2, options=joint_options(joint_work=0))
        self.assertEqual(result['outcome'], 'solved')
        self.assertTrue(check_solution(problem, result['board']))
        counters = result['rule_counters_session']
        self.assertGreater(counters['joint_checks'], 0)
        self.assertEqual(counters['joint_skips'], counters['joint_checks'])
        self.assertEqual(counters['joint_failures'], 0)
        self.assertEqual(counters['joint_work'], 0)
        self.assertEqual(before, (problem.domains, problem.faces, problem.fixed))

    def test_actual_component_completion_and_failure_are_integrated(self):
        for negative in (False, True):
            problem, _ = binary_components(negative, groups=2)
            result = exact_dfs.solve(problem, seconds=3, seed=7,
                                    options={'component_completion': True, 'component_work': 100000})
            self.assertEqual(result['outcome'], 'infeasible' if negative else 'solved')
            counters = result['component_counters_session']
            self.assertGreater(counters['calls'], 0)
            self.assertGreater(counters['splits'], 0)
            self.assertGreater(counters['work'], 0)
            self.assertGreater(counters['candidate_placements'], 0)
            self.assertEqual(result['branches'], counters['candidate_placements'])
            self.assertEqual(result['cell_branches'], result['branches'])
            if negative:
                self.assertEqual(counters['failures'], 1)
                self.assertTrue(result['coverage_verified'])
                self.assertIsNone(result['board'])
            else:
                self.assertEqual(counters['completions'], 1)
                self.assertTrue(check_solution(problem, result['board']))

    def test_both_extensions_coexist_on_real_rotation_inventory(self):
        # Actual generated tiles are rotations, unlike the binary CSP fixture.
        problem, _ = generated_problem(3, 3, 59, clue_count=1)
        options = joint_options(joint_work=10000)
        options.update(component_completion=True, component_work=10000)
        result = exact_dfs.solve(problem, seconds=3, options=options)
        self.assertEqual(result['outcome'], 'solved')
        self.assertTrue(check_solution(problem, result['board']))
        self.assertGreater(result['rule_counters_session']['joint_checks'], 0)

    def test_unknown_component_can_stop_and_resume_without_losing_pending_frontier(self):
        real_helper = residual_components.solve_components
        for negative in (False, True):
            problem, _ = binary_components(negative, groups=2)
            options = {'component_completion': True, 'component_work': 0}
            stopped = [False]
            observed = []
            def stop_after_unknown(*args, **kwargs):
                answer = real_helper(*args, **kwargs)
                observed.append(answer)
                stopped[0] = True
                return answer
            with patch('residual_components.solve_components', side_effect=stop_after_unknown):
                first = exact_dfs.solve(problem, seconds=3, seed=7, options=options,
                                        should_stop=lambda: stopped[0])
            self.assertEqual(first['outcome'], 'stopped')
            self.assertEqual(observed[0]['outcome'], 'work_limit')
            self.assertFalse(observed[0]['complete'])
            self.assertEqual(first['checkpoint']['stack'], [{'phase': 'pending'}])
            self.assertFalse(first['coverage_verified'])
            snapshot = json.loads(json.dumps(first['checkpoint']))
            resumed = exact_dfs.solve(problem, seconds=3, seed=7, options=options, checkpoint=snapshot)
            fresh = exact_dfs.solve(problem, seconds=3, seed=7, options=options)
            self.assertEqual(resumed['outcome'], fresh['outcome'])
            self.assertEqual(resumed['board'], fresh['board'])
            self.assertEqual(resumed['branches'], fresh['branches'])
            self.assertTrue(resumed['resumed'])
            self.assertEqual(resumed['proof_scope'], 'remaining_checkpoint_frontier')
            self.assertFalse(resumed['coverage_verified'])
            if not negative:self.assertTrue(check_solution(problem, resumed['board']))

    def test_verified_component_solution_survives_late_stop_or_deadline(self):
        problem, _ = binary_components(False, groups=2)
        real_helper = residual_components.solve_components
        for event in ('stop', 'deadline'):
            with self.subTest(event=event):
                stopped = [False]; now = [100.0]; observed = []
                def finish_then_interrupt(*args, **kwargs):
                    answer = real_helper(*args, **kwargs)
                    self.assertEqual(answer['outcome'], 'solved')
                    self.assertTrue(check_solution(problem, answer['board']))
                    observed.append(answer)
                    if event == 'stop': stopped[0] = True
                    else: now[0] += 5.0
                    return answer
                with patch('exact_dfs.time.perf_counter', side_effect=lambda: now[0]), \
                     patch('residual_components.solve_components', side_effect=finish_then_interrupt):
                    result = exact_dfs.solve(problem, seconds=1, seed=7,
                        options={'component_completion': True, 'component_work': 100000},
                        should_stop=lambda: stopped[0])
                self.assertEqual(len(observed), 1)
                self.assertEqual(result['outcome'], 'solved')
                self.assertTrue(result['solution_verified'])
                self.assertTrue(check_solution(problem, result['board']))
                self.assertIsNone(result['checkpoint'])


if __name__ == '__main__':
    unittest.main()
