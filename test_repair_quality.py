"""Controller accounting/validation tests; no GPU, live state or network."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import unittest
from unittest.mock import patch

from exact_model import check_solution, generated_problem
from exact_optimize import evaluate_board
from repair_quality import QualityRepairController
from test_repair_multisource import three_parents


class QualityControllerTests(unittest.TestCase):
    def setUp(self):
        self.problem, _, self.cells, self.parents = three_parents()
        self.controller = QualityRepairController(self.problem)

    def test_bounded_ejection_is_independently_valid_and_counts_work_separately(self):
        reference = self.parents[0]
        answer = self.controller.eject(reference, seed=17)
        checked = evaluate_board(self.problem, answer['board'])
        self.assertTrue(checked['valid'])
        self.assertEqual(answer['score'], checked['score'])
        self.assertGreaterEqual(answer['score'], evaluate_board(self.problem, reference)['score'])
        stats = self.controller.stats()
        self.assertEqual(stats['quality_ejection_calls'], 1)
        self.assertEqual(stats['quality_native_branches'], 0)
        self.assertLessEqual(stats['quality_candidate_evaluations'], 128)
        self.assertLessEqual(stats['quality_path_expansions'], 512)
        self.assertEqual(stats['quality_accepted_piece_changes'], 0)
        self.assertEqual(stats['quality_returned_piece_changes'],
                         sum(a != b for a, b in zip(reference, answer['board'])) if answer['changed'] else 0)
        self.assertEqual(answer['best_true_board'], answer['board'])

    def test_only_explicit_adoption_counts_accepted_changes(self):
        reference = self.parents[0]
        answer = self.controller.eject(reference, seed=17)
        self.assertEqual(self.controller.stats()['quality_accepted_piece_changes'], 0)
        changes = self.controller.record_adoption(reference, answer['board'])
        self.assertEqual(changes, sum(a != b for a, b in zip(reference, answer['board'])))
        self.assertEqual(self.controller.stats()['quality_accepted_piece_changes'], changes)
        before = self.controller.export_state()
        self.assertEqual(self.controller.record_adoption(answer['board'], answer['board']), 0)
        self.assertEqual(self.controller.export_state(), before)
        invalid = answer['board'].copy(); invalid[0] = invalid[1]
        with self.assertRaises(ValueError):
            self.controller.record_adoption(answer['board'], invalid)
        self.assertEqual(self.controller.export_state(), before)

    def test_cancelled_ejection_returns_reference_and_records_no_search_work(self):
        reference = self.parents[0]
        answer = self.controller.eject(reference, seed=1, should_stop=lambda: True)
        self.assertEqual(answer['termination'], 'stopped')
        self.assertEqual(answer['board'], reference)
        self.assertFalse(answer['changed'])
        stats = self.controller.stats()
        self.assertEqual(stats['quality_stopped_calls'], 1)
        for key in ('candidate_evaluations', 'path_expansions', 'heuristic_state_evaluations'):
            self.assertEqual(stats['quality_' + key], 0)

    def test_rejects_false_backend_score_and_illegal_board_without_committing_counts(self):
        reference = self.parents[0]
        raw = self.controller.guidance.ejection_repair(reference, seed=4, max_evaluations=128, max_expansions=512)
        for corrupt in ('score', 'board', 'work'):
            result = deepcopy(raw)
            if corrupt == 'score': result['best_true_score'] += 1
            elif corrupt == 'board': result['best_true_board'][0] = result['best_true_board'][1]
            else: result['candidate_evaluations'] = 129
            before = self.controller.export_state()
            with self.subTest(corrupt=corrupt), patch.object(self.controller.guidance, 'ejection_repair', return_value=result):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.controller.eject(reference, seed=4)
            self.assertEqual(self.controller.export_state(), before)

    def test_strict_state_roundtrip_and_export_detachment(self):
        self.controller.eject(self.parents[0], seed=5)
        state = json.loads(json.dumps(self.controller.export_state()))
        restored = QualityRepairController(self.problem)
        restored.restore_state(state)
        self.assertEqual(restored.stats(), self.controller.stats())
        state['counters']['ejection_calls'] = 900
        self.assertNotEqual(restored.export_state()['counters']['ejection_calls'], 900)
        exported = restored.export_state()
        exported['counters']['candidate_evaluations'] = 100000
        self.assertNotEqual(restored.stats()['quality_candidate_evaluations'], 100000)

    def test_malformed_or_forged_state_is_rejected_atomically(self):
        valid = self.controller.export_state()
        variants = []
        state = deepcopy(valid); state['extra'] = 1; variants.append(state)
        state = deepcopy(valid); state['schema'] = 'unknown'; variants.append(state)
        state = deepcopy(valid); state['problem'] = '0' * 64; variants.append(state)
        state = deepcopy(valid); state['counters']['ejection_calls'] = True; variants.append(state)
        state = deepcopy(valid); state['counters']['path_expansions'] = -1; variants.append(state)
        state = deepcopy(valid); state['counters']['deterministic_work'] = float('nan'); variants.append(state)
        state = deepcopy(valid); state['counters']['deterministic_work'] = float('inf'); variants.append(state)
        state = deepcopy(valid); state['counters']['unknown'] = 0; variants.append(state)
        state = deepcopy(valid); state['counters']['candidate_evaluations'] = 1; variants.append(state)
        state = deepcopy(valid); state['counters']['native_branches'] = 1; variants.append(state)
        state = deepcopy(valid); state['counters']['accepted_piece_changes'] = 1; variants.append(state)
        for state in variants:
            with self.subTest(state=state), self.assertRaises(ValueError):
                self.controller.restore_state(state)
            self.assertEqual(self.controller.export_state(), valid)

    def test_bad_inputs_do_not_change_state(self):
        before = self.controller.export_state()
        for kwargs in ({'seed': True}, {'seed': -1}, {'seed': 4, 'should_stop': 1}):
            with self.assertRaises(ValueError):
                self.controller.eject(self.parents[0], **kwargs)
        with self.assertRaises(ValueError):
            self.controller.eject(self.parents[0][:-1], seed=4)
        with self.assertRaises(ValueError):
            self.controller.recombine(self.parents[:2], seed=4)
        with self.assertRaises(ValueError):
            QualityRepairController(replace(self.problem, edges=()))
        self.assertEqual(self.controller.export_state(), before)

    def test_problem_mapping_is_detached(self):
        expected = dict(self.problem.fixed)
        self.problem.fixed.clear()
        self.assertEqual(self.controller.problem.fixed, expected)
        invalid = self.parents[0].copy()
        cell = next(iter(expected))
        invalid[cell] = 4 * (invalid[cell] // 4) + (invalid[cell] + 1) % 4
        with self.assertRaises(ValueError):
            self.controller.eject(invalid, seed=4)


@unittest.skipUnless(importlib.util.find_spec('ortools'), 'OR-Tools unavailable')
class RecombinationControllerTests(unittest.TestCase):
    def setUp(self):
        self.problem, _, self.cells, self.parents = three_parents()
        self.controller = QualityRepairController(self.problem)

    def test_three_parent_gain_is_valid_and_saved_counters_resume(self):
        answer = self.controller.recombine(self.parents, seed=17, seconds=3)
        self.assertTrue(check_solution(self.problem, answer['board']))
        self.assertEqual(answer['score'], 24)
        self.assertTrue(answer['changed'])
        stats = self.controller.stats()
        self.assertEqual(stats['quality_recombination_calls'], 1)
        self.assertEqual(stats['quality_ejection_calls'], 0)
        self.assertEqual(stats['quality_returned_board_improvements'], 1)
        self.assertEqual(stats['quality_accepted_board_improvements'], 0)
        self.assertEqual(stats['quality_candidate_evaluations'], 0)
        self.assertGreater(stats['quality_table_rows'], 0)
        restored = QualityRepairController(self.problem)
        restored.restore_state(json.loads(json.dumps(self.controller.export_state())))
        self.assertEqual(restored.stats(), stats)

    def test_adoption_into_different_parent_can_change_more_than_returned_proposal(self):
        answer = self.controller.recombine(self.parents, seed=17, seconds=3)
        before = answer['board'].copy()
        for cell in (5, 6, 9, 10):
            if cell not in self.problem.fixed:
                before[cell] = 4 * (before[cell] // 4) + (before[cell] + 1) % 4
        changed = self.controller.record_adoption(before, answer['board'])
        stats = self.controller.stats()
        self.assertEqual(changed, 4)
        self.assertEqual(stats['quality_accepted_piece_changes'], changed)
        self.assertGreater(changed, stats['quality_returned_piece_changes'])
        self.assertEqual(stats['quality_accepted_board_improvements'], 1)
        restored = QualityRepairController(self.problem)
        restored.restore_state(json.loads(json.dumps(self.controller.export_state())))
        self.assertEqual(restored.stats(), stats)

    def test_cancel_and_zero_budget_preserve_best_parent(self):
        best = max(evaluate_board(self.problem, b)['score'] for b in self.parents)
        for kwargs in ({'should_stop': lambda: True}, {'seconds': 0}):
            answer = self.controller.recombine(self.parents, seed=17, **kwargs)
            self.assertEqual(answer['score'], best)
            self.assertFalse(answer['changed'])
            self.assertEqual(answer['branches'], 0)
        self.assertEqual(self.controller.stats()['quality_recombination_calls'], 2)
        self.assertEqual(self.controller.stats()['quality_stopped_calls'], 1)

    def test_false_native_score_or_region_is_rejected_without_counting(self):
        import repair_multisource
        raw = repair_multisource.multi_parent_recombine(self.problem, self.parents, seconds=3)
        for corrupt in ('score', 'region', 'native_work'):
            answer = deepcopy(raw)
            if corrupt == 'score': answer['score'] += 1
            elif corrupt == 'region': answer['region_cells'] = []
            else: answer['deterministic_work'] = float('nan')
            before = self.controller.export_state()
            with self.subTest(corrupt=corrupt), patch('repair_quality.multi_parent_recombine', return_value=answer):
                with self.assertRaises(RuntimeError):
                    self.controller.recombine(self.parents, seed=4)
            self.assertEqual(self.controller.export_state(), before)


if __name__ == '__main__':
    unittest.main()
