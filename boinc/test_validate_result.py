"""Offline tests for the server-side candidate handoff example."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_result import _object, _nonfinite, canonical_input_sha256, validate_result
from validator import load_bundle


class CandidateValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()
        cls.workunit = json.loads((Path(__file__).with_name('workunit.sample.json')).read_text())

    def candidate(self):
        return {
            'schema': 'eternity-gpu-result/v1',
            'workunit_id': self.workunit['workunit_id'],
            'input_sha256': canonical_input_sha256(self.workunit),
            'pieces_sha256': self.workunit['pieces_sha256'],
            'mode': 'five-clue', 'board': list(self.bundle.record_board),
            'score': 466, 'complete': False, 'exhausted': False,
            'steps_per_replica': self.workunit['steps_per_replica'],
            'completed_steps_per_replica': self.workunit['steps_per_replica'],
            'budget_complete': True, 'status': 'completed',
        }

    def test_finished_search_is_not_a_puzzle_solution(self):
        report = validate_result(self.workunit, self.candidate(), self.bundle)
        self.assertEqual(report['score'], 466)
        self.assertFalse(report['puzzle_solved'])
        self.assertTrue(report['budget_complete'])

    def test_rejects_false_claims_and_incorrect_binding(self):
        for key, value in (
            ('score', 480), ('complete', True), ('exhausted', True),
            ('input_sha256', '0' * 64), ('pieces_sha256', '0' * 64),
            ('workunit_id', 'other-task'), ('mode', 'center-clue'),
            ('status', 'interrupted'), ('budget_complete', False),
            ('steps_per_replica', 17), ('completed_steps_per_replica', 1),
            ('completed_steps_per_replica', True), ('score', True),
        ):
            with self.subTest(field=key, value=value):
                result = self.candidate()
                result[key] = value
                with self.assertRaises(ValueError):
                    validate_result(self.workunit, result, self.bundle)

    def test_rejects_duplicate_piece_and_changed_clue(self):
        result = self.candidate()
        result['board'][0] = result['board'][1]
        with self.assertRaisesRegex(ValueError, 'Illegal candidate'):
            validate_result(self.workunit, result, self.bundle)
        result = self.candidate()
        result['board'][34] ^= 1
        with self.assertRaisesRegex(ValueError, 'Illegal candidate'):
            validate_result(self.workunit, result, self.bundle)

    def test_rejects_unfinished_budget_even_with_matching_flags(self):
        result = self.candidate()
        result['completed_steps_per_replica'] = 1
        result['budget_complete'] = False
        with self.assertRaisesRegex(ValueError, 'did not finish'):
            validate_result(self.workunit, result, self.bundle)

    def test_rejects_invalid_workunit(self):
        for key, value in (('replicas', 31), ('seed', -1), ('steps_per_replica', True),
                           ('pieces_sha256', '0' * 64), ('seconds', 10)):
            with self.subTest(field=key):
                workunit = deepcopy(self.workunit)
                workunit[key] = value
                with self.assertRaises(ValueError):
                    validate_result(workunit, self.candidate(), self.bundle)

    def test_digest_uses_supplied_fields_and_ignores_object_key_order(self):
        reversed_fields = dict(reversed(list(self.workunit.items())))
        self.assertEqual(canonical_input_sha256(self.workunit),
                         canonical_input_sha256(reversed_fields))
        expanded = dict(self.workunit, board=list(self.bundle.record_board))
        self.assertNotEqual(canonical_input_sha256(self.workunit),
                            canonical_input_sha256(expanded))

    def test_parser_rejects_duplicate_keys_and_nonfinite_values(self):
        for raw in ('{"seed": 1, "seed": 2}', '{"seed": NaN}', '{"seed": Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                json.loads(raw, object_pairs_hook=_object, parse_constant=_nonfinite)


if __name__ == '__main__':
    unittest.main()
