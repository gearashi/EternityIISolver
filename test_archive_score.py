"""The streaming archive check must make the same decision as the board validator."""
import random
import unittest

from archive_import import _archive_scorer
from validator import load_bundle, validate_board


class ArchiveScoringTests(unittest.TestCase):
    def test_same_score_and_rejections_as_validator(self):
        bundle = load_bundle()
        score = _archive_scorer(bundle)
        original = list(bundle.record_board)
        examples = [original, original[:-1], original + [0]]
        for invalid in (True, 1.0, '3', None, [], {}, -1, 1024, 10**100):
            edited = original.copy()
            edited[128] = invalid
            examples.append(edited)
        edited = original.copy()
        edited[67] = edited[68]
        examples.append(edited)
        for cell in [*bundle.fixed_clues, 0, 15, 240, 255, 8, 128, 143, 247]:
            edited = original.copy()
            edited[cell] ^= 1
            examples.append(edited)
        randomizer = random.Random(4829)
        inner = [cell for cell in range(256) if 0 < cell % 16 < 15 and 0 < cell // 16 < 15
                 and cell not in bundle.fixed_clues]
        for _ in range(30):
            edited = original.copy()
            cells = randomizer.sample(inner, 12)
            for first, second in zip(cells[::2], cells[1::2]):
                edited[first], edited[second] = edited[second] ^ 1, edited[first]
            examples.append(edited)
        for number, board in enumerate(examples):
            with self.subTest(case=number):
                report = validate_board(board, bundle)
                self.assertEqual(score(board), report['score'] if report['valid'] else None)
