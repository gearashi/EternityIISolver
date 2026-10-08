"""Meaningful input, rule and scoring checks; no GPU dependencies."""
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
import uuid
import tempfile

from validator import OFFICIAL_CLUES, load_bundle, piece_taxonomy, validate_board


def naive_score(board, raw_pieces):
    """Separate rotation/scoring implementation, without the orientation table."""
    def edge(cell, direction):
        code = board[cell]
        u, d, l, r = raw_pieces[code // 4]
        source = [u, r, d, l]
        return source[(direction - code % 4) % 4]
    return sum(edge(i, 1) == edge(i+1, 3) for i in range(256) if i % 16 < 15) + sum(edge(i, 2) == edge(i+16, 0) for i in range(240))


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def test_original_466(self):
        report = validate_board(self.bundle.record_board, self.bundle)
        self.assertTrue(report['valid'])
        self.assertFalse(report['complete'])
        self.assertEqual(report['score'], 466)
        self.assertEqual(report['break_count'], 14)
        self.assertEqual(report['breaks'], [[76,92],[93,109],[110,126],[139,155],[141,142],[158,174],[172,173],[173,174],[189,190],[202,203],[204,220],[206,207],[221,222],[236,237]])
        self.assertEqual(naive_score(self.bundle.record_board, self.bundle.pieces_udlr), 466)

    def test_duplicate_piece_rejected(self):
        board = list(self.bundle.record_board)
        board[17] = board[18]
        report = validate_board(board, self.bundle)
        self.assertFalse(report['valid'])
        self.assertFalse(report['pieces_unique'])
        self.assertEqual(len(report['duplicate_piece_ids']), 1)
        self.assertEqual(len(report['missing_piece_ids']), 1)

    def test_each_clue_rotation_rejected(self):
        for cell, state in OFFICIAL_CLUES.items():
            with self.subTest(cell=cell):
                board = list(self.bundle.record_board)
                board[cell] = state // 4 * 4 + (state + 1) % 4
                report = validate_board(board, self.bundle)
                self.assertFalse(report['valid'])
                self.assertFalse(report['clues_valid'])
                self.assertTrue(report['pieces_unique'])

    def test_rotated_frame_rejected(self):
        board = list(self.bundle.record_board)
        board[0] = board[0] // 4 * 4 + (board[0] + 1) % 4
        report = validate_board(board, self.bundle)
        self.assertFalse(report['valid'])
        self.assertFalse(report['frame_valid'])
        self.assertTrue(report['pieces_unique'])
        self.assertTrue(report['clues_valid'])

    def test_100_valid_swaps_and_rotations_agree_with_naive(self):
        rng = random.Random(466)
        cells = [16*r+c for r in range(1,15) for c in range(1,15) if 16*r+c not in OFFICIAL_CLUES]
        for case in range(100):
            board = list(self.bundle.record_board)
            a, b = rng.sample(cells, 2)
            board[a], board[b] = board[b], board[a]
            for cell in (a,b):
                board[cell] = board[cell]//4*4 + rng.randrange(4)
            report = validate_board(board, self.bundle)
            with self.subTest(case=case, a=a, b=b):
                self.assertTrue(report['valid'])
                self.assertEqual(report['score'], naive_score(board, self.bundle.pieces_udlr))

    def test_taxonomy_covers_all_256_pieces(self):
        taxonomy = piece_taxonomy(self.bundle.pieces_udlr)
        self.assertEqual(taxonomy['piece_count'], 256)
        self.assertEqual(taxonomy['counts'], {'corner':4,'edge':56,'interior':196})
        self.assertEqual(taxonomy['palette'], list(range(23)))
        self.assertEqual(taxonomy['color_counts']['0'], 64)
        self.assertEqual(sum(taxonomy['color_counts'].values()), 1024)
        self.assertEqual(len(self.bundle.oriented_edges), 1024)
        self.assertTrue(all(count % 2 == 0 for color, count in taxonomy['color_counts'].items() if color != '0'))

    def test_malformed_states_rejected(self):
        for value in (-1, 1024, 1.0, True, '1', None):
            board = list(self.bundle.record_board)
            board[17] = value
            with self.subTest(value=value):
                self.assertFalse(validate_board(board, self.bundle)['valid'])
        self.assertFalse(validate_board(self.bundle.record_board[:-1], self.bundle)['valid'])

    def test_cli_valid_and_invalid_exit_codes(self):
        script = Path(__file__).resolve().parent/'validator.py'
        temporary = tempfile.TemporaryDirectory(prefix='eternity-validator-')
        path = Path(temporary.name) / 'board.json'
        try:
            path.write_text(json.dumps({'board':self.bundle.record_board}), encoding='utf-8')
            valid = subprocess.run([sys.executable,str(script),str(path)],capture_output=True,text=True)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(json.loads(valid.stdout)['score'], 466)
            board = list(self.bundle.record_board)
            board[17] = board[18]
            path.write_text(json.dumps(board), encoding='utf-8')
            invalid = subprocess.run([sys.executable,str(script),str(path)],capture_output=True,text=True)
            self.assertEqual(invalid.returncode, 2)
            self.assertFalse(json.loads(invalid.stdout)['valid'])
        finally:
            temporary.cleanup()


if __name__ == '__main__':
    unittest.main(verbosity=2)
