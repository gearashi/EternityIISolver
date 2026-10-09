"""Local export validation and roundtrip tests; no GPU or project requests."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import manual_export
from validator import load_bundle, validate_board


class ManualExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()
        cls.reference = json.loads((cls.bundle.data_dir / 'record466.json').read_text())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='eternity-export-')
        self.home = Path(self.temporary.name) / 'state'
        (self.home / 'runtime').mkdir(parents=True)
        self.best = self.home / 'runtime' / 'best.json'

    def tearDown(self):
        self.temporary.cleanup()

    def write_best(self, board=None, score=466):
        self.best.write_text(json.dumps({'board': board if board is not None else list(self.bundle.record_board),
                                        'score': score}), encoding='utf-8')

    def test_reference_fallback_is_explicit_attributed_and_roundtrips(self):
        result = manual_export.export_best(self.home)
        document = json.loads(Path(result['json_path']).read_text())
        self.assertTrue(result['used_reference_fallback'])
        self.assertTrue(result['matches_bundled_reference'])
        self.assertEqual(document['discoverer_name'], 'Jef')
        self.assertIn('not a new result', document['attribution'])
        for key in ('name', 'size', 'score', 'breaks', 'board', 'board_edges', 'board_pieces'):
            self.assertEqual(document[key], self.reference[key])
        self.assertEqual(len(document['board_edges']), 1024)
        self.assertEqual(len(document['board_pieces']), 768)
        report = manual_export.validate_export_document(document)
        self.assertTrue(report['valid']); self.assertTrue(report['text_encodings_valid'])
        self.assertEqual(report['score'], 466)
        self.assertFalse(document['project_acceptance_confirmed'])
        self.assertFalse(document['completed_cpu_workunit'])
        self.assertEqual(document['discovery_status'], 'unverified')
        self.assertFalse(self.best.exists())
        directory = Path(result['directory'])
        self.assertEqual(directory.parent, self.home / 'exports')
        self.assertRegex(directory.name, r'^[A-Za-z0-9_-]+$')
        self.assertEqual({p.name for p in directory.iterdir()}, {'board.json', 'layout.txt', 'validation.json'})

    def test_runtime_best_export_keeps_runtime_files_unchanged(self):
        board = list(self.bundle.record_board)
        board[17], board[18] = board[18], board[17]
        score = validate_board(board, self.bundle)['score']
        self.write_best(board, score)
        for name in ('STOP', 'checkpoint.npz'):
            (self.home / 'runtime' / name).write_bytes(b'preserve exact bytes')
        before = {p.name: p.read_bytes() for p in (self.home / 'runtime').iterdir()}
        result = manual_export.export_best(self.home)
        document = json.loads(Path(result['json_path']).read_text())
        self.assertEqual(document['board'], board)
        self.assertEqual(document['score'], score)
        self.assertFalse(result['used_reference_fallback'])
        self.assertFalse(result['matches_bundled_reference'])
        self.assertNotIn('discoverer_name', document)
        self.assertEqual({p.name: p.read_bytes() for p in (self.home / 'runtime').iterdir()}, before)
        checked = manual_export.validate_export_document(document)
        self.assertEqual(checked['score'], score)
        # Reconstruct exact placement codes using the human-readable layout.
        lines = Path(result['layout_path']).read_text().splitlines()
        heading = lines.index('row column piece_id clockwise_quarter_turns placement_code')
        rows = [list(map(int, line.split())) for line in lines[heading + 1:]]
        self.assertEqual(len(rows), 256)
        rebuilt = []
        for cell, (row, column, piece, rotation, code) in enumerate(rows):
            self.assertEqual((row, column), (cell // 16 + 1, cell % 16 + 1))
            self.assertEqual(code, 4 * (piece - 1) + rotation)
            rebuilt.append(code)
        self.assertEqual(rebuilt, board)

    def test_invalid_score_or_board_creates_no_export(self):
        broken = list(self.bundle.record_board); broken[17] = broken[18]
        cases = [(list(self.bundle.record_board), 480), (broken, 466), (list(self.bundle.record_board), True)]
        output = Path(self.temporary.name) / 'invalid-exports'
        for board, score in cases:
            with self.subTest(score=score):
                self.write_best(board, score)
                with self.assertRaises(manual_export.ExportError):
                    manual_export.export_best(self.home, output)
                self.assertFalse(output.exists())
                self.assertFalse((self.home / 'exports').exists())

    def test_strict_bounded_input_rejects_duplicate_and_nonfinite(self):
        for text in ('{"score":466,"score":466}', '{"value":NaN}', '{"value":1e999}',
                     ' ' * (manual_export.MAX_JSON_BYTES + 1)):
            with self.subTest(text=text[:40]):
                self.best.write_text(text, encoding='utf-8')
                with self.assertRaises(manual_export.ExportError):
                    manual_export.export_best(self.home)
                self.assertFalse((self.home / 'exports').exists())

    def test_export_validator_rejects_tampered_encodings_and_hash(self):
        document = json.loads(Path(manual_export.export_best(self.home)['json_path']).read_text())
        for changes in ({'board_edges': 'a' * 1024}, {'board_pieces': '001' * 256}, {'breaks': 0},
                        {'local_board_sha256_uint16le': '0' * 64}, {'pieces_sha256': '0' * 64},
                        {'orientation': 'rotated-180'}):
            with self.subTest(changes=list(changes)), self.assertRaises(manual_export.ExportError):
                manual_export.validate_export_document({**document, **changes})

    def test_palette_comes_from_reference_not_numeric_color_offsets(self):
        _, palette = manual_export._reference_palette(self.bundle)
        self.assertTrue(any(palette[color] != chr(97 + color) for color in range(23)))
        self.assertEqual(set(palette.values()), set('abcdefghijklmnopqrstuvw'))
        self.assertEqual(palette[0], 'a')

    def test_repeated_exports_are_unique_and_never_connect(self):
        output = Path(self.temporary.name) / 'manual'
        with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')) as connect, \
             patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')) as urlopen:
            first = manual_export.export_best(self.home, output)
            second = manual_export.export_best(self.home, output)
            connect.assert_not_called(); urlopen.assert_not_called()
        self.assertNotEqual(first['directory'], second['directory'])
        self.assertEqual(first['local_board_sha256_uint16le'], second['local_board_sha256_uint16le'])
        self.assertEqual(Path(first['directory']).parent, output)

    def test_cli_returns_paths_to_local_files(self):
        with patch('sys.stdout', new_callable=io.StringIO) as output:
            code = manual_export.main(['--state-dir', str(self.home)])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result['exported'])
        for key in ('json_path', 'layout_path', 'validation_path'):
            self.assertTrue(Path(result[key]).is_file())


if __name__ == '__main__':
    unittest.main()
