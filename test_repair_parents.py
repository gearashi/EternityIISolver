"""Finite, deterministic tests for read-only diverse cache parent selection."""
from contextlib import closing
from dataclasses import replace
import random
import sqlite3
import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from exact_model import check_solution, generated_problem, grid_edges
from repair_parents import _distance, _features, cached_parents


def shuffled(problem, board, seed):
    rng = random.Random(seed)
    result = list(board)
    groups = {}
    for cell in range(len(board)):
        if cell not in problem.fixed:
            groups.setdefault(sum(color == 0 for color in problem.faces[board[cell]]), []).append(cell)
    for cells in groups.values():
        pieces = [board[cell] // 4 for cell in cells]
        rng.shuffle(pieces)
        for cell, piece in zip(cells, pieces):
            result[cell] = rng.choice([state for state in problem.domains[cell] if state // 4 == piece])
    return result


def actual_score(problem, board):
    return sum(problem.faces[board[a]][sa] == problem.faces[board[b]][sb]
               for a, b, sa, sb in grid_edges(problem.size))


def database(path, rows):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE boards(public_sha TEXT PRIMARY KEY,score INTEGER,placement BLOB)')
        for i, (score, board) in enumerate(rows):
            blob = board if isinstance(board, (bytes, str)) or board is None else struct.pack('<' + str(len(board)) + 'H', *board)
            db.execute('INSERT INTO boards VALUES(?,?,?)', (str(i), score, blob))


class RepairParentTests(unittest.TestCase):
    def test_missing_file_and_zero_budget_never_create_database(self):
        problem, _ = generated_problem(3, 4, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'absent.sqlite3'
            self.assertEqual(cached_parents(problem, path), [])
            self.assertEqual(cached_parents(problem, path, max_seconds=0), [])
            self.assertFalse(path.exists())

    def test_validation_dedup_independent_score_and_read_only(self):
        problem, answer = generated_problem(3, 4, 2, clue_count=1)
        alternative = shuffled(problem, answer, 6)
        duplicate_piece = list(answer)
        duplicate_piece[0] = duplicate_piece[1]
        out_of_range = list(answer)
        out_of_range[0] = 65535
        bad_clue = list(answer)
        clue = next(iter(problem.fixed))
        bad_clue[clue] = 4 * (bad_clue[clue] // 4) + (bad_clue[clue] + 1) % 4
        invalid_frame = list(answer)
        invalid_frame[0] = 4 * (invalid_frame[0] // 4) + (invalid_frame[0] + 1) % 4
        rows = [(12, b'bad'), (12, 'not bytes'), (12, duplicate_piece),
                (12, out_of_range), (12, invalid_frame), (12, bad_clue),
                (12, alternative), (0, answer), (12, answer), (12, None)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.sqlite3'
            database(path, rows)
            before = path.read_bytes()
            files = sorted(p.name for p in path.parent.iterdir())
            result = cached_parents(problem, path)
            self.assertEqual(result[0], answer)
            self.assertEqual({tuple(board) for board in result}, {tuple(answer), tuple(alternative)})
            self.assertEqual(len(result), 2)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), files)
            self.assertTrue(all(check_solution(replace(problem, edges=()), board) for board in result))

    def test_seeded_selection_is_reproducible_and_preserves_best_sample(self):
        problem, answer = generated_problem(4, 5, 3, clue_count=1)
        boards = [answer] + [shuffled(problem, answer, seed) for seed in range(60)]
        rows = [(actual_score(problem, board), board) for board in boards]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.sqlite3'
            database(path, rows)
            first = cached_parents(problem, path, seed=17, limit=8, candidate_limit=32)
            second = cached_parents(problem, path, seed=17, limit=8, candidate_limit=32)
            third = cached_parents(problem, path, seed=18, limit=8, candidate_limit=32)
            self.assertEqual(first, second)
            self.assertNotEqual(first, third)
            self.assertEqual(first[0], answer)
            self.assertEqual(len(first), 8)
            self.assertEqual(len(set(map(tuple, first))), 8)

    def test_diverse_choices_match_independent_farthest_first_oracle(self):
        problem, answer = generated_problem(4, 4, 9, clue_count=1)
        boards = [answer] + [shuffled(problem, answer, seed) for seed in range(15)]
        rows = [(actual_score(problem, board), board) for board in boards]
        edges = grid_edges(problem.size)
        def distance(left, right):
            pieces = sum(a // 4 != b // 4 for a, b in zip(left, right))
            states = sum(a != b for a, b in zip(left, right))
            lm = {i for i, (a, b, sa, sb) in enumerate(edges)
                  if problem.faces[left[a]][sa] != problem.faces[left[b]][sb]}
            rm = {i for i, (a, b, sa, sb) in enumerate(edges)
                  if problem.faces[right[a]][sa] != problem.faces[right[b]][sb]}
            mismatch = 0 if not lm | rm else 400 * len(lm ^ rm) // len(lm | rm)
            return 400 * pieces // len(left) + 200 * states // len(left) + mismatch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.sqlite3'
            database(path, rows)
            # All rows fit, and limit=4 reserves exactly one elite.
            selected = cached_parents(problem, path, limit=4)
        self.assertEqual(selected[0], answer)
        remaining = {tuple(board) for board in boards}
        remaining.remove(tuple(answer))
        for i, board in enumerate(selected[1:], 1):
            expected_distance = max(min(distance(candidate, prior) for prior in selected[:i])
                                    for candidate in remaining)
            self.assertEqual(min(distance(board, prior) for prior in selected[:i]), expected_distance)
            remaining.remove(tuple(board))

    def test_feature_distance_accounts_for_rotation_and_mismatch_locations(self):
        problem, answer = generated_problem(4, 3, 5)
        alternative = shuffled(problem, answer, 7)
        edges = grid_edges(problem.size)
        a = _features(problem, answer, edges)
        b = _features(problem, alternative, edges)
        self.assertEqual(_distance(a, a, 16), 0)
        self.assertEqual(_distance(a, b, 16), _distance(b, a, 16))
        self.assertGreater(_distance(a, b, 16), 0)
        # Same placements/pieces isolate the mismatch-location term.
        self.assertEqual(_distance((a[0], a[1], 1), (a[0], a[1], 2), 16), 400)
        # Changing an interior orientation leaves its physical piece in place.
        rotated = list(answer)
        rotated[5] = 4 * (rotated[5] // 4) + (rotated[5] + 1) % 4
        rf = _features(problem, rotated, edges)
        self.assertEqual(a[0], rf[0])
        self.assertNotEqual(a[1], rf[1])

    def test_bounded_blob_fetch_and_all_score_bands_are_sampled(self):
        problem, answer = generated_problem(4, 4, 2)
        boards = [shuffled(problem, answer, i) for i in range(96)]
        # Misleading metadata is deliberate: it guides sampling, not scoring.
        metadata = [24] * 80 + [20] * 8 + [8] * 8
        rows = list(zip(metadata, boards))
        queries = []
        fetched = []
        original = sqlite3.connect
        class RecordingConnection(sqlite3.Connection):
            def execute(self, sql, parameters=()):
                queries.append((sql, parameters))
                if sql.startswith('SELECT placement'):
                    fetched.append(parameters[0] - 1)
                return super().execute(sql, parameters)
        def connect(*args, **kwargs):
            self.assertIn('mode=ro', args[0])
            kwargs['factory'] = RecordingConnection
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.sqlite3'
            database(path, rows)
            with patch('repair_parents.sqlite3.connect', side_effect=connect):
                result = cached_parents(problem, path, limit=4, candidate_limit=16)
        self.assertEqual(len(fetched), 16)
        self.assertTrue(any(metadata[i] == 20 for i in fetched))
        self.assertTrue(any(metadata[i] == 8 for i in fetched))
        self.assertTrue(all('RANDOM(' not in query.upper() for query, _ in queries))
        self.assertTrue(all('WHERE rowid=?' in query for query, _ in queries if query.startswith('SELECT placement')))
        self.assertEqual(len(result), 4)

    def test_stop_and_deadline_are_prompt_and_return_only_valid_boards(self):
        problem, answer = generated_problem(3, 4, 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.sqlite3'
            database(path, [(12, answer)] * 100)
            self.assertEqual(cached_parents(problem, path, should_stop=lambda: True), [])
            started = time.monotonic()
            result = cached_parents(problem, path, max_seconds=.000001)
            self.assertLess(time.monotonic() - started, .5)
            self.assertTrue(all(check_solution(replace(problem, edges=()), board) for board in result))

    def test_callback_failure_and_sql_corruption_are_not_silently_hidden(self):
        problem, answer = generated_problem(3, 4, 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.sqlite3'
            database(path, [(12, answer)])
            def broken():
                raise RuntimeError('cancel failure')
            with self.assertRaisesRegex(RuntimeError, 'cancel failure'):
                cached_parents(problem, path, should_stop=broken)
            path.write_bytes(b'not a sqlite database')
            with self.assertRaises(sqlite3.DatabaseError):
                cached_parents(problem, path)


    def test_cancellation_after_validation_preserves_already_validated_candidates(self):
        problem, answer = generated_problem(4, 4, 4)
        boards = [answer] + [shuffled(problem, answer, seed) for seed in range(12)]
        rows = [(actual_score(problem, board), board) for board in boards]
        validated = []
        original = check_solution
        def validate(model, board):
            result = original(model, board)
            if result:
                validated.append(tuple(board))
            return result
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.sqlite3'
            database(path, rows)
            with patch('repair_parents.check_solution', side_effect=validate):
                result = cached_parents(problem, path, limit=8,
                                        should_stop=lambda: len(validated) >= 3)
        self.assertEqual(len(validated), 3)
        self.assertEqual(set(map(tuple, result)), set(validated))

    def test_sql_progress_handler_can_cancel_metadata_scan(self):
        problem, answer = generated_problem(3, 4, 0)
        calls = []
        def stop():
            calls.append(1)
            return len(calls) >= 3
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.sqlite3'
            database(path, [(12, answer)] * 1000)
            before = path.read_bytes()
            self.assertEqual(cached_parents(problem, path, should_stop=stop), [])
            self.assertEqual(path.read_bytes(), before)
        self.assertLessEqual(len(calls), 4)


    def test_scan_deadline_preserves_time_for_validation_and_global_timeout_keeps_elites(self):
        problem, answer = generated_problem(4, 4, 17, clue_count=1)
        choices = [answer] + [shuffled(problem, answer, seed) for seed in range(40)]
        rows = [(actual_score(problem, choices[i % len(choices)]),
                 choices[i % len(choices)]) for i in range(600)]
        original = sqlite3.connect
        for jump, full_timeout in ((3., False), (6., True)):
            clock = [0.]
            scanned = []
            point_reads = []
            class SlowMetadata:
                def __init__(self, cursor):
                    self.cursor = cursor
                def __iter__(self):
                    for row in self.cursor:
                        scanned.append(row[0])
                        if len(scanned) == 4:
                            clock[0] = jump
                        yield row
            class Connection(sqlite3.Connection):
                def execute(self, sql, parameters=()):
                    cursor = super().execute(sql, parameters)
                    if sql.endswith('WHERE placement IS NOT NULL ORDER BY rowid'):
                        return SlowMetadata(cursor)
                    if sql.startswith('SELECT placement'):
                        point_reads.append((parameters[0], clock[0]))
                    return cursor
            def connect(*args, **kwargs):
                kwargs['factory'] = Connection
                return original(*args, **kwargs)
            with self.subTest(full_timeout=full_timeout), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'cache.sqlite3'
                database(path, rows)
                before = path.read_bytes()
                with patch('repair_parents.sqlite3.connect', side_effect=connect), \
                     patch('repair_parents.time.monotonic', side_effect=lambda: clock[0]):
                    selected = cached_parents(problem, path, limit=8,
                                              candidate_limit=32, max_seconds=5)
                self.assertTrue(selected)
                self.assertEqual(selected[0], answer)
                self.assertLess(len(scanned), len(rows))
                self.assertLessEqual(len(point_reads), 32)
                self.assertTrue(all(check_solution(replace(problem, edges=()), board) for board in selected))
                self.assertEqual(path.read_bytes(), before)
                if full_timeout:
                    self.assertTrue(all(read_time == 0 for _, read_time in point_reads))
                    self.assertLessEqual(len(selected), 2)
                else:
                    self.assertTrue(any(read_time == 3 for _, read_time in point_reads))

    def test_argument_validation(self):
        problem, _ = generated_problem(3, 4, 0)
        cases = [{'limit': 0}, {'limit': True}, {'limit': 257},
                 {'seed': -1}, {'seed': True}, {'seed': 2**32},
                 {'candidate_limit': True}, {'candidate_limit': 15},
                 {'candidate_limit': 1025}, {'candidate_limit': 16, 'limit': 32},
                 {'max_seconds': float('inf')}, {'max_seconds': float('nan')},
                 {'max_seconds': -1}, {'max_seconds': True}, {'should_stop': 42}]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                cached_parents(problem, 'missing.sqlite3', **options)


if __name__ == '__main__':
    unittest.main()
