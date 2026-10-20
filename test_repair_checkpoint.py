"""Finite checkpoint invariants; isolated files, no production search or network."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from exact_model import generated_problem, grid_edges
from repair_checkpoint import (CheckpointCodec, MAX_BYTES, MAX_SEEN,
                               load_checkpoint, neighborhood_key, restore_rng_state)


def initial_state(problem, board, seed=17):
    score = sum(problem.faces[board[a]][sa] == problem.faces[board[b]][sb]
                for a, b, sa, sb in grid_edges(problem.size))
    return {'best': list(board), 'best_score': score, 'bank': [list(board)],
            'bank_scores': [score], 'peaks': [score], 'stalls': [0],
            'recent': [[list(board)]], 'rng_state': random.Random(seed).getstate(),
            'completed': 0, 'cycle_calls': 0, 'rounds': 0, 'batches': 0,
            'counts': {'diversification_attempts': 0, 'diversifications': 0,
                       'tabu_rejections': 0, 'region_calls': 0,
                       'region_improvements': 0, 'region_seconds': 0.0},
            'seen': [], 'elapsed_total': 0.0, 'policy_state': {'regions': None}}


def resign(payload):
    payload = copy.deepcopy(payload)
    payload.pop('checksum', None)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    payload['checksum'] = hashlib.sha256(encoded.encode()).hexdigest()
    return payload


class RepairCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.problem, self.board = generated_problem(3, 4, 71, 1)
        self.settings = {'seed': 17, 'gpu_backend': None, 'gpu_batch_size': 16,
                         'neutral_ties': False, 'adaptive_regions': False}
        self.codec = CheckpointCodec(self.problem, self.settings)
        self.state = initial_state(self.problem, self.board)

    def test_json_roundtrip_retains_random_continuation_and_is_detached(self):
        rng = random.Random(17)
        for _ in range(25): rng.randrange(2**32)
        self.state['rng_state'] = rng.getstate()
        payload = self.codec.encode(self.state)
        decoded = self.codec.decode(json.loads(json.dumps(payload)))
        resumed = random.Random(); resumed.setstate(restore_rng_state(decoded['rng_state']))
        self.assertEqual([rng.getrandbits(32) for _ in range(100)],
                         [resumed.getrandbits(32) for _ in range(100)])
        decoded['bank'][0][0] = -1
        self.assertEqual(payload['state']['bank'][0], self.board)
        self.assertEqual(self.state['bank'][0], self.board)

    def test_checksum_and_algorithm_revision_rejected(self):
        payload = self.codec.encode(self.state)
        payload['state']['rounds'] += 1
        with self.assertRaisesRegex(ValueError, 'checksum'): self.codec.decode(payload)
        payload = self.codec.encode(self.state); payload['algorithm'] = 'other-algorithm'
        with self.assertRaisesRegex(ValueError, 'algorithm/version'): self.codec.decode(resign(payload))
        payload = self.codec.encode(self.state); payload['schema_version'] = True
        with self.assertRaises(ValueError): self.codec.decode(resign(payload))

    def test_settings_and_rules_fingerprint_prevent_false_resume(self):
        payload = self.codec.encode(self.state)
        for options in ({'seed':18}, {'gpu_backend':'cuda'}, {'neutral_ties':True}, {'gpu_batch_size':8}):
            other = CheckpointCodec(self.problem, dict(self.settings, **options))
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'settings mismatch'):
                other.decode(payload)
        different, _ = generated_problem(3, 4, 72, 1)
        with self.assertRaisesRegex(ValueError, 'settings mismatch'):
            CheckpointCodec(different, self.settings).decode(payload)

    def test_incorrect_scores_cannot_be_forged_with_valid_checksum(self):
        for field in ('best_score', 'bank_scores', 'peaks'):
            payload = self.codec.encode(self.state)
            if field == 'best_score': payload['state'][field] -= 1
            else: payload['state'][field][0] -= 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.codec.decode(resign(payload))

    def test_restored_boards_preserve_pieces_frame_and_fixed_orientation(self):
        for kind in ('duplicate', 'frame', 'fixed', 'bool'):
            payload = self.codec.encode(self.state)
            board = payload['state']['bank'][0]
            if kind == 'duplicate': board[0] = board[1]
            elif kind == 'frame': board[0] = 4 * (board[0]//4) + ((board[0]+1)%4)
            elif kind == 'fixed':
                cell = next(iter(self.problem.fixed)); board[cell] = (board[cell]//4)*4 + ((board[cell]+1)%4)
            else: board[0] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.codec.decode(resign(payload))

    def test_every_recent_board_is_validated(self):
        payload = self.codec.encode(self.state)
        payload['state']['recent'][0].append([False] * 9)
        with self.assertRaises(ValueError): self.codec.decode(resign(payload))

    def test_counter_consistency_and_typed_bounds(self):
        changes = [('completed', 1), ('batches', True), ('rounds', -1),
                   ('elapsed_total', float('inf')), ('policy_state', []), ('unknown', 0)]
        for key, value in changes:
            state = copy.deepcopy(self.state); state[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.codec.encode(state)
        for key in ('diversifications', 'region_improvements', 'region_seconds'):
            state = copy.deepcopy(self.state); state['counts'][key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError): self.codec.encode(state)

    def test_completed_neighborhood_digest_bound_and_tie_identity(self):
        cells = [0, 2]
        plain = neighborhood_key(self.board, cells)
        tied = neighborhood_key(self.board, cells, 7)
        self.assertNotEqual(plain, tied)
        self.assertNotEqual(tied, neighborhood_key(self.board, cells, 8))
        self.assertEqual(tied, neighborhood_key(tuple(self.board), tuple(cells), 7))
        self.state.update(completed=1, rounds=1, batches=1, seen=[tied])
        self.assertEqual(self.codec.decode(self.codec.encode(self.state))['seen'], [tied])
        self.state['seen'] = [tied, tied]; self.state.update(completed=2, rounds=2)
        with self.assertRaises(ValueError): self.codec.encode(self.state)
        self.state['seen'] = ['x' * 64]
        with self.assertRaises(ValueError): self.codec.encode(self.state)
        self.state['seen'] = ['0' * 64] * (MAX_SEEN + 1)
        with self.assertRaises(ValueError): self.codec.encode(self.state)

    def test_digest_rejects_invalid_or_boolean_cells_and_seeds(self):
        for cells, seed in (([0, 0], None), ([True], None), ([99], None), ([0], True), ([0], -1)):
            with self.subTest(cells=cells, seed=seed), self.assertRaises(ValueError):
                neighborhood_key(self.board, cells, seed)

    def test_rng_rejects_unsafe_or_unsupported_state(self):
        original = list(random.Random(17).getstate()); original[1] = list(original[1])
        variants = [[2, original[1], None], [True, original[1], None],
                    [3, original[1][:-1], None], [3, original[1], float('nan')]]
        wrong = copy.deepcopy(original); wrong[1][-1] = 625; variants.append(wrong)
        wrong = copy.deepcopy(original); wrong[1][0] = True; variants.append(wrong)
        for value in variants:
            with self.assertRaises(ValueError): restore_rng_state(value)

    def test_input_size_and_recursive_policy_data_are_rejected(self):
        cyclic = {}; cyclic['self'] = cyclic
        state = copy.deepcopy(self.state); state['policy_state'] = cyclic
        with self.assertRaises(ValueError): self.codec.encode(state)
        state = copy.deepcopy(self.state); state['policy_state'] = {'x':'z' * 4097}
        with self.assertRaises(ValueError): self.codec.encode(state)

    def test_file_reader_rejects_duplicate_keys_constants_and_oversize(self):
        with tempfile.TemporaryDirectory(prefix='eternity-checkpoint-reader-') as temporary:
            path = Path(temporary) / 'state.json'
            valid = self.codec.encode(self.state)
            path.write_text(json.dumps(valid))
            self.assertEqual(load_checkpoint(path), valid)
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"broken":'):
                path.write_text(text)
                with self.assertRaises(ValueError): load_checkpoint(path)
            path.write_bytes(b' ' * (MAX_BYTES + 1))
            with self.assertRaisesRegex(ValueError, 'byte limit'): load_checkpoint(path)

    def test_mutated_problem_refuses_checkpoint(self):
        self.problem.fixed[0] = self.board[0]
        with self.assertRaisesRegex(ValueError, 'puzzle changed'): self.codec.encode(self.state)


if __name__ == '__main__': unittest.main()
