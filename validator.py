"""Dependency-free validator for the official 16x16 Eternity II puzzle.

Board entries are state = 4 * (one-based piece ID - 1) + clockwise rotation.
The source piece file is U,D,L,R; oriented_edges is U,R,D,L.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable

SIZE = 16
N_PIECES = SIZE * SIZE
N_EDGES = 2 * SIZE * (SIZE - 1)
# Official five-clue locations/orientations in the supplied piece convention.
OFFICIAL_CLUES = {34: 831, 45: 1019, 135: 554, 210: 723, 221: 992}
from app_paths import resource_root
DEFAULT_DATA_DIR = resource_root() / 'data'


@dataclass(frozen=True)
class PuzzleBundle:
    pieces_udlr: tuple[tuple[int, int, int, int], ...]
    oriented_edges: tuple[tuple[int, int, int, int], ...]
    record_board: tuple[int, ...]
    fixed_clues: dict[int, int]
    metadata: dict[str, Any]
    data_dir: Path


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pieces(path: str | Path) -> tuple[tuple[int, int, int, int], ...]:
    pieces = []
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = tuple(int(value) for value in line.split())
        except ValueError as exc:
            raise ValueError(f'Non-integer piece color at line {line_number}') from exc
        if len(values) != 4 or any(value < 0 or value > 22 for value in values):
            raise ValueError(f'Invalid U,D,L,R piece at line {line_number}: {values}')
        pieces.append(values)
    if len(pieces) != N_PIECES:
        raise ValueError(f'Expected {N_PIECES} pieces, found {len(pieces)}')
    return tuple(pieces)


def piece_taxonomy(pieces: Iterable[tuple[int, int, int, int]]) -> dict[str, Any]:
    kinds = []
    colors: Counter[int] = Counter()
    for piece_id, (u, d, l, r) in enumerate(pieces, 1):
        sides = (u, r, d, l)
        zeros = [i for i, color in enumerate(sides) if color == 0]
        if len(zeros) == 0:
            kind = 'interior'
        elif len(zeros) == 1:
            kind = 'edge'
        elif len(zeros) == 2 and ((zeros[1] - zeros[0]) % 4 in (1, 3)):
            kind = 'corner'
        else:
            raise ValueError(f'Piece {piece_id} has invalid gray-side geometry')
        kinds.append(kind)
        colors.update(sides)
    return {
        'piece_count': len(kinds),
        'counts': dict(Counter(kinds)),
        'piece_kinds': kinds,
        'palette': sorted(colors),
        'color_counts': {str(color): colors[color] for color in sorted(colors)},
    }


def make_oriented_edges(pieces: Iterable[tuple[int, int, int, int]]) -> tuple[tuple[int, int, int, int], ...]:
    states = []
    for u, d, l, r in pieces:
        states.extend(((u, r, d, l), (l, u, r, d), (d, l, u, r), (r, d, l, u)))
    return tuple(states)


def board_from_json(value: Any) -> Any:
    if isinstance(value, dict):
        if 'board' in value:
            return value['board']
        if 'best_board' in value:
            return value['best_board']
        raise ValueError('Board JSON object must contain "board" or "best_board"')
    if isinstance(value, list):
        return value
    raise ValueError('Board JSON must be an array or a board object')


def load_bundle(data_dir: str | Path | None = None) -> PuzzleBundle:
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    metadata = json.loads((directory / 'puzzle.json').read_text(encoding='utf-8'))
    if metadata.get('size') != SIZE:
        raise ValueError('This validator requires the official 16x16 puzzle')
    for name in ('pieces.txt', 'record466.json'):
        expected = metadata['input_sha256'][name]
        if sha256(directory / name) != expected:
            raise ValueError(f'Input SHA-256 mismatch: {name}')
    pieces = read_pieces(directory / 'pieces.txt')
    taxonomy = piece_taxonomy(pieces)
    if taxonomy['counts'] != {'corner': 4, 'edge': 56, 'interior': 196}:
        raise ValueError('Expected four corners, 56 edge pieces, and 196 interior pieces')
    if taxonomy['palette'] != list(range(23)):
        raise ValueError('Expected gray 0 and all 22 non-gray colors')
    clues = {int(item['cell']): int(item['state']) for item in metadata['fixed_clues']}
    if clues != OFFICIAL_CLUES:
        raise ValueError('Clue metadata does not match the official five fixed states')
    record = json.loads((directory / 'record466.json').read_text(encoding='utf-8'))
    record_board = tuple(board_from_json(record))
    bundle = PuzzleBundle(pieces, make_oriented_edges(pieces), record_board, clues, metadata, directory)
    report = validate_board(record_board, bundle)
    if not report['valid'] or report['score'] != 466:
        raise ValueError(f'Reference record failed validation: {report}')
    return bundle


def validate_board(board: Iterable[int], bundle: PuzzleBundle | None = None) -> dict[str, Any]:
    bundle = bundle if bundle is not None else load_bundle()
    result: dict[str, Any] = {
        'valid': False, 'complete': False, 'score': None, 'break_count': None,
        'breaks': [], 'pieces_unique': False, 'frame_valid': False,
        'clues_valid': False, 'length_valid': False, 'states_valid': False,
        'duplicate_piece_ids': [], 'missing_piece_ids': [],
        'frame_errors': [], 'clue_errors': [], 'errors': [],
    }
    try:
        values = list(board)
    except TypeError:
        result['errors'].append('Board must be an iterable of integer states')
        return result
    if len(values) != N_PIECES:
        result['errors'].append(f'Expected {N_PIECES} cells, found {len(values)}')
        return result
    result['length_valid'] = True
    bad_states = [i for i, state in enumerate(values)
                  if isinstance(state, bool) or not isinstance(state, Integral) or not 0 <= state < 1024]
    if bad_states:
        result['errors'].append(f'Invalid integer states at zero-based cells {bad_states}')
        return result
    values = [int(state) for state in values]
    result['states_valid'] = True
    counts = Counter(state // 4 + 1 for state in values)
    result['duplicate_piece_ids'] = sorted(piece for piece, count in counts.items() if count != 1)
    result['missing_piece_ids'] = sorted(set(range(1, 257)) - set(counts))
    result['pieces_unique'] = not result['duplicate_piece_ids'] and not result['missing_piece_ids']
    if not result['pieces_unique']:
        result['errors'].append('Every physical piece must appear exactly once')
    side_names = ('U', 'R', 'D', 'L')
    oriented = [bundle.oriented_edges[state] for state in values]
    for cell, sides in enumerate(oriented):
        exterior = (cell < 16, cell % 16 == 15, cell >= 240, cell % 16 == 0)
        for direction, (color, outside) in enumerate(zip(sides, exterior)):
            if (color == 0) != outside:
                result['frame_errors'].append({'cell': cell, 'side': side_names[direction], 'color': color, 'exterior': outside})
    result['frame_valid'] = not result['frame_errors']
    if not result['frame_valid']:
        result['errors'].append('Gray edges must occur exactly on the outside frame')
    for cell, expected in bundle.fixed_clues.items():
        if values[cell] != expected:
            result['clue_errors'].append({'cell': cell, 'expected_state': expected, 'actual_state': values[cell]})
    result['clues_valid'] = not result['clue_errors']
    if not result['clues_valid']:
        result['errors'].append('All five clue pieces and orientations must remain fixed')
    for row in range(SIZE):
        for column in range(SIZE):
            cell = row * SIZE + column
            if column < SIZE - 1 and oriented[cell][1] != oriented[cell + 1][3]:
                result['breaks'].append([cell, cell + 1])
            if row < SIZE - 1 and oriented[cell][2] != oriented[cell + SIZE][0]:
                result['breaks'].append([cell, cell + SIZE])
    result['break_count'] = len(result['breaks'])
    result['score'] = N_EDGES - result['break_count']
    result['valid'] = result['pieces_unique'] and result['frame_valid'] and result['clues_valid']
    result['complete'] = result['valid'] and result['score'] == N_EDGES
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('board_json', type=Path, help='JSON board array or object containing board/best_board')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output', type=Path, help='Optionally save the validation report as JSON')
    args = parser.parse_args()
    try:
        board = board_from_json(json.loads(args.board_json.read_text(encoding='utf-8')))
        report = validate_board(board, load_bundle(args.data_dir))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report = {'valid': False, 'complete': False, 'errors': [str(exc)]}
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    return 0 if report['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
