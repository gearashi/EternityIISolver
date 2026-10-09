"""Export a validated board locally for manual project review; no networking.

The library-style representation is not a completed CPU workunit. Acceptance
by the project has not been confirmed, and no discovery claim is made.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import uuid

from app_paths import state_root
from validator import load_bundle, validate_board

MAX_JSON_BYTES = 131072
ENCODING = '4*(piece_id-1)+clockwise_rotation'


class ExportError(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExportError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _nonfinite(value):
    raise ExportError(f'Non-finite JSON number: {value}')


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        _nonfinite(text)
    return value


def _read_document(path):
    with Path(path).open('rb') as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ExportError(f'Board JSON exceeds {MAX_JSON_BYTES} bytes')
    try:
        document = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object,
                              parse_constant=_nonfinite, parse_float=_finite_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ExportError(f'Invalid board JSON: {exc}') from exc
    if not isinstance(document, dict):
        raise ExportError('Board JSON must be an object')
    return document


def _validated_board(document, bundle):
    board = document.get('board')
    if not isinstance(board, list):
        raise ExportError('Board must be an array of 256 placement codes')
    report = validate_board(board, bundle)
    if not report['valid']:
        raise ExportError('Invalid board: ' + '; '.join(report['errors']))
    if type(document.get('score')) is not int or document['score'] != report['score']:
        raise ExportError('Claimed score does not match the independent board score')
    return list(board), report


def _local_hash(board):
    return hashlib.sha256(struct.pack('<256H', *board)).hexdigest()


def _reference_palette(bundle):
    """Recover the exact bijection from the trusted reference's NESW letters."""
    reference = _read_document(bundle.data_dir / 'record466.json')
    board, _ = _validated_board(reference, bundle)
    if tuple(board) != bundle.record_board:
        raise ExportError('Reference board differs from the trusted puzzle bundle')
    letters = reference.get('board_edges')
    if not isinstance(letters, str) or len(letters) != 1024 or any(c < 'a' or c > 'w' for c in letters):
        raise ExportError('Invalid reference letter encoding')
    palette = {}
    for cell, code in enumerate(board):
        for color, letter in zip(bundle.oriented_edges[code], letters[4 * cell:4 * cell + 4]):
            if color in palette and palette[color] != letter:
                raise ExportError('Reference palette is inconsistent')
            palette[color] = letter
    if set(palette) != set(range(23)) or set(palette.values()) != set('abcdefghijklmnopqrstuvw') or palette[0] != 'a':
        raise ExportError('Reference palette must map all 23 colors uniquely and preserve gray')
    return reference, palette


def validate_export_document(document, bundle=None):
    """CPU-validate a library-style board, including both textual encodings."""
    if not isinstance(document, dict):
        raise ExportError('Export document must be an object')
    bundle = load_bundle() if bundle is None else bundle
    board, report = _validated_board(document, bundle)
    if type(document.get('size')) is not int or document['size'] != 16:
        raise ExportError('Board size must be 16')
    name = document.get('name')
    if not isinstance(name, str) or not 1 <= len(name) <= 200 or not name.isprintable():
        raise ExportError('Board name must be a printable string of 1..200 characters')
    if type(document.get('breaks')) is not int or document['breaks'] != report['break_count']:
        raise ExportError('Break count does not match the board')
    _, palette = _reference_palette(bundle)
    expected_edges = ''.join(palette[color] for code in board for color in bundle.oriented_edges[code])
    expected_pieces = ''.join(f'{code // 4 + 1:03d}' for code in board)
    if document.get('board_edges') != expected_edges:
        raise ExportError('board_edges differs from the board or trusted reference palette')
    if document.get('board_pieces') != expected_pieces:
        raise ExportError('board_pieces differs from the board')
    digest = _local_hash(board)
    pieces_hash = bundle.metadata['input_sha256']['pieces.txt']
    for key, expected in (('local_board_sha256_uint16le', digest), ('pieces_sha256', pieces_hash),
                          ('encoding', ENCODING), ('board_order', 'row-major'), ('edge_order', 'NESW'),
                          ('orientation', 'canonical-five-clue')):
        if key in document and document[key] != expected:
            raise ExportError(f'Export {key} is inconsistent')
    return {**report, 'local_board_sha256_uint16le': digest, 'pieces_sha256': pieces_hash,
            'text_encodings_valid': True, 'canonical_five_clue_orientation': True}


def _layout(board, score, attribution):
    lines = ['Eternity II board for manual review', f'Score: {score}/480; unmatched edges: {480-score}',
             'Project acceptance is unconfirmed. This is not a completed CPU workunit.',
             'No upload was performed. Discovery status is unverified.',
             'Rows and columns are 1-based, from the top left, in canonical five-clue orientation.',
             'Piece IDs are 1-based. Rotation is clockwise quarter-turns (0,1,2,3)',
             'relative to the bundled pieces.txt U,D,L,R orientation.',
             f'Placement code: {ENCODING}', attribution, '',
             'row column piece_id clockwise_quarter_turns placement_code']
    lines.extend(f'{cell//16+1:02d} {cell%16+1:02d} {code//4+1:03d} {code%4} {code:04d}'
                 for cell, code in enumerate(board))
    return '\n'.join(lines) + '\n'


def export_best(state_dir, output_dir=None):
    """Validate first, then save three files in a new local export directory."""
    home = state_root(state_dir)
    bundle = load_bundle()
    reference, palette = _reference_palette(bundle)
    try:
        candidate = _read_document(home / 'runtime' / 'best.json')
        fallback = False
    except FileNotFoundError:
        candidate = reference
        fallback = True
    board, report = _validated_board(candidate, bundle)
    digest = _local_hash(board)
    exact_reference = tuple(board) == bundle.record_board
    attribution = (f"Exact copy of the bundled 466/480 reference credited to {reference.get('discoverer_name', 'Jef')}; not a new result."
                   if exact_reference else 'Local saved board; discoverer and novelty have not been verified.')
    document = {'name': reference['name'] if exact_reference else f'FiveClue_{report["score"]}_local_{digest[:12]}',
                'size': 16, 'score': report['score'], 'breaks': report['break_count'], 'board': board,
                'board_edges': ''.join(palette[color] for code in board for color in bundle.oriented_edges[code]),
                'board_pieces': ''.join(f'{code//4+1:03d}' for code in board),
                'encoding': ENCODING, 'board_order': 'row-major', 'edge_order': 'NESW',
                'orientation': 'canonical-five-clue', 'pieces_sha256': bundle.metadata['input_sha256']['pieces.txt'],
                'local_board_sha256_uint16le': digest,
                'export_purpose': 'manual project review; acceptance unconfirmed',
                'project_acceptance_confirmed': False, 'completed_cpu_workunit': False,
                'discovery_status': 'unverified', 'automatic_upload': False,
                'source_kind': 'bundled_reference_fallback' if fallback else 'runtime_best',
                'matches_bundled_reference': exact_reference, 'attribution': attribution}
    if exact_reference:
        document['discoverer_name'] = reference.get('discoverer_name', 'Jef')
    checked = validate_export_document(document, bundle)
    validation = {**checked, 'validation_method': 'independent CPU piece/frame/five-clue/edge validation',
                  'project_acceptance_confirmed': False, 'completed_cpu_workunit': False,
                  'discovery_status': 'unverified', 'used_reference_fallback': fallback,
                  'matches_bundled_reference': exact_reference, 'attribution': attribution}
    payloads = {'board.json': json.dumps(document, indent=2, allow_nan=False) + '\n',
                'layout.txt': _layout(board, report['score'], attribution),
                'validation.json': json.dumps(validation, indent=2, allow_nan=False) + '\n'}
    # No output directory or file is created until all data has been validated.
    base = Path(output_dir).expanduser().resolve() if output_dir is not None else home / 'exports'
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = base / f'{stamp}_{report["score"]}_{digest[:12]}_{uuid.uuid4().hex[:12]}'
    directory.mkdir(parents=True, exist_ok=False)
    for name, text in payloads.items():
        (directory / name).write_text(text, encoding='utf-8')
    return {'directory': str(directory), 'json_path': str(directory / 'board.json'),
            'layout_path': str(directory / 'layout.txt'), 'validation_path': str(directory / 'validation.json'),
            'score': report['score'], 'local_board_sha256_uint16le': digest,
            'used_reference_fallback': fallback, 'matches_bundled_reference': exact_reference,
            'attribution': attribution, 'project_acceptance_confirmed': False, 'automatic_upload': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        result = export_best(args.state_dir, args.output_dir)
    except (ExportError, OSError, ValueError) as exc:
        print(json.dumps({'exported': False, 'error': str(exc)}))
        return 2
    print(json.dumps({'exported': True, **result}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
