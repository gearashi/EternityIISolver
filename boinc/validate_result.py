"""Check a BOINC candidate locally; this is not a registered BOINC validator.

Run from a source checkout with Python 3.11+. No GPU libraries are imported.
This checks candidate validity and workunit binding, not proof of computation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validator import PuzzleBundle, load_bundle, sha256, validate_board

MAX_JSON_BYTES = 131072
ENCODING = '4*(piece_id-1)+clockwise_rotation'


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'Duplicate JSON key: {key}')
        value[key] = item
    return value


def _nonfinite(value: str) -> None:
    raise ValueError(f'Non-finite JSON number: {value}')


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open('rb') as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f'JSON exceeds {MAX_JSON_BYTES} bytes')
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=_object, parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ValueError('JSON document must be an object')
    return value


def canonical_input_sha256(workunit: dict[str, Any]) -> str:
    raw = json.dumps(workunit, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _integer(value: Any, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f'{name} must be a boolean')
    return value


def validate_result(workunit: dict[str, Any], result: dict[str, Any],
                    bundle: PuzzleBundle | None = None) -> dict[str, Any]:
    """Raise ValueError for an invalid candidate or unfinished bounded task."""
    bundle = load_bundle() if bundle is None else bundle
    required = {'schema', 'workunit_id', 'pieces_sha256', 'replicas', 'seed', 'steps_per_replica'}
    optional = {'board', 'backend', 'encoding', 'board_order'}
    if required - workunit.keys() or workunit.keys() - required - optional:
        raise ValueError('Missing or unknown workunit fields')
    if workunit.get('schema') != 'eternity-gpu-workunit/v1':
        raise ValueError('Unsupported workunit schema')
    workunit_id = workunit.get('workunit_id')
    if not isinstance(workunit_id, str) or not 1 <= len(workunit_id) <= 200 or not workunit_id.isprintable():
        raise ValueError('workunit_id must be a printable string of 1..200 characters')
    if 'seconds' in workunit:
        raise ValueError('v1 uses steps_per_replica, not a wall-clock budget')
    target = _integer(workunit.get('steps_per_replica'), 0, 10**9, 'steps_per_replica')
    _integer(workunit.get('replicas'), 32, 32768, 'replicas')
    _integer(workunit.get('seed'), 0, 2**32 - 1, 'seed')
    if workunit.get('encoding', ENCODING) != ENCODING:
        raise ValueError('Unsupported board encoding')
    if workunit.get('board_order', 'row-major') != 'row-major':
        raise ValueError('Unsupported board order')
    if workunit.get('backend', 'cuda') not in ('auto', 'cuda', 'opencl'):
        raise ValueError('Unsupported backend')
    pieces_hash = sha256(bundle.data_dir / 'pieces.txt')
    if workunit.get('pieces_sha256') != pieces_hash:
        raise ValueError('Workunit piece-set hash does not match the trusted local data')
    if 'board' in workunit and not validate_board(workunit['board'], bundle)['valid']:
        raise ValueError('Workunit starting board is not a legal five-clue board')
    if result.get('schema') != 'eternity-gpu-result/v1':
        raise ValueError('Unsupported result schema')
    if result.get('workunit_id') != workunit_id:
        raise ValueError('Result workunit_id does not match')
    input_hash = canonical_input_sha256(workunit)
    if result.get('input_sha256') != input_hash:
        raise ValueError('Result input_sha256 does not match')
    if result.get('pieces_sha256') != pieces_hash:
        raise ValueError('Result pieces_sha256 does not match')
    if result.get('mode') != 'five-clue':
        raise ValueError('Result must use the five-clue constraint regime')
    if not isinstance(result.get('board'), list):
        raise ValueError('Result board must be a JSON array')
    report = validate_board(result['board'], bundle)
    if not report['valid']:
        raise ValueError('Illegal candidate board: ' + '; '.join(report['errors']))
    score = _integer(result.get('score'), 0, 480, 'score')
    if score != report['score']:
        raise ValueError('Claimed score does not match independent edge counting')
    complete = _boolean(result.get('complete'), 'complete')
    if complete != (score == 480):
        raise ValueError('complete must mean a valid 480/480 puzzle solution')
    if _boolean(result.get('exhausted'), 'exhausted'):
        raise ValueError('A heuristic search cannot claim exhaustive coverage')
    if _integer(result.get('steps_per_replica'), 0, 10**9, 'result steps_per_replica') != target:
        raise ValueError('Result step budget does not match')
    steps = _integer(result.get('completed_steps_per_replica'), 0, target,
                     'completed_steps_per_replica')
    budget_complete = _boolean(result.get('budget_complete'), 'budget_complete')
    if budget_complete != (steps >= target):
        raise ValueError('budget_complete does not agree with completed steps')
    if result.get('status') != 'completed' or not (budget_complete or complete):
        raise ValueError('Result did not finish its step budget or solve the puzzle')
    return {'valid': True, 'workunit_id': workunit_id, 'input_sha256': input_hash,
            'pieces_sha256': pieces_hash, 'score': score, 'puzzle_solved': complete,
            'budget_complete': budget_complete, 'completed_steps_per_replica': steps,
            'exhausted': False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate_result(read_json(args.input), read_json(args.result),
                                 load_bundle(args.data_dir))
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(json.dumps({'valid': False, 'error': str(exc)}, ensure_ascii=True))
        return 1
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
