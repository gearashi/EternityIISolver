"""Experimental bounded GPU worker for a BOINC wrapper or local diagnostics.

This is not a native BOINC client and does not accept existing CPU-solver
workunits. Scheduled mode requires an explicit CUDA device ordinal supplied by
the wrapper. It performs no networking, dashboard work or novelty checking.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import uuid
import zipfile
import numpy as np
from validator import load_bundle, validate_board

WORKUNIT_SCHEMA = 'eternity-gpu-workunit/v1'
RESULT_SCHEMA = 'eternity-gpu-result/v1'
CHECKPOINT_SCHEMA = 'eternity-gpu-checkpoint/v1'
METADATA_KEY = '_boinc_metadata'
ENCODING = '4*(piece_id-1)+clockwise_rotation'
MAX_STEPS = 1_000_000_000
MAX_JSON_BYTES = 131072


class WorkunitError(ValueError):
    pass


def _reject_constant(value):
    raise WorkunitError(f'Non-finite JSON value: {value}')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WorkunitError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def strict_json(text):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def canonical_input_hash(document):
    encoded = json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise WorkunitError(f'{name} must be an integer in {lower}..{upper}')
    return value


def load_workunit(path, bundle):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise WorkunitError(f'Workunit JSON exceeds {MAX_JSON_BYTES} bytes')
    document = strict_json(path.read_text(encoding='utf-8'))
    required = {'schema', 'workunit_id', 'pieces_sha256', 'replicas', 'seed', 'steps_per_replica'}
    optional = {'board', 'backend', 'encoding', 'board_order'}
    if not isinstance(document, dict):
        raise WorkunitError('Workunit must be a JSON object')
    if required - document.keys():
        raise WorkunitError('Missing workunit fields: ' + ', '.join(sorted(required - document.keys())))
    if document.keys() - required - optional:
        raise WorkunitError('Unknown workunit fields: ' + ', '.join(sorted(document.keys() - required - optional)))
    if document['schema'] != WORKUNIT_SCHEMA:
        raise WorkunitError('Unsupported workunit schema')
    identity = document['workunit_id']
    if not isinstance(identity, str) or not 1 <= len(identity) <= 200 or not identity.isprintable():
        raise WorkunitError('workunit_id must be a printable string of 1..200 characters')
    expected = hashlib.sha256((bundle.data_dir / 'pieces.txt').read_bytes()).hexdigest()
    if document['pieces_sha256'] != expected:
        raise WorkunitError('Workunit pieces_sha256 does not match the bundled puzzle')
    _integer(document['replicas'], 'replicas', 32, 32768)
    _integer(document['seed'], 'seed', 0, 2**32 - 1)
    _integer(document['steps_per_replica'], 'steps_per_replica', 0, MAX_STEPS)
    if document.get('backend', 'auto') not in ('auto', 'cuda', 'opencl'):
        raise WorkunitError('Invalid workunit backend')
    if document.get('encoding', ENCODING) != ENCODING or document.get('board_order', 'row-major') != 'row-major':
        raise WorkunitError('Unsupported piece encoding or board order')
    board = document.get('board', list(bundle.record_board))
    report = validate_board(board, bundle)
    if not report['valid']:
        raise WorkunitError('Workunit starting board is illegal: ' + '; '.join(report['errors']))
    return document, canonical_input_hash(document), list(board), report['score']


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bind_cuda_device(ordinal):
    # Preserve the scheduler's inherited visibility/order environment. The
    # wrapper's GPU_DEVICE_NUM is a CUDA ordinal in that same environment.
    import cupy as cp
    count = cp.cuda.runtime.getDeviceCount()
    if not 0 <= ordinal < count:
        raise WorkunitError(f'Allocated CUDA device {ordinal} is unavailable; visible count is {count}')
    cp.cuda.Device(ordinal).use()


def _read_checkpoint(path, document, input_hash, bundle):
    try:
        with Path(path).open('rb') as handle:
            archive = np.load(handle, allow_pickle=False)
            if not isinstance(archive, np.lib.npyio.NpzFile):
                raise WorkunitError('Checkpoint must be an NPZ archive')
            with archive:
                if METADATA_KEY not in archive:
                    raise WorkunitError('Checkpoint is not bound to a GPU workunit')
                encoded = archive[METADATA_KEY]
                if encoded.shape != () or encoded.dtype.kind != 'U':
                    raise WorkunitError('Malformed checkpoint metadata')
                metadata = strict_json(str(encoded.item()))
                if not isinstance(metadata, dict) or metadata.get('schema') != CHECKPOINT_SCHEMA:
                    raise WorkunitError('Unsupported checkpoint metadata')
                if metadata.get('input_sha256') != input_hash or metadata.get('workunit_id') != document['workunit_id'] or metadata.get('pieces_sha256') != document['pieces_sha256']:
                    raise WorkunitError('Checkpoint belongs to a different workunit')
                _integer(metadata.get('completed_steps_per_replica'), 'checkpoint completed steps', 0, document['steps_per_replica'])
                active = metadata.get('gpu_active_seconds')
                if type(active) not in (int, float) or not math.isfinite(active) or active < 0:
                    raise WorkunitError('Invalid checkpoint GPU time')
                if 'boards' not in archive or archive['boards'].shape != (256, document['replicas']):
                    raise WorkunitError('Checkpoint replica count differs from its bound workunit')
                report = validate_board(metadata.get('best_board'), bundle)
                if not report['valid'] or type(metadata.get('best_score')) is not int or report['score'] != metadata['best_score']:
                    raise WorkunitError('Checkpoint best board failed independent validation')
        return metadata
    except WorkunitError:
        raise
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise WorkunitError(f'Malformed checkpoint archive: {exc}') from exc


def _save_checkpoint(engine, path, metadata):
    """Embed binding/progress in the same atomic NPZ as engine state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.name + '.' + uuid.uuid4().hex
    raw = path.with_name(stem + '.engine.npz')
    bound = path.with_name(stem + '.bound.npz')
    try:
        engine.checkpoint(raw)
        with np.load(raw, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files if key != METADATA_KEY}
        arrays[METADATA_KEY] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(',', ':')))
        np.savez_compressed(bound, **arrays)
        os.replace(bound, path)
    finally:
        for temporary in (raw, raw.with_suffix('.tmp.npz'), bound):
            temporary.unlink(missing_ok=True)


def run_workunit(input_path, output_path, checkpoint_path, progress_path, *, backend=None,
                 cuda_device=None, standalone_diagnostic=False, engine_factory=None,
                 device_binder=None, stop_requested=None, checkpoint_interval=30.0):
    paths = [Path(value).expanduser().resolve() for value in (input_path, output_path, checkpoint_path, progress_path)]
    if len(set(paths)) != len(paths):
        raise WorkunitError('Input, output, checkpoint and progress paths must be different')
    source, output, checkpoint, progress = paths
    bundle = load_bundle()
    document, input_hash, best_board, best_score = load_workunit(source, bundle)
    requested_backend = backend or document.get('backend', 'auto')
    if requested_backend not in ('auto', 'cuda', 'opencl'):
        raise WorkunitError('Invalid runtime backend')
    if document.get('backend', 'auto') not in ('auto', requested_backend):
        raise WorkunitError('Runtime backend conflicts with the workunit backend')
    if not standalone_diagnostic and (backend != 'cuda' or cuda_device is None):
        raise WorkunitError('Scheduled mode requires --backend cuda and --cuda-device from the BOINC wrapper; OpenCL allocation needs a native bridge')
    if cuda_device is not None:
        _integer(cuda_device, 'CUDA device ordinal', 0, 1023)
        if requested_backend != 'cuda':
            raise WorkunitError('A CUDA allocation cannot be used with another backend')
    if not math.isfinite(checkpoint_interval) or not 0 < checkpoint_interval <= 60:
        raise WorkunitError('checkpoint_interval must be positive and at most60 seconds')
    completed = 0
    active_seconds = 0.0
    resumed = checkpoint.exists()
    if resumed:
        metadata = _read_checkpoint(checkpoint, document, input_hash, bundle)
        completed = metadata['completed_steps_per_replica']
        active_seconds = float(metadata['gpu_active_seconds'])
        if metadata['best_score'] > best_score:
            best_board, best_score = metadata['best_board'], metadata['best_score']
    target = document['steps_per_replica']
    event = threading.Event()
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, lambda *_: event.set())
    engine = None
    started = time.monotonic()
    last_save = started
    last_progress = float('-inf')
    proposed = accepted = 0
    allocation = {'mode': 'standalone-diagnostic' if standalone_diagnostic else 'boinc-wrapper',
                  'api': requested_backend, 'cuda_ordinal': cuda_device}

    def stopping():
        return event.is_set() or (stop_requested is not None and stop_requested())

    def report_progress(force=False):
        nonlocal last_progress
        current = time.monotonic()
        if not force and current - last_progress < 1.0:
            return
        last_progress = current
        fraction = 1.0 if best_score == 480 or target == 0 else min(1.0, completed / target)
        _atomic_text(progress, f'{fraction:.12f}\n')

    def collect_best():
        nonlocal best_board, best_score, proposed, accepted
        if engine is None:
            return
        scores = np.asarray(engine.bestscores.get())
        if scores.shape != (document['replicas'],) or scores.dtype.kind not in 'iu' or np.any((scores < 0) | (scores > 480)):
            raise RuntimeError('GPU best-score array is invalid')
        candidate = int(np.argmax(scores))
        if int(scores[candidate]) > best_score:
            board = engine.bestboards[:, candidate].get().tolist()
            report = validate_board(board, bundle)
            if not report['valid'] or report['score'] != int(scores[candidate]):
                raise RuntimeError('GPU result failed independent CPU validation')
            best_board, best_score = board, report['score']
        counters = np.asarray(engine.counters.get())
        if counters.shape != (2 * document['replicas'],) or counters.dtype.kind not in 'iu' or np.any(counters < 0):
            raise RuntimeError('GPU move counters are invalid')
        proposed = sum(map(int, counters[:document['replicas']]))
        accepted = sum(map(int, counters[document['replicas']:]))
        if accepted > proposed:
            raise RuntimeError('GPU accepted count exceeds proposed count')

    def save():
        nonlocal last_save
        collect_best()
        metadata = {'schema': CHECKPOINT_SCHEMA, 'workunit_id': document['workunit_id'],
                    'input_sha256': input_hash, 'pieces_sha256': document['pieces_sha256'],
                    'completed_steps_per_replica': completed, 'gpu_active_seconds': active_seconds,
                    'best_board': best_board, 'best_score': best_score}
        if engine is not None:
            _save_checkpoint(engine, checkpoint, metadata)
        report_progress(force=True)
        last_save = time.monotonic()

    try:
        # A zero-step fresh diagnostic validates the contract without needing a
        # GPU. Existing checkpoints still load/validate their engine state.
        if target > 0 or resumed:
            if cuda_device is not None:
                (device_binder or _bind_cuda_device)(cuda_device)
            if engine_factory is None:
                from gpu_engine import create_engine
                engine_factory = create_engine
            engine = engine_factory(bundle, document['replicas'], document['seed'],
                                    backend=requested_backend, cache_dir=checkpoint.parent / 'gpu-cache')
            if not standalone_diagnostic and getattr(engine, 'backend', None) != 'cuda':
                raise RuntimeError('Scheduled worker did not bind the assigned CUDA backend')
            engine.initialize([best_board])
            if resumed and not engine.resume(checkpoint):
                raise WorkunitError('Bound checkpoint could not be resumed')
            engine.verify_device_scores(range(min(32, document['replicas'])))
        report_progress(force=True)
        steps = 1
        while completed < target and best_score != 480 and not stopping():
            count = min(steps, target - completed)
            engine.cool(45.0 * completed / max(1, target))
            duration = float(engine.step(int(count)))
            if not math.isfinite(duration) or duration <= 0:
                raise RuntimeError('GPU returned an invalid kernel duration')
            completed += count
            active_seconds += duration / 1000.0
            if not math.isfinite(active_seconds):
                raise RuntimeError('Accumulated GPU time overflowed')
            collect_best()
            report_progress()
            if duration > 180:
                steps = max(1, steps // 2)
            elif duration < 30:
                steps = min(256, steps * 2)
            if time.monotonic() - last_save >= checkpoint_interval:
                save()
        save()
        verified = validate_board(best_board, bundle)
        if not verified['valid'] or verified['score'] != best_score:
            raise RuntimeError('Final result failed independent CPU validation')
        finished = completed >= target or best_score == 480
        result = {'schema': RESULT_SCHEMA, 'workunit_id': document['workunit_id'],
                  'input_sha256': input_hash, 'pieces_sha256': document['pieces_sha256'],
                  'mode': 'five-clue', 'status': 'completed' if finished else 'interrupted',
                  'backend': getattr(engine, 'backend', requested_backend),
                  'device': getattr(engine, 'device', 'not used: zero-step workunit'),
                  'allocation': allocation, 'replicas': document['replicas'],
                  'score': best_score, 'board': best_board, 'complete': best_score == 480,
                  'exhausted': False, 'heuristic': True,
                  'steps_per_replica': target, 'completed_steps_per_replica': completed,
                  'budget_complete': completed >= target, 'elapsed_seconds': active_seconds,
                  'elapsed_kind': 'accumulated GPU event time; not the work budget',
                  'wall_seconds_this_invocation': time.monotonic() - started,
                  'proposed_moves': proposed, 'accepted_moves': accepted, 'resumed': resumed}
        _atomic_text(output, json.dumps(result, indent=2) + '\n')
        return (0 if finished else 75), result
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)


def main(argv=None, engine_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--progress', required=True, type=Path)
    parser.add_argument('--backend', choices=('auto', 'cuda', 'opencl'))
    parser.add_argument('--cuda-device', type=int, help='Allocated BOINC CUDA device ordinal, supplied by the wrapper')
    parser.add_argument('--standalone-diagnostic', action='store_true', help='Local testing only; permits automatic or OpenCL GPU selection')
    args = parser.parse_args(argv)
    try:
        status, result = run_workunit(args.input, args.output, args.checkpoint, args.progress,
                                     backend=args.backend, cuda_device=args.cuda_device,
                                     standalone_diagnostic=args.standalone_diagnostic, engine_factory=engine_factory)
        print(json.dumps({'status': result['status'], 'score': result['score'],
                          'completed_steps_per_replica': result['completed_steps_per_replica']}))
        return status
    except (WorkunitError, ValueError, OSError, KeyError) as exc:
        print(f'Invalid workunit or checkpoint: {exc}', file=sys.stderr)
        return 2
    except Exception as exc:
        print(f'GPU worker failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
