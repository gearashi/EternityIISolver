"""Generate fresh complete legal boards on a GPU, then validate <=64 hints.

Counts are whole generated/scored boards, not local moves and not a claim of
uniqueness. This experimental sampler does no exact pruning or networking.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np

from validator import validate_board


MAX_HINTS = 64
KERNEL_SOURCE = r'''
__device__ __forceinline__ unsigned int hs_rot(unsigned int x, int k) {
    return (x << k) | (x >> (32 - k));
}
__device__ __forceinline__ unsigned int hs_next(unsigned int *state) {
    unsigned int result = hs_rot(state[1] * 5u, 7) * 9u;
    unsigned int temporary = state[1] << 9;
    state[2] ^= state[0]; state[3] ^= state[1];
    state[1] ^= state[2]; state[0] ^= state[3];
    state[2] ^= temporary; state[3] = hs_rot(state[3], 11);
    return result;
}
__device__ __forceinline__ unsigned int hs_bounded(unsigned int *state, unsigned int bound) {
    unsigned int threshold = (0u - bound) % bound;
    unsigned int value;
    do { value = hs_next(state); } while (value < threshold);
    return value % bound;
}
extern "C" __global__ void sample_full_boards(
    short *boards, short *bestboards, int *scores, int *bestscores,
    unsigned int *rng, const short *base, const short *cells,
    const int *offsets, const unsigned char *fixed,
    const unsigned char *rotation_masks, const unsigned char *faces, int N) {
    int replica = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (replica >= N) return;
    unsigned int state[4];
    for (int k = 0; k < 4; ++k) state[k] = rng[k * N + replica];
    for (int c = 0; c < 256; ++c) boards[c * N + replica] = base[c];
    for (int group = 0; group < 3; ++group) {
        int begin = offsets[group], count = offsets[group + 1] - begin;
        for (int i = count - 1; i > 0; --i) {
            int j = (int)hs_bounded(state, (unsigned int)(i + 1));
            int a = cells[begin + i], b = cells[begin + j];
            short value = boards[a * N + replica];
            boards[a * N + replica] = boards[b * N + replica];
            boards[b * N + replica] = value;
        }
    }
    for (int c = 0; c < 256; ++c) {
        if (fixed[c]) continue;
        int piece = boards[c * N + replica] / 4;
        int rotation = (int)(hs_next(state) & 3u);
        unsigned int mask = rotation_masks[c * 256 + piece];
        while ((mask & (1u << rotation)) == 0u) rotation = (rotation + 1) & 3;
        boards[c * N + replica] = (short)(4 * piece + rotation);
    }
    int score = 0;
    for (int c = 0; c < 256; ++c) {
        int code = boards[c * N + replica];
        if (c % 16 != 15) score += faces[code * 4 + 1] == faces[boards[(c + 1) * N + replica] * 4 + 3];
        if (c < 240) score += faces[code * 4 + 2] == faces[boards[(c + 16) * N + replica] * 4];
    }
    scores[replica] = score;
    if (score > bestscores[replica]) {
        bestscores[replica] = score;
        for (int c = 0; c < 256; ++c) bestboards[c * N + replica] = boards[c * N + replica];
    }
    for (int k = 0; k < 4; ++k) rng[k * N + replica] = state[k];
}
extern "C" __global__ void gather_samples(
    const short *boards, const short *bestboards, const int *indices,
    const int *banks, short *output, int N, int K) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= K * 256) return;
    int candidate = index / 256, cell = index % 256, replica = indices[candidate];
    output[index] = banks[candidate] ? bestboards[cell * N + replica] : boards[cell * N + replica];
}
'''


def _layout(bundle):
    if not validate_board(bundle.record_board, bundle)['valid']:
        raise ValueError('Sampler requires a validated five-clue reference board')
    faces = np.asarray(bundle.oriented_edges, dtype=np.uint8)
    fixed = np.zeros(256, dtype=np.uint8)
    for cell in bundle.fixed_clues:
        fixed[cell] = 1
    exterior = np.asarray([(c < 16, c % 16 == 15, c >= 240, c % 16 == 0) for c in range(256)])
    legal = np.all((faces.reshape(1, 256, 4, 4) == 0) == exterior[:, None, None, :], axis=3)
    masks = (legal * (1 << np.arange(4))[None, None, :]).sum(axis=2).astype(np.uint8)
    groups = [np.flatnonzero((exterior.sum(axis=1) == border) & (fixed == 0)) for border in (2, 1, 0)]
    cells = np.concatenate(groups).astype(np.int16)
    offsets = np.asarray([0, *np.cumsum([len(group) for group in groups])], dtype=np.int32)
    return np.asarray(bundle.record_board, dtype=np.int16), cells, offsets, fixed, masks, faces


def _selection(best_scores, current_scores, rng):
    """At most32 high-scoring stream maxima plus32 random current boards."""
    best_scores, current_scores = np.asarray(best_scores), np.asarray(current_scores)
    if best_scores.ndim != 1 or current_scores.shape != best_scores.shape or len(best_scores) < 1:
        raise ValueError('Malformed GPU score arrays')
    count = min(MAX_HINTS // 2, len(best_scores))
    # Stable tie-breaking makes selection reproducible on identical scores.
    top = np.argsort(-best_scores, kind='stable')[:count]
    random = rng.choice(len(best_scores), count, replace=False)
    indices = np.concatenate((top, random)).astype(np.int32)
    banks = np.asarray([1] * count + [0] * count, dtype=np.int32)
    scores = np.concatenate((best_scores[top], current_scores[random]))
    return indices, banks, scores


def _screen(boards, scores, bundle, should_stop=None):
    if len(boards) != len(scores) or len(boards) > MAX_HINTS:
        raise ValueError('Candidate screen requires at most64 paired boards/scores')
    unique = {}
    duplicates = selected = validated = 0
    for board, claimed in zip(boards, scores):
        if should_stop is not None and should_stop():
            break
        if np.asarray(board).dtype.kind not in 'iu' or isinstance(claimed, (bool, np.bool_)) or not isinstance(claimed, (int, np.integer)):
            raise RuntimeError('GPU sampler returned non-integer states or scores')
        candidate = tuple(int(state) for state in board)
        if len(candidate) != 256:
            raise RuntimeError('Malformed sampled board')
        selected += 1
        if candidate in unique:
            duplicates += 1
            score = unique[candidate]
        else:
            report = validate_board(candidate, bundle)
            validated += 1
            if not report['valid']:
                raise RuntimeError('GPU sampler returned an illegal board')
            score = report['score']
            unique[candidate] = score
        if score != int(claimed):
            raise RuntimeError('GPU sampler score failed independent CPU validation')
    ordered = sorted(unique, key=lambda board: (-unique[board], board))
    return {'boards': [list(board) for board in ordered], 'scores': [unique[board] for board in ordered],
            'selected_boards': selected, 'cpu_validated_boards': validated,
            'duplicate_candidates': duplicates, 'retained_unique_boards': len(ordered)}


class _Device:
    def __init__(self, bundle, replicas, seed, backend, cache_dir, allow_opencl_cpu):
        from gpu_backends import make_backend, default_cache_dir
        from kernel_port import to_opencl
        self.n = replicas
        self.selection_rng = np.random.default_rng(seed ^ 0xE27A912B)
        started = time.perf_counter()
        self.runtime = make_backend(backend, cache_dir, allow_opencl_cpu=allow_opencl_cpu)
        self.backend, self.device = self.runtime.name, self.runtime.device
        self.details = dict(self.runtime.details)
        xp = self.runtime.xp
        if self.backend == 'cuda':
            self.module = xp.RawModule(code=KERNEL_SOURCE, options=('--std=c++17',),
                                       name_expressions=('sample_full_boards', 'gather_samples'))
            self.kernel = self.module.get_function('sample_full_boards')
            self.gather = self.module.get_function('gather_samples')
        else:
            runtime = self.runtime
            directory = Path(cache_dir) if cache_dir is not None else default_cache_dir()
            self.module = runtime.cl.Program(runtime.context, to_opencl(KERNEL_SOURCE)).build(
                options=['-cl-std=CL1.2'], cache_dir=str(directory / 'opencl'))
            def wrap(name):
                kernel = runtime.cl.Kernel(self.module, name)
                limit = kernel.get_work_group_info(runtime.cl.kernel_work_group_info.WORK_GROUP_SIZE, runtime.device_object)
                def launch(grid, block, args):
                    total = int(grid[0]) * int(block[0])
                    local = min(128, int(block[0]), int(limit))
                    values = tuple(getattr(value, 'buffer', value) for value in args)
                    return kernel(runtime.queue, (total,), (local,) if total % local == 0 else None, *values)
                return launch
            self.kernel, self.gather = wrap('sample_full_boards'), wrap('gather_samples')
        self.compile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self.tables = tuple(xp.asarray(value) for value in _layout(bundle))
        self.boards = xp.zeros((256, replicas), dtype=np.int16)
        self.bestboards = xp.zeros((256, replicas), dtype=np.int16)
        self.scores = xp.zeros(replicas, dtype=np.int32)
        self.bestscores = xp.asarray(np.full(replicas, -1, dtype=np.int32))
        states = np.random.default_rng(seed).integers(0, 2**32, (4, replicas), dtype=np.uint32)
        states[0, np.all(states == 0, axis=0)] = 1
        self.rng = xp.asarray(states)
        self.initialization_seconds = time.perf_counter() - started

    def generate(self):
        return self.runtime.timed(self.kernel, ((self.n + 127) // 128,), (128,),
                                  (self.boards, self.bestboards, self.scores, self.bestscores,
                                   self.rng, *self.tables, np.int32(self.n)))

    def selected(self):
        best, current = self.bestscores.get(), self.scores.get()
        indices, banks, scores = _selection(best, current, self.selection_rng)
        xp = self.runtime.xp
        output = xp.zeros((len(indices), 256), dtype=np.int16)
        self.runtime.timed(self.gather, ((len(indices) * 256 + 127) // 128,), (128,),
                           (self.boards, self.bestboards, xp.asarray(indices), xp.asarray(banks), output,
                            np.int32(self.n), np.int32(len(indices))))
        return output.get(), scores

    def close(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        runtime = getattr(self, 'runtime', None)
        for name in ('tables', 'boards', 'bestboards', 'scores', 'bestscores', 'rng', 'kernel', 'gather', 'module'):
            setattr(self, name, None)
        self.runtime = None
        if runtime is not None and runtime.name == 'cuda':
            runtime.xp.get_default_memory_pool().free_all_blocks()
            runtime.xp.get_default_pinned_memory_pool().free_all_blocks()

    def __del__(self):
        # Also release partially initialized arrays after a constructor error.
        try:
            self.close()
        except Exception:
            pass


def sample(bundle, seconds=3, replicas=4096, seed=20261007, backend='auto', cache_dir=None,
           should_stop=None, on_progress=None, allow_opencl_cpu=False):
    """Generate whole boards for a bounded stage, returning <=64 verified hints.

    elapsed_seconds includes GPU generation, final transfer and CPU screening;
    compilation and initialization are separate. A final kernel/transfer may
    slightly exceed the requested generation duration. Device initialization
    is not preemptible, so cancellation is checked immediately before/after it.
    """
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if type(replicas) is not int or not 32 <= replicas <= 32768:
        raise ValueError('replicas must be32..32768')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32')
    if backend not in ('auto', 'cuda', 'opencl'):
        raise ValueError('Invalid sampler backend')
    if allow_opencl_cpu and backend != 'opencl':
        raise ValueError('CPU diagnostics require explicit OpenCL')
    empty = {'boards': [], 'scores': [], 'selected_boards': 0, 'cpu_validated_boards': 0,
             'duplicate_candidates': 0, 'retained_unique_boards': 0}
    result = {**empty, 'generated_boards': 0, 'batches': 0, 'elapsed_seconds': 0.0,
              'boards_per_second': 0.0, 'compile_seconds': 0.0, 'initialization_seconds': 0.0,
              'backend': backend, 'device': None, 'replicas': replicas, 'seed': seed,
              'counter_semantics': 'complete generated and GPU-scored boards; uniqueness not claimed',
              'validation_scope': 'only selected candidates are individually CPU validated',
              'external_network_enabled': False, 'outcome': 'finished_bounded_run'}
    if should_stop is not None and should_stop():
        result['outcome'] = 'stopped'
        return result
    if seconds == 0:
        return result
    if on_progress is not None:
        on_progress({'phase': 'gpu_compilation', 'generated_boards': 0})
    device = _Device(bundle, replicas, seed, backend, cache_dir, allow_opencl_cpu)
    result.update(backend=device.backend, device=device.device, backend_details=device.details,
                  compile_seconds=device.compile_seconds, initialization_seconds=device.initialization_seconds)
    started = time.perf_counter()
    report_at = started
    kernel_ms = 0.0
    try:
        while time.perf_counter() - started < seconds:
            if should_stop is not None and should_stop():
                result['outcome'] = 'stopped'
                break
            kernel_ms += device.generate()
            result['generated_boards'] += replicas
            result['batches'] += 1
            elapsed = time.perf_counter() - started
            if on_progress is not None and time.perf_counter() >= report_at:
                on_progress({'phase': 'gpu_sampling', 'generated_boards': result['generated_boards'],
                             'elapsed_seconds': elapsed, 'boards_per_second': result['generated_boards'] / max(elapsed, 1e-9)})
                report_at = time.perf_counter() + 0.5
        if result['generated_boards'] and not (should_stop is not None and should_stop()):
            if on_progress is not None:
                on_progress({'phase': 'cpu_screening', 'generated_boards': result['generated_boards']})
            boards, scores = device.selected()
            result.update(_screen(boards, scores, bundle, should_stop))
        if should_stop is not None and should_stop():
            result['outcome'] = 'stopped'
    finally:
        device.close()
    result['elapsed_seconds'] = time.perf_counter() - started
    result['boards_per_second'] = result['generated_boards'] / max(result['elapsed_seconds'], 1e-9)
    result['kernel_seconds'] = kernel_ms / 1000
    return result


def diagnose(bundle, *, backend='auto', cache_dir=None, allow_opencl_cpu=False, seed=20261007):
    """Two tiny32-board kernel batches, all64 outputs CPU-validated explicitly."""
    started = time.perf_counter()
    device = _Device(bundle, 32, seed, backend, cache_dir, allow_opencl_cpu)
    previous = None
    changed = 0
    all_scores = []
    try:
        for _ in range(2):
            device.generate()
            boards = device.boards.get().T
            scores = device.scores.get()
            report = _screen(boards, scores, bundle)
            if report['selected_boards'] != 32:
                raise RuntimeError('Incomplete diagnostic board validation')
            all_scores.extend(int(value) for value in scores)
            if previous is not None:
                changed = int(np.any(boards != previous, axis=1).sum())
            previous = boards.copy()
    finally:
        device.close()
    if changed != 32:
        raise RuntimeError('Diagnostic RNG did not produce a fresh board in each stream')
    return {'passed': True, 'generated_boards': 64, 'individually_cpu_checked_boards': 64,
            'replicas': 32, 'batches': 2, 'changed_streams': changed,
            'min_score': min(all_scores), 'max_score': max(all_scores),
            'elapsed_seconds': time.perf_counter() - started,
            'compile_seconds': device.compile_seconds, 'backend': device.backend, 'device': device.device}


def main(argv=None):
    """Explicit bounded diagnostic/benchmark CLI, including optional PoCL checks."""
    import argparse
    import hashlib
    import json
    from validator import load_bundle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('auto', 'cuda', 'opencl'), default='auto')
    parser.add_argument('--replicas', type=int, default=4096)
    parser.add_argument('--seconds', type=float, default=3)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--allow-opencl-cpu', action='store_true', help='Explicit CPU OpenCL diagnostic; not production CPU fallback')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 30:
        parser.error('seconds must be greater than zero and no more than30')
    if not 32 <= args.replicas <= 32768:
        parser.error('replicas must be32..32768')
    try:
        bundle = load_bundle()
        diagnostic = diagnose(bundle, backend=args.backend, cache_dir=args.cache_dir,
                              allow_opencl_cpu=args.allow_opencl_cpu, seed=args.seed)
        result = sample(bundle, seconds=args.seconds, replicas=args.replicas, seed=args.seed,
                        backend=args.backend, cache_dir=args.cache_dir, allow_opencl_cpu=args.allow_opencl_cpu)
        result.update(diagnostics=diagnostic, kernel_sha256=hashlib.sha256(KERNEL_SOURCE.encode()).hexdigest(),
                      pieces_sha256=bundle.metadata['input_sha256']['pieces.txt'],
                      timing_note='Generation, final candidate transfer and CPU screening included; compilation and initialization reported separately',
                      experimental=True, claimed_exact_speedup=False)
        if Path(__file__).is_file():
            result['module_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({key: value for key, value in result.items() if key not in ('boards', 'scores')}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'passed': False, 'error': f'{type(exc).__name__}: {exc}'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
