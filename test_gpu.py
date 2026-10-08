"""Bounded device checks against independent full-board CPU validation.

Use --backend opencl --allow-opencl-cpu only for CPU OpenCL CI diagnostics.
Normal opencl and auto modes require an actual GPU; they never fall back to CPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from gpu_engine import create_engine
from validator import load_bundle, validate_board
from app_paths import resource_root


def run(backend='auto', replicas=128, allow_opencl_cpu=False, device_name=None, output=None):
    start = time.time()
    bundle = load_bundle()
    with tempfile.TemporaryDirectory(prefix='eternity-gpu-test-', ignore_cleanup_errors=True) as temp:
        g = create_engine(bundle, replicas, 718290, backend, Path(temp) / 'cache',
                          allow_opencl_cpu=allow_opencl_cpu, device_name=device_name)
        xp = g.xp
        g.initialize([bundle.record_board])
        rng = np.random.default_rng(19466)
        base = np.tile(np.asarray(bundle.record_board, np.int16), (g.n, 1))
        boards = g.perturb(base, np.full(g.n, 12))
        aa, bb, na, nb, expected = [], [], [], [], []
        adjacent = rotations = 0
        free = np.flatnonzero(g.fixed == 0)
        for i, board in enumerate(boards):
            a = int(rng.choice(free)); b = -1
            if i % 8:
                same = g.groups[g.types[a]]
                neighbors = [int(q) for q in g.neighbors[a] if q >= 0 and not g.fixed[q] and g.types[q] == g.types[a]]
                b = int(rng.choice(neighbors)) if neighbors and i % 2 else int(rng.choice(same[same != a]))
                adjacent += int(b in g.neighbors[a])
                pa, pb = int(board[b]) // 4, int(board[a]) // 4
                ca = pa * 4 + int(rng.choice(np.flatnonzero(g.allowed[a, pa * 4:pa * 4 + 4])))
                cb = pb * 4 + int(rng.choice(np.flatnonzero(g.allowed[b, pb * 4:pb * 4 + 4])))
            else:
                rotations += 1
                pa = int(board[a]) // 4
                ca = pa * 4 + int(rng.choice(np.flatnonzero(g.allowed[a, pa * 4:pa * 4 + 4])))
                cb = 0
            altered = board.copy(); altered[a] = ca
            if b >= 0:
                altered[b] = cb
            before, after = validate_board(board, bundle), validate_board(altered, bundle)
            assert before['valid'] and after['valid']
            aa.append(a); bb.append(b); na.append(ca); nb.append(cb)
            expected.append(after['score'] - before['score'])
        dboards = xp.asarray(boards.T.copy())
        vectors = [xp.asarray(value, dtype=np.int16) for value in (aa, bb, na, nb)]
        out = xp.empty(g.n, np.int32)
        grid = ((g.n + 127) // 128,)
        g.delta_kernel(grid, (128,), (dboards, g.tables[0], g.tables[1], *vectors, out, np.int32(g.n)))
        assert np.array_equal(out.get(), expected), 'Device delta differs from independent full-board scoring'
        g.score_kernel(grid, (128,), (dboards, g.tables[0], g.tables[1], out, np.int32(g.n)))
        independent_scores = [validate_board(board, bundle)['score'] for board in boards]
        assert np.array_equal(out.get(), independent_scores), 'Device full-score mismatch'
        # The independently exhaustively checked466 board has no neutral or
        # improving change involving at most two cells.
        g.boards = xp.asarray(base.T.copy()); g.bestboards = g.boards.copy()
        g.scores.fill(466); g.bestscores.fill(466)
        g.positions = xp.asarray(np.argsort(base // 4, axis=1).astype(np.int16).T.copy())
        g.temps.fill(0)
        zero_ms = g.step(32)
        assert np.array_equal(g.boards.get().T, base)
        assert not g.counters[g.n:].get().any()
        g.temps.fill(1.1)
        durations = [g.step(16) for _ in range(4)]
        checked = g.verify_device_scores(range(g.n))
        assert int(g.counters[g.n:].get().sum()) > 0
        best, bestscores = g.bestboards.get().T, g.bestscores.get()
        for board, score in zip(best, bestscores):
            report = validate_board(board, bundle)
            assert report['valid'] and report['score'] == int(score)
        current, inverse = g.boards.get().T, g.positions.get().T
        assert np.array_equal(inverse, np.argsort(current // 4, axis=1)), 'Inverse piece map mismatch'
        # These methods exercise advanced-index buffer writes and shared state.
        ids = g.reseed()
        g.verify_device_scores(ids)
        assert np.array_equal(g.positions.get().T, np.argsort(g.boards.get().T // 4, axis=1))
        path = Path(temp) / 'test-checkpoint.npz'
        g.checkpoint(path); saved = g.boards.get(); g.step(4)
        assert g.resume(path) and np.array_equal(g.boards.get(), saved)
        g.verify_device_scores(range(g.n))
        g.arm_rewards[3] = 10; g.adapt()
        assert int((g.arms == 3).sum()) > g.n // 4
        g.cool(12.0)
        g.checkpoint(path)
        original_payload={key:value.copy() for key,value in np.load(path,allow_pickle=False).items()}
        grown=create_engine(bundle,g.n+32,718291,backend,Path(temp)/'cache',
                            allow_opencl_cpu=allow_opencl_cpu,device_name=device_name)
        grown.initialize([bundle.record_board])
        added_boards=grown.boards.get()[:,g.n:].copy()
        added_rng=grown.rng.get()[g.n:].copy()
        # Corrupt scores must be rejected before making a migration backup.
        bad_payload={key:value.copy() for key,value in original_payload.items()}
        bad_payload['bestscores'][0]-=1
        invalid=Path(temp)/'invalid.npz';np.savez_compressed(invalid,**bad_payload)
        try:
            grown.resume(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed checkpoint was accepted')
        assert not invalid.with_name(f'invalid.replicas-{g.n}.npz').exists()
        assert grown.resume(path)
        assert np.array_equal(grown.boards.get()[:,:g.n],original_payload['boards'])
        assert np.array_equal(grown.bestboards.get()[:,:g.n],original_payload['bestboards'])
        assert np.array_equal(grown.boards.get()[:,g.n:],added_boards)
        assert np.array_equal(grown.rng.get()[g.n:],added_rng)
        for field in ('scores','bestscores','rng'):
            assert np.array_equal(getattr(grown,field).get()[:g.n],original_payload[field])
        assert np.array_equal(grown.counters.get()[:g.n],original_payload['counters'][:g.n])
        assert np.array_equal(grown.counters.get()[grown.n:grown.n+g.n],original_payload['counters'][g.n:])
        assert np.array_equal(grown.arms[:g.n],original_payload['arms'])
        assert np.array_equal(grown.arm_trials,original_payload['arm_trials'])
        assert np.array_equal(grown.arm_rewards,original_payload['arm_rewards'])
        assert np.array_equal(grown.seeds,original_payload['seed_pool'])
        backup=path.with_name(f'{path.stem}.replicas-{g.n}.npz')
        assert backup.read_bytes()==path.read_bytes()
        grown.verify_device_scores(range(grown.n))
        # Make quality ranks nonuniform with legal states so the shrink test
        # verifies preferential preservation, not merely stable equal-score ties.
        replacement=boards[:16]
        grown.boards[:,:16]=grown.xp.asarray(replacement.T.copy())
        grown.bestboards[:,:16]=grown.xp.asarray(replacement.T.copy())
        replacement_scores=np.array([validate_board(board,bundle)['score'] for board in replacement],np.int32)
        grown.scores[:16]=grown.xp.asarray(replacement_scores)
        grown.bestscores[:16]=grown.xp.asarray(replacement_scores)
        grown.positions[:,:16]=grown.xp.asarray(np.argsort(replacement//4,axis=1).astype(np.int16).T.copy())
        grown.step(1);grown.verify_device_scores(range(grown.n));grown.checkpoint(path)
        with np.load(path,allow_pickle=False) as archive:
            large={key:archive[key].copy() for key in archive.files}
        assert len(np.unique(large['bestscores']))>1
        smaller=create_engine(bundle,64,718292,backend,Path(temp)/'cache',
                              allow_opencl_cpu=allow_opencl_cpu,device_name=device_name)
        smaller.initialize([bundle.record_board]);assert smaller.resume(path)
        chosen=np.argsort(-large['bestscores'],kind='stable')[:smaller.n]
        assert np.array_equal(smaller.boards.get(),large['boards'][:,chosen])
        assert np.array_equal(smaller.bestboards.get(),large['bestboards'][:,chosen])
        for field in ('scores','bestscores','rng'):
            assert np.array_equal(getattr(smaller,field).get(),large[field][chosen])
        assert np.array_equal(smaller.counters.get()[:smaller.n],large['counters'][chosen])
        assert np.array_equal(smaller.counters.get()[smaller.n:],large['counters'][grown.n+chosen])
        assert np.array_equal(smaller.arms,large['arms'][chosen])
        assert np.array_equal(smaller.arm_rewards,large['arm_rewards'])
        assert np.array_equal(smaller.positions.get().T,np.argsort(smaller.boards.get().T//4,axis=1))
        assert path.with_name(f'{path.stem}.replicas-{grown.n}.npz').read_bytes()==path.read_bytes()
        smaller.step(1);smaller.verify_device_scores(range(smaller.n))
        result = {'passed': True, **g.backend_details,
                  'diagnostic_cpu': bool(g.runtime.diagnostic_cpu),
                  'delta_cases': g.n, 'adjacent_swap_cases': adjacent, 'rotation_cases': rotations,
                  'zero_temperature_unchanged': True, 'positive_temperature_all_states_verified': checked,
                  'best_states_verified': len(best), 'inverse_piece_map_verified': True,
                  'checkpoint_roundtrip': True, 'reseed_verified': True, 'adaptive_allocation_test': True,
                  'checkpoint_replica_growth': [g.n,grown.n], 'checkpoint_replica_shrink': [grown.n,smaller.n],
                  'migration_backups_verified':True,'malformed_checkpoint_rejected_before_backup':True,
                  'zero_temperature_kernel_ms': zero_ms, 'positive_temperature_kernel_ms': durations,
                  'elapsed_seconds': time.time() - start,
                  'source_sha256': {name: hashlib.sha256((resource_root() / name).read_bytes()).hexdigest()
                                    for name in ('kernels.cu', 'kernels.cl')}}
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('auto', 'cuda', 'opencl'), default='auto')
    parser.add_argument('--replicas', type=int, default=128)
    parser.add_argument('--allow-opencl-cpu', action='store_true', help='Explicitly select a CPU OpenCL device for CI diagnostics only')
    parser.add_argument('--device', help='OpenCL device-name substring (also ETERNITY_OPENCL_DEVICE)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not 64 <= args.replicas <= 4096:
        parser.error('diagnostic replica count must be in64..4096')
    if args.allow_opencl_cpu and args.backend != 'opencl':
        parser.error('--allow-opencl-cpu requires --backend opencl')
    run(args.backend, args.replicas, args.allow_opencl_cpu, args.device, args.output)


if __name__ == '__main__':
    main()
