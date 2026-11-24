"""Bounded, offline CUDA fast-score experiment. Run with the source CUDA environment."""
import time
START = time.perf_counter()
import argparse
from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import sys


def build_tables(problem, colors):
    n, faces, fixed = problem.size, problem.faces, problem.fixed
    fixed_pieces = {state // 4 for state in fixed.values()}
    signatures, cell_sigs = {}, []
    offsets, values, starts, ends = [], [], [], []
    for cell in range(n*n):
        r, c = divmod(cell, n)
        frame = (r == 0, c == n-1, r == n-1, c == 0)
        right = faces[fixed[cell+1]][3] if c+1 < n and cell+1 in fixed else None
        bottom = faces[fixed[cell+n]][0] if r+1 < n and cell+n in fixed else None
        key = (frame, right, bottom, fixed.get(cell))
        if key not in signatures:
            signatures[key] = len(offsets)
            groups = [[] for _ in range(1088)]
            for state, (t, e, b, l) in enumerate(faces):
                if cell in fixed:
                    if state != fixed[cell]:
                        continue
                elif state // 4 in fixed_pieces:
                    continue
                if (t == 0, e == 0, b == 0, l == 0) != frame:
                    continue
                if (right is not None and e != right) or (bottom is not None and b != bottom):
                    continue
                for group in (l*32+t, 1024+l, 1056+t):
                    groups[group].append(state)
            for group in groups:
                offsets.append(len(values))
                if group:
                    starts.append(len(values))
                    values.extend(group)
                    ends.append(len(values))
        cell_sigs.append(signatures[key])
    offsets.append(len(values))
    packed = [t | e << 5 | b << 10 | l << 15 for t,e,b,l in faces]
    depletion = [sum(v in colors for v in face) for face in faces]
    return cell_sigs, offsets, values, starts, ends, packed, depletion


def depletion_schedule(points, cells):
    need = [0] * (cells+1)
    pd, pv = 0, 0.0
    for d, v in points:
        v = float(f'{v:g}')  # the CPU adapter serializes --hsched this way
        for k in range(pd, d+1):
            need[k] = int(v if d == pd else pv + (v-pv)*(k-pd)/(d-pd))
        pd, pv = d, v
    for k in range(pd, cells+1):
        need[k] = int(pv)
    return need


class Search:
    def __init__(self, problem, profile, lanes, seed):
        import cupy as cp
        import numpy as np
        self.cp, self.np = cp, np
        self.n, self.cells, self.lanes, self.seed = problem.size, problem.size**2, lanes, seed
        sig, offsets, values, starts, ends, faces, depletion = build_tables(problem, profile['depletion_colors'])
        self.sig, self.offsets, self.starts, self.ends = (cp.asarray(a,dtype='i') for a in (sig,offsets,starts,ends))
        self.base = cp.asarray(values,dtype='H')
        self.faces = cp.asarray(faces,dtype='I')
        self.depletion = cp.asarray(depletion,dtype='B')
        self.need = cp.asarray(depletion_schedule(profile['depletion_schedule'], self.cells),dtype='H')
        self.breaks = profile['breaks']
        self.allowed = cp.zeros(self.cells,dtype='B')
        self.restart_limit = ((profile['restart_nodes']+1048575)//1048576)*1048576
        self.groups = len(starts)
        self.order = cp.empty((len(values),lanes),dtype='H')
        self.rng = cp.empty((4,lanes), dtype='Q')
        self.restart = cp.zeros(lanes,dtype='i')
        self.active = cp.ones(lanes,dtype='i')
        self.depth = cp.zeros(lanes,dtype='i')
        self.seg = cp.empty((self.cells,lanes),dtype='B')
        self.cursor, self.end = (cp.empty((self.cells,lanes),dtype='i') for _ in range(2))
        self.board = cp.empty((self.cells,lanes),dtype='H')
        self.hsum = cp.zeros((self.cells+1,lanes),dtype='H')
        self.broken = cp.zeros((self.cells+1,lanes),dtype='B')
        self.used = cp.zeros((8,lanes),dtype='I')
        self.nodes, self.restart_nodes = (cp.zeros(lanes,dtype='Q') for _ in range(2))
        self.deepest, self.status = (cp.zeros(lanes,dtype='i') for _ in range(2))
        kernel_path = Path(__file__).with_name('fast_score.cu')
        module = cp.RawModule(code=kernel_path.read_text(),options=('--std=c++17',))
        self.make_orders, self.advance = (module.get_function(s) for s in ('make_orders','advance'))
        self.grid = ((lanes+63)//64,)
        self.block = (64,)
        self.order_seconds = 0.0
        self.search_seconds = 0.0
        self.max_launch_ms = 0.0
        self.order_max_launch_ms = 0.0

    def order_lanes(self):
        np, cp = self.np, self.cp
        for first in range(0,self.groups,64):
            args = (np.int32(self.lanes),np.int32(first),np.int32(min(self.groups,first+64)),
                    np.uint64(self.seed),self.restart,self.active,self.rng,self.starts,self.ends,
                    self.base,self.depletion,self.order)
            before, after = cp.cuda.Event(), cp.cuda.Event()
            before.record()
            self.make_orders(self.grid,self.block,args)
            after.record()
            after.synchronize()
            ms = cp.cuda.get_elapsed_time(before, after)
            self.order_seconds += ms/1000
            self.order_max_launch_ms = max(ms,self.order_max_launch_ms)

    def reset(self, budget, due=None):
        cp = self.cp
        if due is None:
            # All lanes begin each tighter stage with the original order, like new CPU helper workers.
            selected = slice(None)
            self.restart.fill(0)
            self.active.fill(1)
            thresholds = [self.breaks[k*len(self.breaks)//budget] for k in range(budget)]
            self.allowed.set(self.np.asarray([bisect_right(thresholds,d) for d in range(self.cells)],dtype='B'))
        else:
            selected = cp.asarray(due)
            self.restart[selected] += 1
            self.active.fill(0)
            self.active[selected] = 1
        self.depth[selected] = 0
        self.seg[0,selected] = 255
        self.status[selected] = 0
        self.restart_nodes[selected] = 0
        self.used[:,selected] = 0
        self.order_lanes()

    def step(self, steps):
        np, cp = self.np, self.cp
        args = (np.int32(self.lanes),np.int32(self.n),np.int32(steps),np.uint64(self.restart_limit),
                self.sig,self.offsets,self.order,self.faces,self.depletion,self.need,self.allowed,
                self.depth,self.seg,self.cursor,self.end,self.board,self.broken,self.hsum,self.used,
                self.nodes,self.restart_nodes,self.deepest,self.status)
        before, after = cp.cuda.Event(), cp.cuda.Event()
        before.record()
        self.advance(self.grid,self.block,args)
        after.record()
        after.synchronize()
        elapsed = cp.cuda.get_elapsed_time(before,after)
        self.search_seconds += elapsed/1000
        self.max_launch_ms = max(elapsed,self.max_launch_ms)
        return self.status.get(), elapsed


def official(args):
    import cupy as cp
    from exact_model import from_bundle
    from fast_search import default_score_profile
    from validator import load_bundle, validate_board
    bundle = load_bundle()
    problem = from_bundle(bundle)
    profile = default_score_profile(problem)
    if profile['name'] != 'official-ladder-1':
        raise ValueError('The requested source checkout has an unrecognized profile.')
    best, boards, total_restarts = None, [], 0
    budget = len(profile['breaks'])
    search = Search(problem,profile,args.lanes,args.seed)
    search.reset(budget)
    ready = time.perf_counter()
    steps = 128
    stages, steps_seen = 1, 0
    while time.perf_counter()-START < args.seconds:
        flags, ms = search.step(steps)
        steps_seen += 1
        for lane in (flags == 2).nonzero()[0]:
            board = search.board[:,int(lane)].get().astype(int).tolist()
            score = 480-int(search.broken[-1,int(lane)].get())
            check = validate_board(board,bundle)
            if not check['valid'] or check['score'] != score or score < 480-budget:
                raise RuntimeError('CUDA proposed an invalid board or false score')
            stamp = time.perf_counter()-START
            sha = hashlib.sha256(json.dumps(board,separators=(',',':')).encode()).hexdigest()
            record = dict(score=score, board=board, lane=int(lane), sha256=sha, seed=args.seed, wall_seconds=stamp)
            path = args.out / f'board-{score}-{sha}.json'
            path.write_text(json.dumps(record,indent=2)+'\n')
            boards.append({k:v for k,v in record.items() if k!='board'})
            best = max(best or 0,score)
        if best == 480 or (flags == 1).all():
            break
        if (flags == 2).any():
            budget = min(budget-1,480-best-1)
            search.reset(budget)
            stages += 1
        else:
            restart = (flags == 3).nonzero()[0]
            if len(restart):
                search.reset(budget,due=restart)
                total_restarts += len(restart)
        # Keep individual kernels short on the display GPU, with a hard iteration cap.
        if ms > 40:
            steps = max(16,steps//2)
        elif ms < 8:
            steps = min(2048,steps*2)
    nodes = int(search.nodes.sum().get())
    elapsed = time.perf_counter()-START
    gpu = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['name']
    return dict(engine='experimental-cuda-fast-score', seed=args.seed, lanes=args.lanes,
                device=gpu.decode() if isinstance(gpu,bytes) else str(gpu), profile=profile['name'],
                requested_seconds=args.seconds, elapsed_seconds=elapsed, initialization_seconds=ready-START,
                order_kernel_seconds=search.order_seconds, search_kernel_seconds=search.search_seconds,
                max_search_launch_ms=search.max_launch_ms, max_order_launch_ms=search.order_max_launch_ms,
                total_launches=steps_seen, placements=nodes, placements_per_wall_second=nodes/elapsed,
                placements_per_search_kernel_second=nodes/search.search_seconds if search.search_seconds else None,
                deepest_placement=int(search.deepest.max().get()), stages=stages, restarts=total_restarts,
                complete_boards=len(boards), best_score=best, boards=boards)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--seconds',type=float,required=True)
    parser.add_argument('--lanes',type=int,default=2048)
    args=parser.parse_args()
    if not 0 <= args.seed < 2**64 or not 1 <= args.lanes <= 4096 or not 0 < args.seconds <= 86400:
        parser.error('seed must fit uint64, lanes must be in 1..4096, seconds in (0,86400]')
    args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.repo.resolve()))
    os.environ['CUPY_CACHE_DIR']=str((args.out/'kernel-cache').resolve())
    summary=official(args)
    text=json.dumps(summary,indent=2)
    (args.out/'summary.json').write_text(text+'\n')
    print(text)
