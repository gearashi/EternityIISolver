"""CUDA row-major score search with persistent independent device lanes."""
import time
from bisect import bisect_right
from app_paths import resource_root

LANES = 2048


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
        kernel_path = resource_root() / 'fast_score.cu'
        if not kernel_path.is_file():
            raise FileNotFoundError('Missing fast_score.cu. Extract the complete Windows release zip '
                                    'or restore fast_score.cu from the matching source release.')
        module = cp.RawModule(code=kernel_path.read_text(encoding='utf-8'), options=('--std=c++17',))
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


def run_ladder(problem, profile, seed, lanes, started, seconds, cache_dir, on_board,
               should_stop=None, on_progress=None):
    """Start all lanes fresh. Deliver each complete candidate before tightening its budget."""
    from fast_search import score_board
    try:
        from gpu_backends import BackendUnavailable, _cuda_problem, open_cuda
    except ImportError as exc:
        from app_paths import dependency_setup_hint
        raise RuntimeError(f'CUDA fast-score dependencies could not load ({exc}). {dependency_setup_hint()}') from exc

    cp = open_cuda(cache_dir, kernel='fast_score.cu')
    try:
        search = Search(problem, profile, lanes, seed)
    except Exception as exc:
        raise BackendUnavailable(_cuda_problem(exc, 'building the CUDA fast-score kernels and search tables')) from exc
    name = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['name']
    device = name.decode() if isinstance(name, bytes) else str(name)
    edges = 2 * problem.size * (problem.size - 1)
    budget = len(profile['breaks'])
    best = None
    best_board = None
    complete = restarts = launches = 0
    stages = 1
    steps = 128
    search.reset(budget)
    ready = time.perf_counter()
    report_time = ready
    outcome = 'timeout'

    def counters():
        elapsed = time.perf_counter() - started
        placements = int(search.nodes.sum().get())
        return dict(branches=placements, nodes_per_second=placements/max(elapsed, 1e-9),
                    max_depth=int(search.deepest.max().get()), complete_boards=complete,
                    restarts=restarts, target_score=edges-budget, engine_best_score=best,
                    gpu=device, backend='cuda', phase='searching')

    if on_progress:
        on_progress(counters())
    while time.perf_counter() - started < seconds:
        if should_stop and should_stop():
            outcome = 'stopped'
            break
        flags, ms = search.step(steps)
        launches += 1
        completed_lanes = (flags == 2).nonzero()[0]
        for lane_number in completed_lanes:
            lane = int(lane_number)
            board = search.board[:, lane].get().astype(int).tolist()
            score = edges - int(search.broken[-1, lane].get())
            if score < edges-budget or score_board(problem, board) != score:
                raise RuntimeError('CUDA fast-score produced a board that failed CPU validation')
            on_board(dict(board=board, score=score, lane=lane, wall_seconds=time.perf_counter()-started))
            complete += 1
            if best is None or score > best:
                best, best_board = score, board
        if best == edges:
            outcome = 'solved'
            break
        if (flags == 1).all():
            outcome = 'unchanged'
            break
        if len(completed_lanes):
            budget = min(budget-1, edges-best-1)
            search.reset(budget)
            stages += 1
        else:
            due = (flags == 3).nonzero()[0]
            if len(due):
                search.reset(budget, due=due)
                restarts += len(due)
        if on_progress and (len(completed_lanes) or time.perf_counter() >= report_time + .5):
            on_progress(counters())
            report_time = time.perf_counter()
        if ms > 40:
            steps = max(16,steps//2)
        elif ms < 8:
            steps = min(2048,steps*2)
    final = counters()
    final.update(outcome=outcome, board=best_board if outcome == 'solved' else None,
                 score=best, coverage_verified=False, proof_scope='heuristic_break_schedule',
                 elapsed_seconds=time.perf_counter()-started, initialization_seconds=ready-started,
                 order_kernel_seconds=search.order_seconds, search_kernel_seconds=search.search_seconds,
                 max_order_launch_ms=search.order_max_launch_ms, max_search_launch_ms=search.max_launch_ms,
                 total_launches=launches, stages=stages)
    return final


def solve(problem, seconds, seed, workers, should_stop, on_progress, on_improvement, cache_dir):
    from fast_search import default_score_profile
    if workers != 1:
        raise ValueError('CUDA fast-score uses one coordinator and 2,048 GPU lanes')
    if seconds <= 0 or should_stop():
        return {'outcome': 'timeout' if seconds <= 0 else 'stopped', 'branches': 0}
    return run_ladder(problem, default_score_profile(problem), seed, LANES, time.perf_counter(),
                      seconds, cache_dir, on_improvement, should_stop, on_progress)
