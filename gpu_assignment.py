"""Bounded integer GPU matrices for exact independent-set assignment repair.

Each CUDA/OpenCL lane evaluates four legal rotations for one cell/piece pair.
The CPU subsequently solves each matching and independently validates the board.
Device kernels contain no search counters, network operations, or submissions.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
import itertools
import numpy as np
from exact_model import check_solution, constraint_edges, frame_ok, grid_edges


class AssignmentCancelled(RuntimeError):
    pass


class AssignmentCallbackError(RuntimeError):
    pass


# The same scalar integer body is compiled for both device APIs. Four neighboring
# cells stay fixed because every supplied cell set is independent.
_KERNEL = r'''
INLINE unsigned int assignment_tie_key(unsigned int seed, unsigned int cell, unsigned int state) {
    unsigned int x = seed ^ (cell * 0x9e3779b9U) ^ (state * 0x85ebca6bU);
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
KERNEL void assignment_costs(
    GLOBAL const int *faces, GLOBAL const unsigned char *allowed,
    GLOBAL const int *boards, GLOBAL const int *cells,
    GLOBAL const int *pieces, GLOBAL const int *widths,
    const int size, const int count, const int width,
    const int use_ties, const unsigned int tie_seed,
    GLOBAL int *weights, GLOBAL int *states) {
    int index = INDEX;
    int total = count * width * width;
    if (index >= total) return;
    int b = index / (width * width);
    int row = (index / width) % width;
    int col = index % width;
    weights[index] = -1;
    states[index] = -1;
    if (row >= widths[b] || col >= widths[b]) return;
    int n = size * size;
    int cell = cells[b * width + row];
    int piece = pieces[b * width + col];
    int current = boards[b * n + cell];
    int best = -1, best_state = -1;
    unsigned int best_tie = 0;
    for (int rot = 0; rot < 4; ++rot) {
        int state = 4 * piece + rot;
        if (!allowed[cell * 4 * n + state]) continue;
        int weight = 0;
        if (cell >= size)
            weight += faces[4 * state] == faces[4 * boards[b*n + cell-size] + 2];
        if (cell % size < size-1)
            weight += faces[4 * state + 1] == faces[4 * boards[b*n + cell+1] + 3];
        if (cell < n-size)
            weight += faces[4 * state + 2] == faces[4 * boards[b*n + cell+size]];
        if (cell % size > 0)
            weight += faces[4 * state + 3] == faces[4 * boards[b*n + cell-1] + 1];
        unsigned int tie = use_ties ? assignment_tie_key(tie_seed, (unsigned int)cell, (unsigned int)state) : 0;
        if (weight > best || (weight == best &&
                (use_ties ? tie > best_tie : state == current))) {
            best = weight; best_state = state; best_tie = tie;
        }
    }
    weights[index] = best; states[index] = best_state;
}
'''


def _positive(name, value, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f'{name} must be an integer in 1..{maximum}')
    return value


def _tie_seed(value):
    if value is not None and (type(value) is not int or not 0 <= value < 2**32):
        raise ValueError('tie_seed must be None or a uint32 integer')
    return value


def _tie_key(seed, cell, state):
    """Exact uint32 arithmetic shared with the CUDA/OpenCL scalar body."""
    x = (seed ^ ((cell * 0x9e3779b9) & 0xffffffff) ^
         ((state * 0x85ebca6b) & 0xffffffff)) & 0xffffffff
    x ^= x >> 16; x = (x * 0x7feb352d) & 0xffffffff; x ^= x >> 15
    x = (x * 0x846ca68b) & 0xffffffff; x ^= x >> 16
    return x


def assignment_arc_costs(matrix):
    """Min-cost arcs preserving primary score, with optional seeded tie rewards.

    Rotation ties are selected first, independently for each cell/piece pair.
    This separate assignment secondary objective uses physical piece identities.
    For n assigned pieces every total secondary reward lies in [0,255*n];
    multiplying primary score by 255*n+1 makes one match strictly dominate all
    secondary choices. All arithmetic is signed int64; with n<=256 and edge
    weights<=4, each complete objective is below 67 million in magnitude.
    Illegal arcs remain excluded by the caller's states>=0 mask.
    """
    seed = _tie_seed(matrix.get('tie_seed'))
    weights = np.asarray(matrix['weights']); states = np.asarray(matrix['states'])
    cells = matrix['cells']; n = len(cells)
    if (n > 256 or weights.shape != (n,n) or states.shape != (n,n)
            or weights.dtype.kind not in 'iu' or states.dtype.kind not in 'iu'
            or np.any(weights < -1) or np.any(weights > 4)):
        raise ValueError('Invalid bounded assignment matrix')
    costs = -weights.astype(np.int64)
    if seed is None or n == 0:
        return costs
    board = matrix['board']
    row = np.asarray(cells, dtype=np.uint32)[:,None]
    piece = np.asarray([board[c]//4 for c in cells], dtype=np.uint32)[None,:]
    x = np.uint32(seed ^ 0xa511e9b3) ^ (row * np.uint32(0x9e3779b9)) ^ (piece * np.uint32(0x85ebca6b))
    x ^= x >> np.uint32(16); x *= np.uint32(0x7feb352d); x ^= x >> np.uint32(15)
    x *= np.uint32(0x846ca68b); x ^= x >> np.uint32(16)
    reward = (x & np.uint32(255)).astype(np.int64)
    return costs * (255*n+1) - reward


def _job(problem, board, cells, *, structural_problem=None):
    board = list(board); cells = list(cells)
    structural=replace(problem,edges=()) if structural_problem is None else structural_problem
    if not check_solution(structural, board):
        raise ValueError('Assignment parent violates pieces, domains, frame or fixed clues')
    if len(cells) != len(set(cells)) or any(type(c) is not int or not 0 <= c < problem.size**2 for c in cells):
        raise ValueError('Invalid independent cells')
    chosen = set(cells)
    if chosen.intersection(problem.fixed) or any(a in chosen and b in chosen for a,b,_,_ in grid_edges(problem.size)):
        raise ValueError('Assignment cells must be nonadjacent and exclude fixed clues')
    return board, cells


def cpu_cost_matrix(problem, board, cells, *, should_stop=None, _structural_problem=None, tie_seed=None):
    """Scalar reference implementation, explicitly labeled CPU in its result."""
    tie_seed = _tie_seed(tie_seed)
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
        raise ValueError('Assignment repair requires all grid edges')
    board,cells = _job(problem,board,cells,structural_problem=_structural_problem)
    count = len(cells)
    weights = np.full((count,count),-1,dtype=np.int32)
    states = np.full_like(weights,-1)
    neighbors = {c:[] for c in cells}
    for a,b,sa,sb in grid_edges(problem.size):
        if a in neighbors: neighbors[a].append((b,sa,sb))
        if b in neighbors: neighbors[b].append((a,sb,sa))
    for i,cell in enumerate(cells):
        if should_stop and should_stop(): raise AssignmentCancelled('Assignment stopped')
        allowed = set(problem.domains[cell])
        for j,other in enumerate(cells):
            if j % 32 == 0 and should_stop and should_stop(): raise AssignmentCancelled('Assignment stopped')
            piece = board[other]//4;best=-1;best_state=-1;best_tie=-1
            for state in range(4*piece,4*piece+4):
                if state not in allowed or not frame_ok(problem.size,cell,problem.faces[state]): continue
                weight = sum(problem.faces[state][sa] == problem.faces[board[b]][sb] for b,sa,sb in neighbors[cell])
                tie = _tie_key(tie_seed,cell,state) if tie_seed is not None else 0
                if weight > best or (weight == best and
                        (tie > best_tie if tie_seed is not None else state == board[cell])):
                    best=weight;best_state=state;best_tie=tie
            weights[i,j]=best;states[i,j]=best_state
    return {'weights':weights,'states':states,'backend':'cpu','device':'CPU',
            'board':board,'cells':cells,'tie_seed':tie_seed}


class AssignmentEvaluator:
    """One reusable device model with bounded matrix batches and explicit fallback.

    Initialization is lazy, so a stopped caller does not load a GPU runtime.
    Every submitted kernel is bounded by max_pairs (default 262144 lanes, hard
    maximum 1048576); kernels/native calls are not preemptible. Cancellation is
    checked during preparation, before launch, and after device synchronization.
    The byte limit bounds arrays for one batch, excluding driver/compiler memory.
    """
    def __init__(self, problem, *, backend='auto', cache_dir=None,
                 allow_cpu_fallback=False, batch_size=16,
                 max_batch_bytes=16*1024*1024, max_pairs=262144, should_stop=None,
                 _diagnostic_opencl_cpu=False):
        if backend not in ('auto','cuda','opencl','cpu'):
            raise ValueError('backend must be auto, cuda, opencl or cpu')
        if type(allow_cpu_fallback) is not bool: raise ValueError('allow_cpu_fallback must be boolean')
        if type(_diagnostic_opencl_cpu) is not bool: raise ValueError('_diagnostic_opencl_cpu must be boolean')
        if _diagnostic_opencl_cpu and (backend != 'opencl' or allow_cpu_fallback):
            raise ValueError('CPU OpenCL diagnostics require explicit opencl without fallback')
        if should_stop is not None and not callable(should_stop): raise ValueError('should_stop must be callable')
        if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
            raise ValueError('Assignment repair requires all grid edges')
        # Snapshot mutable input containers. Later caller mutations cannot alter
        # legality while the device continues to use a previously encoded model.
        self._problem=replace(problem,faces=tuple(tuple(f) for f in problem.faces),
                             domains=tuple(tuple(d) for d in problem.domains),fixed=MappingProxyType(dict(problem.fixed)),
                             edges=None if problem.edges is None else tuple(tuple(e) for e in problem.edges))
        self._structural_problem=replace(self._problem,edges=())
        self._canonical_edges=grid_edges(self._problem.size)
        self._snapshot_edges=tuple(tuple(e) for e in constraint_edges(self._problem))
        # Private opt-in is used only by the bounded diagnostic entry point.
        # Dashboard/search callers never request a CPU OpenCL device.
        self._diagnostic_opencl_cpu=_diagnostic_opencl_cpu
        self.requested=backend;self.backend='uninitialized';self.device=None
        self.cache_dir=cache_dir;self.allow_cpu_fallback=allow_cpu_fallback
        self.batch_size=_positive('batch_size',batch_size,64)
        self.max_batch_bytes=_positive('max_batch_bytes',max_batch_bytes,64*1024*1024)
        self.max_pairs=_positive('max_pairs',max_pairs,1048576)
        self.should_stop=should_stop;self.runtime=None;self.kernel=None
        self.batches=0;self.pairs=0;self.kernel_seconds=0.;self.fallback_reason=None

    @property
    def problem(self):
        """Read-only fully detached device model; its clues are read-only too."""
        return self._problem

    @property
    def structural_problem(self):
        """Validated immutable model view omitting only adjacency equality."""
        return self._structural_problem

    def structurally_valid(self, board):
        """Validate every placement against the detached immutable rules.

        This omits only adjacency equality, so complete repair working boards
        may contain mismatches. It does not rebuild or weaken the rules. Callers
        retaining a mutable source problem must check matches_problem at their
        operation boundaries or operate entirely on the detached snapshot.
        """
        return check_solution(self.structural_problem, board)

    def matches_problem(self, candidate):
        """Exact typed equivalence to the detached model without serialization.

        JSON identity's sequence order and list/tuple normalization are retained.
        Immutable rows shared with the validated snapshot can be compared by
        identity safely; mutable source rows are always checked element by
        element. No whole-problem object identity skips a mutation check.
        """
        def same_rows(rows, reference):
            if type(rows) not in (tuple,list) or len(rows)!=len(reference): return False
            for row,known in zip(rows,reference):
                if row is known: continue  # Known rows are tuples of plain ints.
                if type(row) not in (tuple,list) or len(row)!=len(known): return False
                if any(type(value) is not int or value!=wanted for value,wanted in zip(row,known)): return False
            return True
        try:
            if type(candidate.size) is not int or candidate.size!=self.problem.size: return False
            if not same_rows(candidate.faces,self.problem.faces): return False
            if not same_rows(candidate.domains,self.problem.domains): return False
            fixed=candidate.fixed;expected=self.problem.fixed
            if not isinstance(fixed,Mapping) or len(fixed)!=len(expected): return False
            for cell,state in fixed.items():
                if type(cell) is not int or type(state) is not int or expected.get(cell)!=state: return False
            edges=self._canonical_edges if candidate.edges is None else candidate.edges
            return same_rows(edges,self._snapshot_edges)
        except (AttributeError,TypeError,ValueError):
            return False

    def _check(self):
        try: stopped=bool(self.should_stop and self.should_stop())
        except Exception as exc: raise AssignmentCallbackError('Assignment cancellation callback failed') from exc
        if stopped: raise AssignmentCancelled('Assignment stopped')

    def estimated_bytes(self, count, width):
        n=self.problem.size**2
        static=(4*n*4*4)+(n*4*n)
        return static+4*(count*n+2*count*width+count+2*count*width*width)

    def plan_batches(self, widths):
        """Return half-open slices; reject even one matrix exceeding either cap."""
        widths=list(widths)
        if any(type(w) is not int or w<0 or w>self.problem.size**2 for w in widths):
            raise ValueError('Invalid matrix width')
        result=[];start=0;largest=0
        for i,width in enumerate(widths):
            if self.estimated_bytes(1,width)>self.max_batch_bytes or width*width>self.max_pairs:
                raise ValueError('One assignment matrix exceeds the configured batch limit')
            new_largest=max(largest,width);count=i-start+1
            if count>self.batch_size or count*new_largest*new_largest>self.max_pairs or self.estimated_bytes(count,new_largest)>self.max_batch_bytes:
                result.append((start,i));start=i;largest=width
            else: largest=new_largest
        if start<len(widths): result.append((start,len(widths)))
        return result

    def _initialize(self):
        self._check()
        if self.backend!='uninitialized': return
        if self.requested=='cpu': self.backend='cpu';self.device='CPU';return
        from gpu_backends import BackendUnavailable, make_backend
        try:
            runtime=make_backend(self.requested,cache_dir=self.cache_dir,
                                 allow_opencl_cpu=self._diagnostic_opencl_cpu)
            self._check()
            if runtime.diagnostic_cpu and not (self._diagnostic_opencl_cpu and runtime.name=='opencl'):
                raise BackendUnavailable('CPU diagnostic device is not a GPU')
            if runtime.name=='cuda':
                code='#define INLINE __device__ __forceinline__\n#define KERNEL extern "C" __global__\n#define GLOBAL\n#define INDEX (blockIdx.x*blockDim.x+threadIdx.x)\n'+_KERNEL
                module=runtime.xp.RawModule(code=code,options=('--std=c++17',),name_expressions=('assignment_costs',))
                kernel=module.get_function('assignment_costs')
            else:
                code='#define INLINE inline\n#define KERNEL __kernel\n#define GLOBAL __global\n#define INDEX get_global_id(0)\n'+_KERNEL
                directory=Path(self.cache_dir) if self.cache_dir else __import__('gpu_backends').default_cache_dir()
                directory=directory/'assignment-opencl';directory.mkdir(parents=True,exist_ok=True)
                module=runtime.cl.Program(runtime.context,code).build(options=['-cl-std=CL1.2'],cache_dir=str(directory))
                native=runtime.cl.Kernel(module,'assignment_costs')
                def kernel(grid,block,args):
                    from gpu_backends import OpenCLArray
                    values=tuple(value.buffer if isinstance(value,OpenCLArray) else value for value in args)
                    return native(runtime.queue,(int(grid[0])*int(block[0]),),None,*values)
            self._check()
        except (AssignmentCancelled,AssignmentCallbackError): raise
        except Exception as exc:
            error=exc if isinstance(exc,BackendUnavailable) else BackendUnavailable(f'Assignment GPU initialization failed: {exc}')
            if not self.allow_cpu_fallback: raise error from exc
            self.backend='cpu';self.device='CPU';self.fallback_reason=str(error);return
        p=self.problem;n=p.size**2
        allowed=np.zeros((n,4*n),dtype=np.uint8)
        for cell,domain in enumerate(p.domains):
            self._check()
            for state in domain:
                if frame_ok(p.size,cell,p.faces[state]) and (cell not in p.fixed or p.fixed[cell]==state):
                    allowed[cell,state]=1
        device_faces=runtime.xp.asarray(np.asarray(p.faces,dtype=np.int32))
        device_allowed=runtime.xp.asarray(allowed)
        self._check()
        self.runtime=runtime;self.kernel=kernel;self.module=module
        self.device_faces=device_faces;self.device_allowed=device_allowed
        self.backend=runtime.name;self.device=runtime.device

    def evaluate(self, neighborhoods, *, reuse_validation=True, tie_seed=None):
        """Evaluate at most 64 (board, independent_cells) pairs in bounded chunks.

        Matrices are ordered exactly like input; -1 denotes an illegal pairing.
        Initialization failures fall back only with allow_cpu_fallback=True.
        Device execution errors always propagate, preventing silent CPU labeling.
        """
        if type(reuse_validation) is not bool: raise ValueError('reuse_validation must be boolean')
        tie_seed = _tie_seed(tie_seed)
        self._check()
        structure=self.structural_problem if reuse_validation else None
        jobs=list(itertools.islice(neighborhoods,65))
        if len(jobs)>64: raise ValueError('At most 64 neighborhoods per evaluate call')
        prepared=[]
        for board,cells in jobs:
            self._check();prepared.append(_job(self.problem,board,cells,structural_problem=structure))
        slices=self.plan_batches([len(cells) for _,cells in prepared])
        if not prepared: return []
        self._initialize();result=[]
        for start,end in slices:
            self._check();batch=prepared[start:end]
            if self.backend=='cpu':
                matrices=[cpu_cost_matrix(self.problem,board,cells,should_stop=self.should_stop,
                                          _structural_problem=structure,tie_seed=tie_seed) for board,cells in batch]
            else: matrices=self._device_batch(batch) if tie_seed is None else self._device_batch(batch,tie_seed=tie_seed)
            self._check();result.extend(matrices)
            self.batches+=1;self.pairs+=sum(len(cells)**2 for _,cells in batch)
        return result

    def _device_batch(self, jobs, *, tie_seed=None):
        xp=self.runtime.xp;count=len(jobs);width=max(len(cells) for _,cells in jobs)
        if width==0:
            return [dict(weights=np.empty((0,0),dtype=np.int32),states=np.empty((0,0),dtype=np.int32),
                         board=board,cells=cells,backend=self.backend,device=self.device,tie_seed=tie_seed) for board,cells in jobs]
        boards=np.asarray([board for board,_ in jobs],dtype=np.int32)
        cells=np.full((count,width),-1,dtype=np.int32);pieces=np.full_like(cells,-1)
        widths=np.asarray([len(c) for _,c in jobs],dtype=np.int32)
        for i,(board,chosen) in enumerate(jobs):
            self._check();cells[i,:len(chosen)]=chosen;pieces[i,:len(chosen)]=[board[c]//4 for c in chosen]
        self._check()
        db=xp.asarray(boards);dc=xp.asarray(cells);dp=xp.asarray(pieces);dw=xp.asarray(widths)
        weights=xp.empty((count,width,width),dtype=np.int32);states=xp.empty((count,width,width),dtype=np.int32)
        self._check();total=count*width*width
        ms=self.runtime.timed(self.kernel,((total+127)//128,),(128,),
                              (self.device_faces,self.device_allowed,db,dc,dp,dw,
                               np.int32(self.problem.size),np.int32(count),np.int32(width),
                               np.int32(tie_seed is not None),np.uint32(tie_seed or 0),weights,states))
        self.kernel_seconds+=ms/1000
        self._check();host_weights=weights.get();host_states=states.get();result=[]
        for i,(board,chosen) in enumerate(jobs):
            n=len(chosen)
            result.append(dict(weights=host_weights[i,:n,:n].copy(),states=host_states[i,:n,:n].copy(),
                               board=board,cells=chosen,backend=self.backend,device=self.device,tie_seed=tie_seed))
        return result

    def stats(self):
        return {'cost_backend':self.backend,'cost_device':self.device,'cost_batches':self.batches,
                'cost_pairs':self.pairs,'cost_kernel_seconds':self.kernel_seconds,
                'cost_fallback_reason':self.fallback_reason,
                'cost_diagnostic_cpu':bool(self.runtime is not None and self.runtime.diagnostic_cpu)}


def diagnose(*, backend='auto', size=6, count=16, seed=7, cache_dir=None, seconds=30, bulk_arcs=True,
             allow_opencl_cpu=False, tie_seed=None):
    """Bounded synthetic device/oracle comparison, never a production search."""
    import math
    import random
    import time
    from exact_model import generated_problem
    from smart_repair import assignment_repair_batch, independent_cells, structurally_valid
    _positive('size',size,16);_positive('count',count,64);_tie_seed(tie_seed)
    if type(allow_opencl_cpu) is not bool: raise ValueError('allow_opencl_cpu must be boolean')
    if allow_opencl_cpu and backend!='opencl':
        raise ValueError('CPU OpenCL diagnostics require explicit backend=opencl')
    if type(bulk_arcs) is not bool: raise ValueError('bulk_arcs must be boolean')
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=120:
        raise ValueError('Diagnostic seconds must be in (0,120]')
    if type(seed) is not int or not 0<=seed<2**32: raise ValueError('seed must be a uint32 integer')
    started=time.monotonic();deadline=started+seconds
    def stopped(): return time.monotonic()>=deadline
    problem,answer=generated_problem(size,5,seed,clue_count=min(2,size*size))
    rng=random.Random(seed);jobs=[]
    for i in range(count):
        board=list(answer);groups={}
        for cell in range(size*size):
            if cell not in problem.fixed:
                groups.setdefault(sum(color==0 for color in problem.faces[board[cell]]),[]).append(cell)
        for cells in groups.values():
            pieces=[board[cell]//4 for cell in cells];rng.shuffle(pieces)
            for cell,piece in zip(cells,pieces):
                board[cell]=rng.choice([s for s in problem.domains[cell] if s//4==piece])
        jobs.append((board,independent_cells(problem,rng.randrange(2**32),parity=i if i<2 else None)))
    cpu=AssignmentEvaluator(problem,backend='cpu',batch_size=min(count,64),should_stop=stopped)
    cpu_started=time.monotonic();expected=cpu.evaluate(jobs,tie_seed=tie_seed);cpu_seconds=time.monotonic()-cpu_started
    gpu=AssignmentEvaluator(problem,backend=backend,cache_dir=cache_dir,batch_size=min(count,64),should_stop=stopped,
                            _diagnostic_opencl_cpu=allow_opencl_cpu)
    cold_started=time.monotonic();actual=gpu.evaluate(jobs,tie_seed=tie_seed);cold_seconds=time.monotonic()-cold_started
    warm_started=time.monotonic();warm=gpu.evaluate(jobs,tie_seed=tie_seed);warm_seconds=time.monotonic()-warm_started
    for target in (actual,warm):
        for oracle,got in zip(expected,target):
            if not np.array_equal(oracle['weights'],got['weights']) or not np.array_equal(oracle['states'],got['states']):
                raise RuntimeError('GPU assignment matrix disagrees with scalar CPU oracle')
    # Warm matching import/native code before measuring either complete path.
    assignment_repair_batch(problem,jobs,evaluator=cpu,should_stop=stopped,bulk_arcs=bulk_arcs,**({} if tie_seed is None else {"tie_seed":tie_seed}))
    cpu_match_started=time.monotonic()
    cpu_repaired=assignment_repair_batch(problem,jobs,evaluator=cpu,should_stop=stopped,bulk_arcs=bulk_arcs,**({} if tie_seed is None else {"tie_seed":tie_seed}))
    cpu_match_seconds=time.monotonic()-cpu_match_started
    match_started=time.monotonic()
    repaired=assignment_repair_batch(problem,jobs,evaluator=gpu,should_stop=stopped,bulk_arcs=bulk_arcs,**({} if tie_seed is None else {"tie_seed":tie_seed}))
    gpu_match_seconds=time.monotonic()-match_started
    if len(repaired)!=len(jobs) or not all(item['optimal'] and structurally_valid(problem,item['board']) for item in repaired):
        raise RuntimeError('GPU assignment matching did not complete with valid boards')
    if [item['score'] for item in repaired]!=[item['score'] for item in cpu_repaired]:
        raise RuntimeError('GPU and CPU end-to-end assignments disagree')
    return {'passed':True,'synthetic':True,'size':size,'neighborhoods':count,'seed':seed,'tie_seed':tie_seed,
            'diagnostic_cpu':bool(gpu.runtime is not None and gpu.runtime.diagnostic_cpu),
            'validation_scope':'CPU OpenCL kernel correctness only; not GPU certification or performance'
                               if allow_opencl_cpu else 'Bounded assignment kernel and CPU oracle comparison',
            'cpu_matrix_seconds':cpu_seconds,'gpu_cold_matrix_seconds':cold_seconds,
            'gpu_warm_matrix_seconds':warm_seconds,'gpu_matrix_and_cpu_matching_seconds':gpu_match_seconds,
            'cpu_matrix_and_cpu_matching_seconds':cpu_match_seconds,'bulk_arcs_requested':bulk_arcs,
            'matching_arc_modes':sorted({item.get('matching_arc_mode','empty') for item in repaired}),
            'timing_note':'Both end-to-end paths use identical jobs after matching imports/native code warm up',
            'elapsed_seconds':time.monotonic()-started,'all_matrices_equal':True,'all_repairs_valid':True,
            'input_scores':[item['reference_score'] for item in repaired],
            'output_scores':[item['score'] for item in repaired],**gpu.stats()}


def main(argv=None):
    import argparse
    import json
    parser=argparse.ArgumentParser(description='Bounded GPU assignment matrices versus scalar CPU oracle')
    parser.add_argument('--backend',choices=('auto','cuda','opencl'),default='auto')
    parser.add_argument('--size',type=int,default=6)
    parser.add_argument('--count',type=int,default=16,help='Independent neighborhoods (1..64)')
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--tie-seed',type=int,help='Explore equal-score rotation and matching choices with a uint32 seed')
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--scalar-arcs',action='store_true',help='Ablate bulk native matching arc loading')
    parser.add_argument('--allow-opencl-cpu',action='store_true',
                        help='Explicit OpenCL CPU correctness diagnostic only; requires --backend opencl')
    parser.add_argument('--output',type=Path,help='Save the bounded diagnostic report as JSON')
    args=parser.parse_args(argv)
    try:
        result=diagnose(backend=args.backend,size=args.size,count=args.count,seed=args.seed,
                        cache_dir=args.cache_dir,seconds=args.seconds,bulk_arcs=not args.scalar_arcs,
                        allow_opencl_cpu=args.allow_opencl_cpu,tie_seed=args.tie_seed)
    except Exception as exc:
        result={'passed':False,'error':str(exc)}
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(encoded,encoding='utf-8')
        except OSError as exc:
            print(json.dumps({'passed':False,'error':'Cannot save diagnostic report: '+str(exc)},indent=2));return 1
    print(encoded,end='');return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
