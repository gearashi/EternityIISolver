"""Independent finite oracles for the smart repair neighborhoods."""
import itertools
import random
import sqlite3
import struct
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from pathlib import Path
from exact_model import generated_problem, grid_edges, make_problem
from smart_repair import (assignment_repair,cached_parents,cycle_recombine,
                          independent_cells,permutation_cycles,score,structurally_valid,solve)


def scrambled(problem, answer, seed):
    rng=random.Random(seed)
    board=list(answer)
    groups={}
    for cell in range(len(board)):
        if cell in problem.fixed: continue
        kind=tuple(side==0 for side in problem.faces[board[cell]])
        # Corner/edge/interior types; legal rotations will orient the frame.
        groups.setdefault(sum(kind),[]).append(cell)
    for cells in groups.values():
        pieces=[board[c]//4 for c in cells];rng.shuffle(pieces)
        for cell,piece in zip(cells,pieces):
            board[cell]=rng.choice([s for s in problem.domains[cell] if s//4==piece])
    assert structurally_valid(problem,board)
    return board


class SmartRepairTests(unittest.TestCase):
    def test_matching_agrees_with_exhaustive_permutations_and_rotations(self):
        for seed in range(5):
            p,answer=generated_problem(3,4,seed,clue_count=1)
            board=scrambled(p,answer,seed+99)
            cells=independent_cells(p,seed)[:4]
            pieces=[board[c]//4 for c in cells]
            optimum=score(p,board)
            for perm in itertools.permutations(pieces):
                options=[[s for s in p.domains[c] if s//4==piece] for c,piece in zip(cells,perm)]
                for states in itertools.product(*options):
                    candidate=list(board)
                    for c,s in zip(cells,states): candidate[c]=s
                    if structurally_valid(p,candidate): optimum=max(optimum,score(p,candidate))
            for bulk_arcs in (False,True):
                result=assignment_repair(p,board,cells,bulk_arcs=bulk_arcs)
                self.assertTrue(result['optimal']);self.assertEqual(result['score'],optimum)
                self.assertEqual(result['matching_arc_mode'],'bulk' if bulk_arcs else 'scalar')
                self.assertTrue(all(result['board'][c]==board[c] for c in range(9) if c not in cells))

    def test_cycle_recombination_agrees_with_all_component_choices(self):
        for seed in range(8):
            p,answer=generated_problem(3,3,seed,clue_count=1)
            a=scrambled(p,answer,seed+12);b=scrambled(p,answer,seed+23)
            cycles=permutation_cycles(p,a,b);optimum=-1
            for mask in range(1<<len(cycles)):
                candidate=list(a)
                for i,cycle in enumerate(cycles):
                    if (mask>>i)&1:
                        for c in cycle: candidate[c]=b[c]
                self.assertTrue(structurally_valid(p,candidate))
                optimum=max(optimum,score(p,candidate))
            result=cycle_recombine(p,a,b,seconds=1)
            self.assertTrue(result['optimal']);self.assertEqual(result['score'],optimum)

    def test_invalid_clues_cells_and_omitted_edges_rejected(self):
        p,board=generated_problem(3,4,1,clue_count=1)
        clue=next(iter(p.fixed))
        for cells in ([0,1],[0,0],[clue],[True],[-1],[9]):
            with self.assertRaises(ValueError): assignment_repair(p,board,cells)
        regional=make_problem(3,p.faces,p.fixed,p.domains,edges=())
        with self.assertRaises(ValueError): assignment_repair(regional,board,[])
        invalid=list(board);invalid[0]=invalid[1]
        with self.assertRaises(ValueError): cycle_recombine(p,board,invalid)

    def test_interruption_and_timeout_retain_valid_parent(self):
        p,answer=generated_problem(3,4,4)
        a=scrambled(p,answer,6);b=scrambled(p,answer,9)
        r=assignment_repair(p,a,independent_cells(p),should_stop=lambda:True)
        self.assertEqual(r['termination'],'stopped');self.assertEqual(r['board'],a)
        r=cycle_recombine(p,a,b,seconds=0)
        self.assertGreaterEqual(r['score'],max(score(p,a),score(p,b)))
        r=cycle_recombine(p,a,b,should_stop=lambda:True)
        self.assertEqual(r['termination'],'stopped')

    def test_mixed_sets_are_independent_and_can_cross_parity(self):
        p,_=generated_problem(6,3,1,clue_count=2)
        found=False
        for seed in range(10):
            cells=set(independent_cells(p,seed))
            self.assertFalse(cells.intersection(p.fixed))
            self.assertFalse(any(a in cells and b in cells for a,b,_,_ in grid_edges(6)))
            found |= len({(c//6+c%6)%2 for c in cells})==2
        self.assertTrue(found)

    def test_cache_read_is_local_validated_and_read_only(self):
        p,board=generated_problem(3,4,2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.sqlite3'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE boards(placement BLOB,score INTEGER,public_sha TEXT)')
                db.execute('INSERT INTO boards VALUES(?,?,?)',(struct.pack('<9H',*board),12,'a'))
                db.execute('INSERT INTO boards VALUES(?,?,?)',(b'bad',13,'b'))
            db.close()
            before=path.read_bytes();parents=cached_parents(p,path)
            self.assertEqual(parents,[board]);self.assertEqual(path.read_bytes(),before)

    def test_cycle_cancellation_repeats_through_native_start_race(self):
        from ortools.sat.python import cp_model
        p,answer=generated_problem(3,3,0,clue_count=1)
        a=scrambled(p,answer,12);b=scrambled(p,answer,23)
        active=[cycle for cycle in permutation_cycles(p,a,b) if any(a[c]!=b[c] for c in cycle)]
        self.assertGreaterEqual(len(active),2)
        for cancel in (False,True):
            class Waiting:
                parameters=SimpleNamespace()
                num_branches=0
                def __init__(self):
                    self.entered=threading.Event();self.done=threading.Event();self.stops=0
                def solve(self,model):
                    self.entered.set()
                    if not self.done.wait(1): raise AssertionError('Repeated native cancellation did not arrive')
                    return cp_model.UNKNOWN
                def stop_search(self):
                    self.stops+=1
                    if self.stops>=2: self.done.set()
            fake=Waiting()
            with self.subTest(cancel=cancel),patch.object(cp_model,'CpSolver',return_value=fake), \
                 patch('binary_pairwise.maximize',return_value={'optimal':False}):
                result=cycle_recombine(p,a,b,seconds=2 if cancel else .08,
                                       should_stop=fake.entered.is_set if cancel else None)
            self.assertGreaterEqual(fake.stops,2)
            self.assertEqual(result['termination'],'stopped' if cancel else 'timeout')
            self.assertGreaterEqual(result['score'],max(score(p,a),score(p,b)))
            self.assertFalse(any(t.name=='eternity-cycle-cancel' for t in threading.enumerate()))

    def test_cycle_build_consumes_native_search_budget(self):
        from ortools.sat.python import cp_model
        p,answer=generated_problem(3,3,0,clue_count=1)
        a=scrambled(p,answer,12);b=scrambled(p,answer,23)
        original=cp_model.CpModel.add_allowed_assignments
        def slow_table(model,variables,rows):
            time.sleep(.03)
            return original(model,variables,rows)
        with patch.object(cp_model.CpModel,'add_allowed_assignments',slow_table), \
             patch('binary_pairwise.maximize',return_value={'optimal':False}), \
             patch.object(cp_model,'CpSolver',side_effect=AssertionError('No second native budget')):
            result=cycle_recombine(p,a,b,seconds=.01)
        self.assertEqual(result['termination'],'timeout')
        self.assertTrue(structurally_valid(p,result['board']))

    def test_portfolio_bounds_duplicate_parent_stream_and_keeps_validated_best(self):
        p,answer=generated_problem(3,4,4)
        reference=scrambled(p,answer,6)
        consumed=0
        def duplicates():
            nonlocal consumed
            for _ in range(65):
                consumed+=1
                if consumed>64: raise AssertionError('Duplicate parent stream is unbounded')
                yield reference
        improvements=[]
        result=solve(p,reference,seconds=.05,parents=duplicates(),on_improvement=improvements.append)
        self.assertLessEqual(consumed,64)
        self.assertTrue(structurally_valid(p,result['board']))
        self.assertGreaterEqual(result['score'],score(p,reference))
        self.assertTrue(all(structurally_valid(p,item['board']) and score(p,item['board'])==item['score']
                            for item in improvements))

    def test_portfolio_stop_does_not_consume_parents_or_start_native_work(self):
        p,answer=generated_problem(3,4,4)
        reference=scrambled(p,answer,6)
        def forbidden():
            raise AssertionError('Stopped portfolio inspected parents')
            yield reference
        for kwargs in ({'seconds':0},{'seconds':2,'should_stop':lambda:True}):
            result=solve(p,reference,parents=forbidden(),**kwargs)
            self.assertEqual(result['board'],reference)
            self.assertEqual(result['neighborhoods'],0)
            self.assertEqual(result['recombinations'],0)

    def test_gpu_portfolio_batches_neighborhoods_and_reports_actual_cost_backend(self):
        from gpu_assignment import AssignmentEvaluator
        p,answer=generated_problem(4,5,8,clue_count=1)
        reference=scrambled(p,answer,33)
        created=[]
        def cpu_fixture(problem,**kwargs):
            self.assertEqual(kwargs['backend'],'cuda');self.assertFalse(kwargs['allow_cpu_fallback'])
            kwargs['backend']='cpu';evaluator=AssignmentEvaluator(problem,**kwargs);created.append(evaluator)
            return evaluator
        progress=[]
        with patch('gpu_assignment.AssignmentEvaluator',side_effect=cpu_fixture):
            result=solve(p,reference,seconds=.1,gpu_backend='cuda',gpu_batch_size=8,on_progress=progress.append)
        self.assertTrue(structurally_valid(p,result['board']))
        self.assertGreaterEqual(result['score'],score(p,reference))
        self.assertEqual(result['cost_backend'],'cpu')
        self.assertGreater(result['cost_batches'],0)
        self.assertGreater(result['cost_pairs'],0)
        self.assertTrue(all(item['cost_backend']=='cpu' and 'GPU costs' not in item['backend'] for item in progress))

    def test_gpu_portfolio_unavailable_and_stopped_policies_are_explicit(self):
        from gpu_backends import BackendUnavailable
        p,answer=generated_problem(3,5,2)
        reference=scrambled(p,answer,53)
        with patch('gpu_backends.make_backend',side_effect=BackendUnavailable('unavailable fixture')) as runtime:
            result=solve(p,reference,seconds=0,gpu_backend='cuda')
            self.assertEqual(result['cost_backend'],'uninitialized');runtime.assert_not_called()
            with self.assertRaises(BackendUnavailable): solve(p,reference,seconds=1,gpu_backend='cuda')
            result=solve(p,reference,seconds=.1,gpu_backend='cuda',gpu_allow_cpu_fallback=True)
        self.assertEqual(result['cost_backend'],'cpu')
        self.assertIn('unavailable fixture',result['cost_fallback_reason'])
        self.assertTrue(structurally_valid(p,result['board']))

    def test_cycle_component_fast_path_matches_cp_sat_and_validates_its_objective(self):
        from ortools.sat.python import cp_model
        p,answer=generated_problem(3,3,0,clue_count=1)
        a=scrambled(p,answer,12);b=scrambled(p,answer,23)
        with patch.object(cp_model,'CpSolver',side_effect=AssertionError('Fast path should avoid CP-SAT')):
            fast=cycle_recombine(p,a,b,seconds=1)
        self.assertEqual(fast['cycle_backend'],'component-dp-and-enumeration')
        self.assertTrue(fast['optimal'])
        with patch('binary_pairwise.maximize',return_value={'optimal':False}):
            general=cycle_recombine(p,a,b,seconds=1)
        self.assertTrue(general['optimal']);self.assertEqual(general['score'],fast['score'])
        self.assertEqual(general['cycle_backend'],'cp-sat')
        cycles=[cycle for cycle in permutation_cycles(p,a,b) if any(a[c]!=b[c] for c in cycle)]
        with patch('binary_pairwise.maximize',return_value={'optimal':True,'assignment':[0]*len(cycles),'value':-100}):
            with self.assertRaises(RuntimeError): cycle_recombine(p,a,b,seconds=1)

    def test_bulk_assignment_arcs_fallback_only_for_missing_api(self):
        from ortools.graph.python import linear_sum_assignment
        from gpu_assignment import cpu_cost_matrix
        from smart_repair import _assignment_from_matrix
        p,answer=generated_problem(4,3,4)
        reference=scrambled(p,answer,32);matrix=cpu_cost_matrix(p,reference,independent_cells(p,parity=0))
        Native=linear_sum_assignment.SimpleLinearSumAssignment
        class ScalarOnly:
            add_arcs_with_cost=None
            def __init__(self): self.native=Native()
            def __getattr__(self,name): return getattr(self.native,name)
        with patch.object(linear_sum_assignment,'SimpleLinearSumAssignment',ScalarOnly):
            fallback=_assignment_from_matrix(p,matrix)
        scalar=_assignment_from_matrix(p,matrix,bulk_arcs=False)
        self.assertEqual(fallback['matching_arc_mode'],'scalar');self.assertEqual(fallback['board'],scalar['board'])
        class Broken(ScalarOnly):
            def add_arcs_with_cost(self,*_): raise RuntimeError('bulk failure fixture')
        with patch.object(linear_sum_assignment,'SimpleLinearSumAssignment',Broken):
            with self.assertRaisesRegex(RuntimeError,'bulk failure fixture'): _assignment_from_matrix(p,matrix)

    def test_bulk_assignment_arc_chunks_check_cancellation(self):
        from ortools.graph.python import linear_sum_assignment
        from gpu_assignment import cpu_cost_matrix
        from smart_repair import _assignment_from_matrix
        p,answer=generated_problem(16,3,3)
        matrix=cpu_cost_matrix(p,answer,independent_cells(p,parity=0))
        Native=linear_sum_assignment.SimpleLinearSumAssignment;calls=[]
        class CancelAfterChunk:
            def __init__(self): self.native=Native()
            def __getattr__(self,name): return getattr(self.native,name)
            def add_arcs_with_cost(self,left,right,costs):
                calls.append(len(left));return self.native.add_arcs_with_cost(left,right,costs)
            def solve(self): raise AssertionError('Cancellation should precede native matching')
        with patch.object(linear_sum_assignment,'SimpleLinearSumAssignment',CancelAfterChunk):
            result=_assignment_from_matrix(p,matrix,should_stop=lambda:bool(calls))
        self.assertEqual(calls,[4096]);self.assertEqual(result['termination'],'stopped')
        self.assertEqual(result['board'],answer)

if __name__=='__main__': unittest.main()

