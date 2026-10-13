"""CPU oracle and bounded scheduling checks for GPU assignment matrices."""
from dataclasses import replace
import itertools
import unittest
from unittest.mock import patch
import numpy as np
from exact_model import generated_problem, frame_ok, grid_edges
from gpu_assignment import AssignmentEvaluator, AssignmentCancelled, AssignmentCallbackError, cpu_cost_matrix
from gpu_backends import BackendUnavailable
from smart_repair import assignment_repair_batch, independent_cells, score, structurally_valid
from test_smart_repair import scrambled


def independent_oracle(problem, board, cells):
    weights=np.full((len(cells),len(cells)),-1,dtype=np.int32)
    states=weights.copy()
    for i,cell in enumerate(cells):
        for j,source in enumerate(cells):
            values=[]
            for state in problem.domains[cell]:
                if state//4!=board[source]//4 or not frame_ok(problem.size,cell,problem.faces[state]): continue
                candidate=list(board);candidate[cell]=state
                points=sum(problem.faces[candidate[a]][sa]==problem.faces[candidate[b]][sb]
                           for a,b,sa,sb in grid_edges(problem.size) if cell in (a,b))
                values.append((points,state==board[cell],-state,state))
            if values:
                best=max(values);weights[i,j]=best[0];states[i,j]=best[3]
    return weights,states


class GpuAssignmentTests(unittest.TestCase):
    def test_cpu_matrices_match_separate_oracle_with_clues_frame_and_restricted_domains(self):
        for seed in range(4):
            p,answer=generated_problem(5,4,seed,clue_count=2)
            board=scrambled(p,answer,seed+15)
            # Include illegal frame states and remove legal alternatives. The
            # incumbent always remains feasible; devices must respect both.
            domains=tuple(tuple(sorted({board[c]}|{s for s in d if s%3!=0}|{0})) for c,d in enumerate(p.domains))
            p=replace(p,domains=domains)
            cells=independent_cells(p,seed)
            got=cpu_cost_matrix(p,board,cells);want=independent_oracle(p,board,cells)
            np.testing.assert_array_equal(got['weights'],want[0]);np.testing.assert_array_equal(got['states'],want[1])
            self.assertTrue(np.any(got['weights']==-1))

    def test_batch_partitions_obey_memory_pair_and_count_limits(self):
        p,_=generated_problem(4,3,1)
        evaluator=AssignmentEvaluator(p,backend='cpu',batch_size=3,max_pairs=32)
        slices=evaluator.plan_batches([2,4,4,1,3,0,0,2])
        self.assertEqual(slices,[(0,2),(2,4),(4,7),(7,8)])
        covered=[]
        for a,b in slices:
            widths=[2,4,4,1,3,0,0,2][a:b];covered.extend(range(a,b))
            self.assertLessEqual((b-a)*max(widths)**2,32)
            self.assertLessEqual(b-a,3)
        self.assertEqual(covered,list(range(8)))
        limit=evaluator.estimated_bytes(1,4)
        memory=AssignmentEvaluator(p,backend='cpu',max_batch_bytes=limit)
        self.assertEqual(memory.plan_batches([4,4]),[(0,1),(1,2)])
        with self.assertRaises(ValueError): evaluator.plan_batches([6])
        with self.assertRaises(ValueError): AssignmentEvaluator(p,backend='cpu',max_batch_bytes=1).plan_batches([1])

    def test_batch_matching_agrees_with_independent_exhaustive_neighborhood(self):
        p,answer=generated_problem(3,4,8,clue_count=1);board=scrambled(p,answer,42)
        jobs=[(board,independent_cells(p,seed)[:4]) for seed in range(4)]
        evaluator=AssignmentEvaluator(p,backend='cpu',batch_size=2)
        got=assignment_repair_batch(p,jobs,evaluator=evaluator)
        self.assertEqual(evaluator.batches,2)
        for (_,cells),item in zip(jobs,got):
            best=score(p,board)
            for pieces in itertools.permutations([board[c]//4 for c in cells]):
                for states in itertools.product(*[[s for s in p.domains[c] if s//4==piece] for c,piece in zip(cells,pieces)]):
                    candidate=list(board)
                    for c,s in zip(cells,states): candidate[c]=s
                    if structurally_valid(p,candidate): best=max(best,score(p,candidate))
            self.assertEqual(item['score'],best);self.assertTrue(item['optimal'])
            self.assertTrue(structurally_valid(p,item['board']))

    def test_unavailable_gpu_requires_explicit_fallback_and_cpu_is_labeled(self):
        p,board=generated_problem(3,3,1)
        jobs=[(board,independent_cells(p,2))]
        with patch('gpu_backends.make_backend',side_effect=BackendUnavailable('fixture unavailable')):
            with self.assertRaises(BackendUnavailable): AssignmentEvaluator(p,backend='cuda').evaluate(jobs)
            fallback=AssignmentEvaluator(p,backend='cuda',allow_cpu_fallback=True)
            result=fallback.evaluate(jobs)
        self.assertEqual(result[0]['backend'],'cpu');self.assertEqual(fallback.backend,'cpu')
        self.assertIn('fixture unavailable',fallback.fallback_reason)

    def test_cancellation_does_not_initialize_device_or_report_completed_batch(self):
        p,board=generated_problem(4,3,1)
        evaluator=AssignmentEvaluator(p,should_stop=lambda:True)
        with patch('gpu_backends.make_backend',side_effect=AssertionError('No GPU after cancellation')):
            with self.assertRaises(AssignmentCancelled): evaluator.evaluate([(board,independent_cells(p))])
        self.assertEqual(evaluator.batches,0);self.assertEqual(evaluator.backend,'uninitialized')

    def test_partial_edges_adjacent_cells_and_clues_are_rejected(self):
        p,board=generated_problem(3,3,1,clue_count=1)
        with self.assertRaises(ValueError): AssignmentEvaluator(replace(p,edges=()))
        for cells in ([0,1],[0,0],[True],[next(iter(p.fixed))]):
            with self.assertRaises(ValueError): AssignmentEvaluator(p,backend='cpu').evaluate([(board,cells)])

    def test_input_snapshot_empty_batch_and_bounded_iterable(self):
        p,board=generated_problem(3,3,1)
        evaluator=AssignmentEvaluator(p,backend='cpu')
        p.fixed[0]=board[1]
        result=evaluator.evaluate([(board,[0])])
        self.assertEqual(result[0]['states'][0,0],board[0])
        self.assertEqual(evaluator.evaluate([]),[])
        self.assertEqual(evaluator.evaluate([(board,[])])[0]['weights'].shape,(0,0))
        with self.assertRaises(ValueError): evaluator.evaluate(itertools.repeat((board,[])))

    def test_selected_gpu_cost_corruption_is_rejected_independently(self):
        p,board=generated_problem(3,3,1)
        evaluator=AssignmentEvaluator(p,backend='cpu')
        matrices=evaluator.evaluate([(board,[0])]);matrices[0]['weights'][0,0]+=1
        with patch.object(evaluator,'evaluate',return_value=matrices):
            with self.assertRaises(RuntimeError): assignment_repair_batch(p,[(board,[0])],evaluator=evaluator)

    def test_invalid_resource_controls_and_wrong_problem_are_rejected(self):
        p,board=generated_problem(3,3,1)
        for kwargs in ({'batch_size':True},{'batch_size':65},{'max_pairs':0},{'max_batch_bytes':2**30},
                       {'backend':'fake'},{'allow_cpu_fallback':1},{'should_stop':3}):
            with self.assertRaises(ValueError): AssignmentEvaluator(p,**kwargs)
        other,_=generated_problem(3,3,2)
        with self.assertRaises(ValueError): assignment_repair_batch(p,[(board,[0])],evaluator=AssignmentEvaluator(other,backend='cpu'))


    def test_callback_errors_and_execution_failures_do_not_become_cpu_fallback(self):
        p,board=generated_problem(3,3,1)
        def broken(): raise ValueError('callback fixture')
        evaluator=AssignmentEvaluator(p,allow_cpu_fallback=True,should_stop=broken)
        with self.assertRaises(AssignmentCallbackError): evaluator.evaluate([(board,[0])])
        self.assertEqual(evaluator.backend,'uninitialized')
        evaluator=AssignmentEvaluator(p,allow_cpu_fallback=True)
        evaluator.backend='cuda';evaluator.device='fixture'
        with patch.object(evaluator,'_device_batch',side_effect=RuntimeError('execution fixture')):
            with self.assertRaisesRegex(RuntimeError,'execution fixture'): evaluator.evaluate([(board,[0])])
        self.assertEqual(evaluator.backend,'cuda');self.assertEqual(evaluator.batches,0)

    def test_typed_compatibility_normalizes_containers_and_edges_without_numeric_aliases(self):
        from exact_model import ExactProblem, problem_hash
        p,board=generated_problem(3,4,22,clue_count=2)
        evaluator=AssignmentEvaluator(p,backend='cpu')
        clone=ExactProblem(p.size,[list(row) for row in p.faces],dict(reversed(list(p.fixed.items()))),
                           [list(row) for row in p.domains],list(grid_edges(p.size)))
        self.assertEqual(problem_hash(p),problem_hash(clone));self.assertTrue(evaluator.matches_problem(clone))
        for field in ('faces','domains','edges'):
            for invalid in (True,1.0):
                fresh=ExactProblem(p.size,[list(row) for row in p.faces],dict(p.fixed),
                                   [list(row) for row in p.domains],list(grid_edges(p.size)))
                rows=getattr(fresh,field);original=rows[0][0]
                value=bool(original) if invalid is True else float(original)
                if field=='edges': rows[0]=(value,)+rows[0][1:]
                else: rows[0][0]=value
                self.assertFalse(evaluator.matches_problem(fresh),field)
        fixed=next(iter(p.fixed))
        bad=replace(p,fixed=dict(p.fixed));bad.fixed[fixed]=float(bad.fixed[fixed])
        self.assertFalse(evaluator.matches_problem(bad))
        reversed_edges=replace(p,edges=tuple(reversed(grid_edges(p.size))))
        self.assertNotEqual(problem_hash(p),problem_hash(reversed_edges))
        self.assertFalse(evaluator.matches_problem(reversed_edges))
        reversed_domain=replace(p,domains=tuple(tuple(reversed(d)) for d in p.domains))
        self.assertFalse(evaluator.matches_problem(reversed_domain))

    def test_evaluator_model_and_structural_view_are_read_only(self):
        p,_=generated_problem(3,4,22,clue_count=1)
        evaluator=AssignmentEvaluator(p,backend='cpu')
        with self.assertRaises(AttributeError): evaluator.problem=p
        with self.assertRaises(AttributeError): evaluator.structural_problem=p
        for model in (evaluator.problem,evaluator.structural_problem):
            with self.assertRaises(TypeError): model.fixed[0]=0
            with self.assertRaises(TypeError): model.domains[0][0]=0
            with self.assertRaises(TypeError): model.faces[0][0]=0
        self.assertTrue(evaluator.matches_problem(p))

    def test_batch_detects_source_mutation_in_matrix_and_matching_callbacks(self):
        from exact_model import ExactProblem
        for phase in ('matrix','matching'):
            source,board=generated_problem(3,4,33)
            p=ExactProblem(source.size,[list(f) for f in source.faces],dict(source.fixed),
                           [list(d) for d in source.domains])
            def mutate():
                p.fixed[0]=board[0]
                return False
            evaluator=AssignmentEvaluator(p,backend='cpu',should_stop=mutate if phase=='matrix' else None)
            with self.assertRaisesRegex(ValueError,'changed during'):
                assignment_repair_batch(p,[(board,independent_cells(p))],evaluator=evaluator,
                                        should_stop=mutate if phase=='matching' else None)
            self.assertEqual(evaluator.problem.fixed,{})

    def test_reused_validation_matches_scalar_reference_for_every_board(self):
        p,answer=generated_problem(5,4,74,clue_count=2)
        board=scrambled(p,answer,33)
        jobs=[(board,independent_cells(p,seed)) for seed in range(4)]
        evaluator=AssignmentEvaluator(p,backend='cpu')
        cached=assignment_repair_batch(p,jobs,evaluator=evaluator,reuse_validation=True)
        reference=assignment_repair_batch(p,jobs,evaluator=evaluator,reuse_validation=False)
        self.assertEqual([r['board'] for r in cached],[r['board'] for r in reference])
        self.assertEqual([r['score'] for r in cached],[r['score'] for r in reference])
        for reuse in (True,False):
            bad=list(board);bad[0]=bad[1]
            with self.assertRaises(ValueError): assignment_repair_batch(p,[(bad,[])],evaluator=evaluator,reuse_validation=reuse)
        for bad in (0,1,None):
            with self.assertRaises(ValueError): assignment_repair_batch(p,jobs,evaluator=evaluator,reuse_validation=bad)

class AssignmentDiagnosticTests(unittest.TestCase):
    def test_production_refuses_cpu_opencl_and_diagnostic_opt_in_is_explicit(self):
        from types import SimpleNamespace
        from gpu_assignment import diagnose
        p,_=generated_problem(3,4,4)
        runtime=SimpleNamespace(name='opencl',diagnostic_cpu=True)
        evaluator=AssignmentEvaluator(p,backend='opencl')
        with patch('gpu_backends.make_backend',return_value=runtime) as factory:
            with self.assertRaisesRegex(BackendUnavailable,'CPU diagnostic device'):
                evaluator._initialize()
        factory.assert_called_once_with('opencl',cache_dir=None,allow_opencl_cpu=False)
        self.assertEqual(evaluator.backend,'uninitialized')
        for backend in ('auto','cuda','cpu'):
            with self.assertRaises(ValueError):
                AssignmentEvaluator(p,backend=backend,_diagnostic_opencl_cpu=True)
        with self.assertRaises(ValueError):
            AssignmentEvaluator(p,backend='opencl',_diagnostic_opencl_cpu=True,allow_cpu_fallback=True)
        with self.assertRaises(ValueError):
            AssignmentEvaluator(p,backend='opencl',_diagnostic_opencl_cpu=1)
        for backend in ('auto','cuda'):
            with patch('gpu_backends.make_backend',side_effect=AssertionError('No initialization')):
                with self.assertRaises(ValueError): diagnose(backend=backend,allow_opencl_cpu=True)
        with self.assertRaises(ValueError): diagnose(backend='opencl',allow_opencl_cpu=1)

    def test_diagnostic_cpu_opencl_is_compiled_verified_and_labeled(self):
        from types import SimpleNamespace
        from gpu_assignment import diagnose
        compiled=[]
        class Program:
            def __init__(self,context,code): compiled.append(code)
            def build(self,**kwargs): return self
        runtime=SimpleNamespace(name='opencl',diagnostic_cpu=True,device='PoCL fixture',
                                context=object(),queue=object(),xp=np,
                                cl=SimpleNamespace(Program=Program,Kernel=lambda *args:object()))
        def device_reference(evaluator,jobs):
            # Runtime plumbing only: existing independent tests and CI exercise
            # the actual kernel. This bounded fixture needs no local CPU ICD.
            return [dict(cpu_cost_matrix(evaluator.problem,board,cells),
                         backend='opencl',device=runtime.device) for board,cells in jobs]
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            with patch('gpu_backends.make_backend',return_value=runtime) as factory, \
                    patch.object(AssignmentEvaluator,'_device_batch',device_reference):
                result=diagnose(backend='opencl',allow_opencl_cpu=True,size=3,count=2,
                                seconds=10,cache_dir=directory)
        factory.assert_called_once_with('opencl',cache_dir=directory,allow_opencl_cpu=True)
        self.assertEqual(len(compiled),1);self.assertIn('assignment_costs',compiled[0])
        self.assertTrue(result['passed']);self.assertTrue(result['diagnostic_cpu'])
        self.assertTrue(result['cost_diagnostic_cpu'])
        self.assertEqual(result['cost_backend'],'opencl')
        self.assertIn('not GPU certification',result['validation_scope'])
        self.assertTrue(result['all_matrices_equal']);self.assertTrue(result['all_repairs_valid'])

    def test_cli_saves_success_and_failure_reports_and_forwards_opt_in(self):
        import io
        import json
        from pathlib import Path
        import tempfile
        from gpu_assignment import main
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'nested'/'assignment.json'
            argv=['--backend','opencl','--allow-opencl-cpu','--size','3','--count','2',
                  '--seconds','5','--output',str(output)]
            with patch('gpu_assignment.diagnose',return_value={'passed':True,'diagnostic_cpu':True}) as diagnostic, \
                    patch('sys.stdout',new=io.StringIO()) as stdout:
                self.assertEqual(main(argv),0)
            self.assertTrue(diagnostic.call_args.kwargs['allow_opencl_cpu'])
            self.assertEqual(json.loads(output.read_text()),json.loads(stdout.getvalue()))
            with patch('gpu_assignment.diagnose',side_effect=RuntimeError('ICD fixture unavailable')), \
                    patch('sys.stdout',new=io.StringIO()) as stdout:
                self.assertEqual(main(argv),1)
            failed=json.loads(output.read_text());self.assertFalse(failed['passed'])
            self.assertIn('ICD fixture unavailable',failed['error'])
            self.assertEqual(failed,json.loads(stdout.getvalue()))

if __name__=='__main__': unittest.main()
