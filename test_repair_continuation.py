"""Finite continuation checks for the actual structured repair state machine.

Native time-limited regional searches are disabled in deterministic comparisons.
Fake batches are used only to place cancellation precisely at batch boundaries;
all returned boards still undergo the production legality and scoring checks.
"""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

from exact_model import check_solution, generated_problem
from repair_checkpoint import CheckpointCodec
from smart_repair import checkpoint_settings, score, solve
from test_gpu_assignment import scrambled


def repair_result(problem, board, *, optimal=True):
    return dict(board=list(board),score=score(problem,board),reference_score=score(problem,board),
                optimal=optimal,termination='optimal' if optimal else 'stopped')


def decision_state(checkpoint):
    state=deepcopy(checkpoint['state'])
    state.pop('elapsed_total')
    return state


class RepairContinuationTests(unittest.TestCase):
    def fixture(self):
        p,answer=generated_problem(5,7,273,clue_count=2)
        return p,answer,scrambled(p,answer,192)

    def test_real_cpu_matching_split_run_preserves_complete_decision_state(self):
        p,answer,board=self.fixture()
        kwargs=dict(seconds=10,seed=17,diversify=False,region_repair_enabled=False,
                    pair_selection=False,on_checkpoint=lambda state:None)
        full=solve(p,board,max_batches=8,**kwargs)
        first=solve(p,board,max_batches=3,**kwargs)
        resumed=solve(p,board,checkpoint=first['checkpoint'],max_batches=5,**kwargs)
        self.assertTrue(resumed['resumed'])
        self.assertEqual(decision_state(full['checkpoint']),decision_state(resumed['checkpoint']))
        self.assertEqual(full['score'],resumed['score'])
        self.assertEqual(full['board'],resumed['board'])
        self.assertEqual(full['neighborhoods'],resumed['neighborhoods'])
        self.assertGreaterEqual(resumed['score'],first['score'])
        self.assertTrue(check_solution(replace(p,edges=()),resumed['board']))
        self.assertEqual(p.fixed,{c:answer[c] for c in p.fixed})

    def test_partial_gpu_batch_replays_requests_without_marking_unfinished_work(self):
        p,_,board=self.fixture();stop=[False];calls=[]
        def batch(problem,jobs,**kwargs):
            jobs=list(jobs)
            calls.append((deepcopy(jobs),kwargs['tie_seed']))
            if len(calls)==1:
                stop[0]=True
                return [repair_result(problem,b,optimal=i==0) for i,(b,_) in enumerate(jobs)]
            return [repair_result(problem,b) for b,_ in jobs]
        kwargs=dict(seconds=10,seed=17,gpu_backend='cuda',gpu_batch_size=4,
                    diversify=False,region_repair_enabled=False,pair_selection=False,
                    should_stop=lambda:stop[0],on_checkpoint=lambda state:None)
        with patch('smart_repair.assignment_repair_batch',side_effect=batch), \
                patch('gpu_backends.make_backend',side_effect=AssertionError('No GPU in state-machine test')):
            first=solve(p,board,max_batches=1,**kwargs)
            state=first['checkpoint']['state']
            self.assertEqual(first['neighborhoods'],1)
            self.assertEqual(state['completed'],0)
            self.assertEqual(state['rounds'],0)
            self.assertEqual(state['batches'],0)
            self.assertEqual(state['seen'],[])
            stop[0]=False
            resumed=solve(p,board,checkpoint=first['checkpoint'],max_batches=1,**kwargs)
        self.assertEqual(calls[0],calls[1])
        self.assertEqual(resumed['checkpoint']['state']['completed'],4)
        self.assertEqual(len(resumed['checkpoint']['state']['seen']),4)
        self.assertEqual(resumed['checkpoint']['state']['batches'],1)

    def test_improvement_from_partial_batch_survives_boundary_rollback(self):
        p,answer,board=self.fixture();stop=[False]
        def batch(problem,jobs,**kwargs):
            jobs=list(jobs);stop[0]=True
            return [repair_result(problem,answer,optimal=True)] + [
                repair_result(problem,b,optimal=False) for b,_ in jobs[1:]]
        kwargs=dict(seconds=10,seed=21,gpu_backend='cuda',gpu_batch_size=4,
                    diversify=False,region_repair_enabled=False,pair_selection=False,
                    should_stop=lambda:stop[0],on_checkpoint=lambda state:None)
        with patch('smart_repair.assignment_repair_batch',side_effect=batch), \
                patch('gpu_backends.make_backend',side_effect=AssertionError('No GPU in state-machine test')):
            result=solve(p,board,max_batches=1,**kwargs)
        state=result['checkpoint']['state']
        self.assertEqual(result['board'],answer)
        self.assertEqual(state['best'],answer)
        self.assertEqual(state['best_score'],40)
        self.assertEqual(state['bank'],[board])
        self.assertEqual(state['seen'],[])
        self.assertEqual(state['completed'],0)
        self.assertTrue(check_solution(p,state['best']))

    def test_callback_payload_mutations_cannot_change_saved_or_working_state(self):
        p,answer,board=self.fixture();notifications=[]
        def improved(payload):
            notifications.append('improved')
            payload['board'][0]=-1
            payload['score']=-1
        def checkpointed(payload):
            notifications.append('checkpoint')
            payload['state']['best'][0]=-1
            payload['state']['bank'][0][0]=-1
            payload['state']['counts']['region_calls']=999
            payload['state']['seen'].append('not-a-digest')
        def matching(problem,working,cells,**kwargs):
            return repair_result(problem,answer)
        with patch('smart_repair.assignment_repair',side_effect=matching):
            result=solve(p,board,seconds=10,seed=7,max_batches=1,region_repair_enabled=False,
                         pair_selection=False,on_improvement=improved,on_checkpoint=checkpointed,
                         checkpoint_interval=1e-8)
        self.assertIn('improved',notifications);self.assertIn('checkpoint',notifications)
        self.assertEqual(result['board'],answer)
        codec=CheckpointCodec(p,checkpoint_settings(seed=7,region_repair_enabled=False,pair_selection=False))
        restored=codec.decode(result['checkpoint'])
        self.assertEqual(restored['best'],answer)
        self.assertEqual(restored['counts']['region_calls'],0)
        self.assertTrue(all(check_solution(replace(p,edges=()),b) for b in restored['bank']))

    def test_changed_continuation_settings_reject_before_search(self):
        p,_,board=self.fixture()
        kwargs=dict(seconds=10,seed=17,region_repair_enabled=False,pair_selection=False,
                    neutral_ties=True,on_checkpoint=lambda state:None,max_batches=1)
        result=solve(p,board,**kwargs)
        for changed in ({'seed':18},{'neutral_ties':False},{'gpu_batch_size':32}):
            with self.subTest(changed=changed), \
                    patch('smart_repair.assignment_repair',side_effect=AssertionError('Do not search incompatible state')):
                with self.assertRaisesRegex(ValueError,'settings mismatch'):
                    solve(p,board,checkpoint=result['checkpoint'],**dict(kwargs,**changed))
        # Invocation limits are intentionally not continuation settings.
        resumed=solve(p,board,checkpoint=result['checkpoint'],**dict(kwargs,seconds=9,max_batches=2))
        self.assertTrue(resumed['resumed'])

    def test_capacity64_parents_keeps_pair_pool_bounded_and_target_mapping_legal(self):
        p,answer,reference=self.fixture()
        parents=[];seen={tuple(reference)}
        for seed in range(1000):
            parent=scrambled(p,answer,seed)
            if tuple(parent) not in seen:
                parents.append(parent);seen.add(tuple(parent))
            if len(parents)==63:break
        self.assertEqual(len(parents),63)
        calls=[]
        def matching(problem,board,cells,**kwargs):return repair_result(problem,board)
        def crossover(problem,left,right,**kwargs):
            calls.append((list(left),list(right)))
            return repair_result(problem,left if score(problem,left)>=score(problem,right) else right)
        with patch('smart_repair.assignment_repair',side_effect=matching), \
                patch('smart_repair.cycle_recombine',side_effect=crossover):
            result=solve(p,reference,parents=parents,seconds=10,max_batches=4,seed=73,
                         region_repair_enabled=False,diversify=False,pair_selection=True,
                         on_checkpoint=lambda state:None)
        self.assertEqual(result['parent_count'],64)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['recombinations'],1)
        self.assertEqual(len(result['checkpoint']['state']['bank']),64)
        structure=replace(p,edges=())
        self.assertTrue(all(check_solution(structure,b) for b in result['checkpoint']['state']['bank']))


if __name__=='__main__':unittest.main()
