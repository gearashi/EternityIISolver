"""Ordering-only changes checked against an independent exhaustive oracle."""
from dataclasses import replace
import itertools,random,unittest
from unittest.mock import patch
from exact_model import generated_problem,grid_edges,check_solution
from refined_diagrams import solve


def independent_oracle(p):
    # Enumerate the supplied states, then check frame, identity and every edge
    # directly. Do not call initial_domains or the solver's witness validator.
    for board in itertools.product(*p.domains):
        if len({s//4 for s in board})!=len(board):continue
        if any(board[c]!=s for c,s in p.fixed.items()):continue
        valid=True
        for cell,state in enumerate(board):
            border=(cell<p.size,cell%p.size==p.size-1,cell>=p.size*(p.size-1),cell%p.size==0)
            if any((p.faces[state][side]==0)!=border[side] for side in range(4)):
                valid=False;break
        if valid and all(p.faces[board[a]][sa]==p.faces[board[b]][sb] for a,b,sa,sb in (grid_edges(p.size) if p.edges is None else p.edges)):
            return list(board)
    return None


class RefinedOrderingTests(unittest.TestCase):
    def test_physical_rotations_against_independent_oracle(self):
        for seed in range(32):
            p,_=generated_problem(2,1+seed%4,seed,clue_count=seed%2)
            rng=random.Random(17000+seed)
            p=replace(p,domains=tuple(tuple(s for s in d if rng.random()<.65) for d in p.domains),
                      edges=tuple(e for i,e in enumerate(grid_edges(2)) if seed%3!=0 or i%2==0))
            expected=independent_oracle(p)
            for abstraction in ('exact','refine'):
                for ordering in ('geometric','domain'):
                    r=solve(p,seconds=3,seed=seed,abstraction=abstraction,ordering=ordering)
                    self.assertEqual(r['outcome'],'solved' if expected is not None else 'infeasible',(seed,abstraction,ordering,r))
                    if r['board'] is not None:self.assertTrue(check_solution(p,r['board']))

    def test_frame_clues_and_nongeometric_edge_scopes(self):
        p,known=generated_problem(3,4,811,clue_count=3)
        for edges in (None,(),tuple(e for e in grid_edges(3) if e[0] not in (1,4,7))):
            q=replace(p,edges=edges)
            r=solve(q,seconds=5,ordering='domain')
            self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(q,r['board']))
            self.assertEqual(sorted(r['cell_order']),list(range(9)))
            self.assertTrue(all(r['board'][c]==s for c,s in p.fixed.items()))

    def test_last_support_mask_preserves_the_greedy_cell_order(self):
        from refined_diagrams import _domain_order
        from exact_model import constraint_edges
        # Pinned orders from the previous per-candidate scan, including tied
        # directions, fixed clues, and cells with shared physical pools.
        for size,seed,clues,expected in (
          (3,189,2,(6,8,7,0,3,4,1,5,2)),
          (4,217,2,(0,4,8,12,13,9,5,1,2,6,10,14,15,11,7,3))):
            problem,_=generated_problem(size,3,seed,clue_count=clues)
            order,_=_domain_order(problem,problem.domains,constraint_edges(problem),
                                  'refine',lambda amount=1:None)
            self.assertEqual(order,expected)

    def test_incomplete_ordering_never_proves_infeasibility(self):
        p,_=generated_problem(3,3,87)
        r=solve(p,ordering='domain',max_work=1000)
        self.assertEqual(r['outcome'],'timeout');self.assertFalse(r['coverage_verified'])
        self.assertEqual(r['reasoning_counters_session']['candidate_placements'],0)
        self.assertEqual(r['reasoning_counters_session']['diagrams'],0)
        r=solve(p,ordering='domain',should_stop=lambda:True)
        self.assertEqual(r['outcome'],'stopped');self.assertIsNone(r['board'])
        with self.assertRaises(ValueError):solve(p,ordering='unknown')

    def test_each_cell_value_shuffle_is_identical_between_modes(self):
        p,_=generated_problem(3,3,912,clue_count=2)
        original=random.Random.shuffle;records=[]
        for mode in ('geometric','domain'):
            calls=[]
            def record(rng,states):
                original(rng,states);calls.append(tuple(states))
            with patch('refined_diagrams.random.Random.shuffle',new=record):
                solve(p,ordering=mode,seed=23)
            records.append(calls)
        self.assertEqual(len(records[0]),9);self.assertEqual(records[0],records[1])

    def test_order_selection_is_counted_once_and_deterministic(self):
        p,_=generated_problem(3,3,189,clue_count=2)
        a=solve(p,ordering='domain',seed=13);b=solve(p,ordering='domain',seed=13)
        self.assertEqual(a['cell_order'],b['cell_order'])
        self.assertEqual(a['reasoning_counters_session'],b['reasoning_counters_session'])
        c=a['reasoning_counters_session']
        self.assertGreater(c['ordering_work'],0);self.assertLessEqual(c['ordering_work'],c['setup_work'])
        self.assertEqual(a['work'],sum(c[k] for k in ('candidate_placements','edge_checks','state_operations','witness_checks','setup_work')))
        self.assertGreaterEqual(c['ordering_candidates'],2)
        self.assertEqual(solve(p)['ordering'],'geometric')

if __name__=='__main__':unittest.main()
