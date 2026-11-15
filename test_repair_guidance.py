"""Independent finite legality/objective oracles for quality-guided repair."""
import itertools
import random
import unittest
from copy import deepcopy
from dataclasses import replace
from exact_model import generated_problem,make_problem,check_solution,grid_edges,frame_ok
from repair_guidance import Guidance,EdgePenalties,weighted_independent_set
from test_smart_repair import scrambled


def actual_score(problem,board):
    return sum(problem.faces[board[a]][sa]==problem.faces[board[b]][sb] for a,b,sa,sb in grid_edges(problem.size))


class IndependentTests(unittest.TestCase):
    def test_weighted_cut_equals_exhaustive_3x3_oracle(self):
        for seed in range(16):
            rng=random.Random(seed);weights=[rng.randrange(8) for _ in range(9)]
            excluded={seed%9};edges=grid_edges(3)
            expected=max(sum(weights[c] for c in range(9) if mask>>c&1)
                for mask in range(512)
                if not any(mask>>c&1 for c in excluded)
                and not any(mask>>a&1 and mask>>b&1 for a,b,_,_ in edges))
            chosen=weighted_independent_set(3,weights,excluded=excluded,seed=seed)
            self.assertEqual(sum(weights[c] for c in chosen),expected)
            self.assertEqual(chosen,weighted_independent_set(3,weights,excluded=excluded,seed=seed))
            self.assertFalse(set(chosen)&excluded)
            self.assertFalse(any(a in chosen and b in chosen for a,b,_,_ in edges))

    def test_zero_weights_and_cancel(self):
        chosen=weighted_independent_set(4,[0]*16,seed=1)
        self.assertFalse(any(a in chosen and b in chosen for a,b,_,_ in grid_edges(4)))
        self.assertEqual(weighted_independent_set(4,[1]*16,should_stop=lambda:True),[])
        calls=[]
        def stop():calls.append(1);return len(calls)>3
        self.assertEqual(weighted_independent_set(4,[1]*16,should_stop=stop),[])

    def test_bad_inputs(self):
        for kwargs in ({'seed':True},{'seed':-1},{'excluded':[16]},{'should_stop':2}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):weighted_independent_set(4,[1]*16,**kwargs)
        for weights in ([1]*15,[1]*17,[True]*16,[-1]*16,[10_000_001]*16):
            with self.assertRaises(ValueError):weighted_independent_set(4,weights)

    def test_weighted_cells_preserve_clues_and_maximize_frustration(self):
        p,answer=generated_problem(3,5,31,clue_count=2)
        board=scrambled(p,answer,77);g=Guidance(p);weights=[1]*9
        for a,b,sa,sb in grid_edges(3):
            if p.faces[board[a]][sa]!=p.faces[board[b]][sb]:weights[a]+=1;weights[b]+=1
        chosen=g.weighted_independent_cells(board,seed=17)
        oracle=weighted_independent_set(3,weights,excluded=p.fixed,seed=17)
        self.assertEqual(chosen,oracle)
        with self.assertRaises(ValueError):g.weighted_independent_cells([0]*9)


class PenaltyTests(unittest.TestCase):
    def fixture(self):
        p,answer=generated_problem(4,6,52,clue_count=2)
        return p,answer,scrambled(p,answer,91)

    def test_integer_decay_and_true_score_separation(self):
        p,answer,board=self.fixture();penalty=EdgePenalties(p,decay_every=3,max_penalty=2)
        original=list(board)
        for _ in range(5):penalty.observe(board)
        self.assertTrue(all(0<=v<=2 for v in penalty.values));self.assertTrue(any(penalty.values))
        value=penalty.evaluate(board)
        self.assertEqual(value['score'],actual_score(p,board))
        self.assertEqual(value['guided_score'],sum(w for w,(a,b,sa,sb) in zip(penalty.weights(),grid_edges(p.size)) if p.faces[board[a]][sa]==p.faces[board[b]][sb]))
        before=penalty.values.copy();penalty.observe(answer)
        self.assertEqual(penalty.values,[v//2 for v in before])
        self.assertEqual(board,original)

    def test_checkpoint_rejects_mismatch_corruption_and_detaches(self):
        p,_,board=self.fixture();first=EdgePenalties(p);first.observe(board);state=first.export_state()
        second=EdgePenalties(p);second.restore_state(state)
        self.assertEqual(first.export_state(),second.export_state())
        state['values'][0]=999
        self.assertNotEqual(state['values'],second.values)
        for change in ({'observations':True},{'observations':-1},{'values':[0]},{'problem':'other'},
                       {'decay_every':4},{'extra':0},{'values':[True]*len(first.values)}):
            damaged=first.export_state();damaged.update(change)
            before=second.export_state()
            with self.subTest(change=change),self.assertRaises(ValueError):second.restore_state(damaged)
            self.assertEqual(before,second.export_state())
        third=EdgePenalties(p,max_penalty=3)
        with self.assertRaises(ValueError):third.restore_state(first.export_state())

    def test_bad_observation_does_not_mutate_state(self):
        p,_,_=self.fixture();penalty=EdgePenalties(p);state=penalty.export_state()
        with self.assertRaises(ValueError):penalty.observe([0]*16)
        self.assertEqual(state,penalty.export_state())


class EjectionTests(unittest.TestCase):
    def fixture(self,restrict=False):
        p,answer=generated_problem(5,7,273)
        cells=[6,7,12]
        domains=list(p.domains)
        if restrict:
            for c in cells:domains[c]=tuple(answer[d] for d in cells)
        p=make_problem(p.size,p.faces,fixed={c:s for c,s in enumerate(answer) if c not in cells},domains=domains)
        board=list(answer)
        for index,c in enumerate(cells):board[c]=answer[cells[(index+1)%3]]
        return p,answer,board,cells

    def test_three_piece_cycle_restores_constructed_solution(self):
        p,answer,board,cells=self.fixture(True);g=Guidance(p);original=list(board)
        result=g.ejection_repair(board,seed=31,max_cycle=3,max_evaluations=100)
        self.assertEqual(result['board'],answer);self.assertTrue(check_solution(p,result['board']))
        self.assertEqual(result['relocated_pieces'],3);self.assertGreater(result['delta'],0)
        self.assertEqual(result['delta'],actual_score(p,answer)-actual_score(p,board))
        self.assertEqual(board,original);self.assertFalse(result['optimal'])
        self.assertGreater(result['candidate_evaluations'],0)

    def test_exact_affected_edge_objective_matches_cycle_enumeration(self):
        p,_,board,cells=self.fixture();g=Guidance(p);weights=[1+(i%5) for i in range(len(g.edges))]
        def rank(candidate):
            matched=[p.faces[candidate[a]][sa]==p.faces[candidate[b]][sb] for a,b,sa,sb in g.edges]
            return sum(w for w,m in zip(weights,matched) if m),sum(matched)
        oracle=rank(board);true_oracle=actual_score(p,board)
        for order in (cells,[cells[0],cells[2],cells[1]]):
            choices=[[s for s in p.domains[c] if s//4==board[order[(i+1)%3]]//4]
                     for i,c in enumerate(order)]
            for states in itertools.product(*choices):
                candidate=list(board)
                for c,s in zip(order,states):candidate[c]=s
                self.assertTrue(check_solution(replace(p,edges=()),candidate))
                oracle=max(oracle,rank(candidate));true_oracle=max(true_oracle,actual_score(p,candidate))
        result=g.ejection_repair(board,seed=31,max_cycle=3,beam_width=16,
                                  max_evaluations=1000,rotations_per_cycle=4096,edge_weights=weights)
        self.assertEqual((result['guided_score'],result['score']),oracle)
        self.assertLessEqual(result['candidate_evaluations'],128)
        self.assertEqual(result['termination'],'perfect_board_validated')
        self.assertEqual(result['best_true_score'],true_oracle)
        self.assertEqual(actual_score(p,result['best_true_board']),true_oracle)
        self.assertTrue(check_solution(replace(p,edges=()),result['board']))
        self.assertEqual(result,g.ejection_repair(board,seed=31,max_cycle=3,beam_width=16,
                         max_evaluations=1000,rotations_per_cycle=4096,edge_weights=weights))

    def test_bounds_cancel_domains_and_clues(self):
        for seed in range(5):
            p,answer=generated_problem(4,7,seed,clue_count=2);board=scrambled(p,answer,seed+90);g=Guidance(p)
            result=g.ejection_repair(board,seed=seed,max_evaluations=9,max_expansions=24)
            self.assertLessEqual(result['candidate_evaluations'],9);self.assertLessEqual(result['path_expansions'],24)
            self.assertTrue(check_solution(replace(p,edges=()),result['board']))
            self.assertEqual(result['score'],actual_score(p,result['board']))
            self.assertGreaterEqual(result['score'],actual_score(p,board))
            self.assertEqual(g.ejection_repair(board,should_stop=lambda:True)['board'],board)
        p,_,board,_=self.fixture()
        # Raw ExactProblem domains need not have been frame-filtered beforehand.
        raw=replace(p,domains=tuple(tuple(range(len(p.faces))) for _ in board))
        result=Guidance(raw).ejection_repair(board,max_evaluations=8)
        self.assertTrue(check_solution(replace(raw,edges=()),result['board']))

    def test_invalid_and_perfect_inputs(self):
        p,answer,board,_=self.fixture();g=Guidance(p)
        self.assertEqual(g.ejection_repair(answer)['candidate_evaluations'],0)
        for options in ({'seed':True},{'max_cycle':2},{'max_cycle':7},{'max_evaluations':0},
                        {'beam_width':17},{'max_expansions':0},{'rotations_per_cycle':4097},
                        {'edge_weights':[1]},{'should_stop':0}):
            with self.subTest(options=options),self.assertRaises(ValueError):g.ejection_repair(board,**options)
        with self.assertRaises(ValueError):Guidance(replace(p,edges=()))


class CorridorTests(unittest.TestCase):
    def fixture(self):
        p,answer=generated_problem(6,12,984)
        board=list(answer)
        for c in (7,28):board[c]=4*(board[c]//4)+(board[c]+1)%4
        return p,answer,board

    def test_connects_distinct_mismatch_clusters_within_cap(self):
        p,_,board=self.fixture();g=Guidance(p);cells=g.corridor_cells(board,seed=13,max_cells=9)
        self.assertEqual(cells,g.corridor_cells(board,seed=13,max_cells=9))
        self.assertLessEqual(len(cells),9)
        visited={cells[0]}
        while True:
            grown=visited|{c for v in visited for c in g.neighbors[v] if c in cells}
            if grown==visited:break
            visited=grown
        self.assertEqual(visited,set(cells))
        self.assertTrue(set(cells)&({7}|g.neighbors[7]));self.assertTrue(set(cells)&({28}|g.neighbors[28]))

    def test_clue_obstacles_perfect_and_cancellation(self):
        p,answer,board=self.fixture()
        fixed={c:board[c] for c in range(12,18)}
        p=replace(p,fixed=fixed);g=Guidance(p)
        cells=g.corridor_cells(board,seed=4,max_cells=15)
        self.assertFalse(set(cells)&fixed.keys())
        self.assertEqual(g.corridor_cells(board,should_stop=lambda:True),[])
        self.assertEqual(Guidance(replace(p,fixed={})).corridor_cells(answer),[])
        with self.assertRaises(ValueError):g.corridor_cells(board,max_cells=65)
        with self.assertRaises(ValueError):g.corridor_cells(board,seed=True)


if __name__=='__main__':unittest.main()
