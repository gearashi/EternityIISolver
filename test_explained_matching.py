"""Finite independent oracles for explained all-different matching propagation."""
from dataclasses import replace
import itertools
import random
import unittest

import explained_dfs
from exact_model import check_solution, generated_problem, grid_edges

WORK_KEYS = ('setup_state_visits','domain_copies','reason_copies','arc_state_checks',
    'piece_state_checks','reason_literals','reason_joins','removed_states',
    'nogood_literal_checks','decision_placements','propagation_rounds',
    'matching_state_visits','matching_edge_visits','matching_graph_visits','matching_reason_checks')


def oracle(problem):
    return [list(board) for board in itertools.product(*problem.domains)
            if check_solution(problem,board)]


def hall_fixture(shortage=False):
    """Physical equal-color 5x5 with nine free interior pieces, fixed frame."""
    problem,known=generated_problem(5,1,841)
    cells=[r*5+c for r in range(1,4) for c in range(1,4)]
    pool=[known[c] for c in cells]
    domains=[(known[c],) for c in range(25)]
    for i,cell in enumerate(cells):
        domains[cell]=tuple(pool[:3] if i<(4 if shortage else 3) else pool)
    fixed={c:s for c,s in enumerate(known) if c not in cells}
    return replace(problem,domains=tuple(domains),fixed=fixed),cells,pool


def search(problem,**kwargs):
    params=dict(seconds=10,seed=101,workers=1,should_stop=None,on_progress=None,
        learning=True,max_work=10000000,max_nogoods=256,max_nogood_literals=64,
        include_nogoods=True,matching=True)
    params.update(kwargs)
    return explained_dfs._Search(problem,**params)


class ExplainedMatchingTests(unittest.TestCase):
    def test_hall_shortage_is_root_proof_without_placements(self):
        p,_,_=hall_fixture(True)
        off=explained_dfs.solve(p,seconds=5,seed=101,matching=False,max_work=10000000)
        on=explained_dfs.solve(p,seconds=5,seed=101,matching=True,max_work=10000000)
        self.assertEqual(off['outcome'],'infeasible');self.assertEqual(on['outcome'],'infeasible')
        self.assertEqual(on['branches'],0);self.assertGreater(off['branches'],0)
        self.assertGreater(on['reasoning_counters_session']['matching_hall_failures'],0)

    def test_scc_tight_hall_set_deletes_only_reserved_pieces(self):
        p,cells,pool=hall_fixture()
        solver=search(p);domains,reasons=solver.setup()
        changed,failure=solver.matching_filter(domains,reasons)
        self.assertTrue(changed);self.assertIsNone(failure)
        self.assertEqual(solver.counters['matching_removed_states'],18)
        for cell in cells[:3]:self.assertEqual(set(explained_dfs._bits(domains[cell])),set(pool[:3]))
        for cell in cells[3:]:
            self.assertEqual(set(explained_dfs._bits(domains[cell])),set(pool[3:]))
            self.assertTrue(all(not reasons[cell][s] for s in pool[:3]))
        self.assertEqual(solver.work,sum(solver.counters[k] for k in WORK_KEYS))

    def test_all_small_graphs_match_independent_permutation_support(self):
        # 4 corners, 4 genuinely rotated pieces; exact permutations are an
        # independent all-different support oracle, not another matching routine.
        p,_=generated_problem(2,1,73)
        rng=random.Random(2187)
        for _ in range(96):
            ds=tuple(tuple(s for s in d if rng.random()<.62) for d in p.domains)
            q=replace(p,domains=ds,edges=())
            solutions=oracle(q)
            solver=search(q);domains,reasons=solver.setup()
            changed,failure=solver.matching_filter(domains,reasons)
            if not solutions:
                self.assertEqual(failure,frozenset());continue
            self.assertIsNone(failure)
            for cell,mask in enumerate(domains):
                supported={board[cell] for board in solutions}
                self.assertEqual(set(explained_dfs._bits(mask)),supported)
                self.assertTrue(all(not reason for reason in reasons[cell].values()))

    def test_every_deduction_and_learned_clause_respects_actual_solutions(self):
        rng=random.Random(569)
        for seed in range(32):
            p,_=generated_problem(2,3,seed)
            ds=tuple(tuple(s for s in d if rng.random()<.8) for d in p.domains)
            edges=tuple(e for e in grid_edges(2) if rng.random()<.8)
            p=replace(p,domains=ds,edges=edges);solutions=oracle(p)
            for learning in (False,True):
                solver=search(p,seed=seed,learning=learning)
                remove=solver.remove
                def audited_remove(c,state,why,domains,reasons):
                    self.assertFalse(any(board[c]==state and all(board[a]==s for a,s in why)
                                         for board in solutions),(c,state,why))
                    return remove(c,state,why,domains,reasons)
                solver.remove=audited_remove
                result=solver.run()
                self.assertEqual(result['outcome'],'solved' if solutions else 'infeasible')
                for clause in result['learned_nogoods']:
                    self.assertFalse(any(all(board[c]==s for c,s in clause) for board in solutions))
                self.assertEqual(result['work'],sum(solver.counters[k] for k in WORK_KEYS))

    def test_scc_reasons_include_decision_excluding_external_piece(self):
        # A 2-cell Hall set emerges only after a decision elsewhere removes its
        # spare piece. Check the full implication against independent witnesses.
        p,_=generated_problem(2,1,19)
        # Each position has one correctly oriented state for each of 4 pieces.
        by_piece=[{s//4:s for s in d} for d in p.domains]
        ds=list(p.domains)
        ds[0]=tuple(by_piece[0][i] for i in (0,1,2))
        ds[1]=tuple(by_piece[1][i] for i in (0,1,2))
        ds[2]=tuple(by_piece[2][i] for i in (2,3))
        p=replace(p,domains=tuple(ds),edges=())
        solutions=oracle(p);solver=search(p);domains,reasons=solver.setup()
        literal=(2,by_piece[2][2]);why=frozenset((literal,))
        for cell in range(4):
            for state in tuple(explained_dfs._bits(domains[cell])):
                if (cell==2 and state!=literal[1]) or (cell!=2 and state//4==2):
                    solver.remove(cell,state,why,domains,reasons)
        changed,failure=solver.matching_filter(domains,reasons)
        self.assertTrue(changed);self.assertIsNone(failure)
        for piece in (0,1):
            state=by_piece[3][piece]
            self.assertIn(literal,reasons[3][state])
            self.assertFalse(any(b[3]==state and all(b[c]==s for c,s in reasons[3][state]) for b in solutions))
        # The implication would be false if its decision reason were omitted.
        self.assertTrue(any(b[3]//4 in (0,1) for b in solutions))

    def test_hall_failure_retains_its_essential_decision_reason(self):
        p,_=generated_problem(2,1,29)
        by_piece=[{s//4:s for s in d} for d in p.domains]
        ds=[tuple(by_piece[c][i] for i in (0,1,2)) for c in range(3)]
        ds.append(tuple(by_piece[3][i] for i in (2,3)))
        p=replace(p,domains=tuple(ds),edges=())
        solutions=oracle(p);self.assertTrue(solutions)
        solver=search(p);domains,reasons=solver.setup()
        literal=(3,by_piece[3][2]);why=frozenset((literal,))
        for cell in range(4):
            for state in tuple(explained_dfs._bits(domains[cell])):
                if (cell==3 and state!=literal[1]) or (cell!=3 and state//4==2):
                    solver.remove(cell,state,why,domains,reasons)
        _,failure=solver.matching_filter(domains,reasons)
        self.assertEqual(failure,why)
        self.assertFalse(any(all(b[c]==state for c,state in failure) for b in solutions))
        solver.conflict(failure)
        self.assertIn(tuple(sorted(why)),solver.nogoods)
        # Empty root reasons would incorrectly declare this satisfiable problem
        # impossible; its failure is specifically conditional on the decision.
        result=explained_dfs.solve(p,seconds=3,matching=True,include_nogoods=True)
        self.assertEqual(result['outcome'],'solved')
        self.assertTrue(check_solution(p,result['board']))

    def test_matching_interruptions_never_become_negative_proofs(self):
        p,_,_=hall_fixture(True)
        for cap in (0,1,100,1000,7000):
            result=explained_dfs.solve(p,seconds=4,matching=True,max_work=cap)
            if result['outcome']=='infeasible':continue
            self.assertEqual(result['outcome'],'timeout');self.assertFalse(result['coverage_verified'])
            self.assertIsNone(result['board']);self.assertLessEqual(result['work'],cap)
        result=explained_dfs.solve(p,matching=True,should_stop=lambda:True)
        self.assertEqual(result['outcome'],'stopped');self.assertFalse(result['coverage_verified'])
        with self.assertRaises(ValueError):explained_dfs.solve(p,matching=1)

    def test_clues_rotations_and_frame_preserved(self):
        for size,colors,seed in ((3,2,271),(4,5,727),(5,5,911)):
            p,_=generated_problem(size,colors,seed,clue_count=2)
            result=explained_dfs.solve(p,seconds=5,matching=True,seed=seed,max_work=10000000)
            self.assertEqual(result['outcome'],'solved');self.assertTrue(check_solution(p,result['board']))
            self.assertTrue(result['matching'])


if __name__=='__main__':unittest.main()
