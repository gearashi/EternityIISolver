"""Exhaustive small-board oracles for regional score and compatibility bounds."""
from dataclasses import replace
import itertools
import unittest
from exact_model import generated_problem,grid_edges
from exact_optimize import evaluate_board
from repair_bounds import RegionBounds
from repair_exact_regions import RegionFailureCache,solve_region
from test_exact_sat import tiny_problem
from test_repair_exact_regions import legal_boards,scramble


def completions(p,reference,cells,prefix):
    selected=set(cells);pool={reference[c]//4 for c in cells}
    return [(score,b) for score,b in legal_boards(p)
            if all(b[c]==reference[c] for c in range(len(b)) if c not in selected)
            and {b[c]//4 for c in cells}==pool and all(b[c]==s for c,s in prefix.items())]


class RegionalBoundsTests(unittest.TestCase):
    def test_all_tiny_prefixes_upper_bound_true_optimum(self):
        checked=0
        for p in (tiny_problem(False),tiny_problem(True)):
            legal=legal_boards(p);reference=min(legal)[1]
            for cells in ([0,1],[0,1,2],[0,3],list(range(4))):
                model=RegionBounds(p,reference,cells)
                prefixes=[{}]
                for count in (1,2):
                    for locations in itertools.combinations(cells,count):
                        for values in itertools.product(*(model.domains[c] for c in locations)):
                            if len({s//4 for s in values})==count:prefixes.append(dict(zip(locations,values)))
                for prefix in prefixes:
                    possible=completions(p,reference,cells,prefix)
                    for method in (model.assignment_upper,model.clique_upper):
                        r=method(prefix,max_work=50000)
                        self.assertTrue(r['complete'],r)
                        if possible:
                            self.assertFalse(r['structurally_infeasible'])
                            self.assertGreaterEqual(r['upper_bound'],max(score for score,_ in possible))
                        if r.get('completion') is not None:
                            witness=reference.copy()
                            for c,s in {**prefix,**r['completion']}.items():witness[c]=s
                            self.assertTrue(evaluate_board(p,witness)['valid'])
                            self.assertEqual(evaluate_board(p,witness)['score'],r['upper_bound'])
                        checked+=1
        self.assertGreater(checked,100)

    def test_assignment_matching_respects_piece_competition(self):
        p=tiny_problem();board=min(legal_boards(p))[1]
        pieces=[s//4 for s in board]
        allowed=({pieces[0],pieces[2]},{pieces[1],pieces[3]},
                 {pieces[2],pieces[0]},{pieces[3],pieces[0]})
        q=replace(p,domains=tuple(tuple(s for s in d if s//4 in allowed[c]) for c,d in enumerate(p.domains)))
        model=RegionBounds(q,board,range(4))
        prefix={c:next(s for s in q.domains[c] if s//4==pieces[c+2]) for c in (0,1)}
        r=model.assignment_upper(prefix)
        self.assertTrue(r['complete']);self.assertTrue(r['structurally_infeasible'])

    def test_clique_detects_at_least_one_required_mismatch(self):
        p=tiny_problem(True);reference=min(legal_boards(p))[1]
        r=RegionBounds(p,reference,range(4)).clique_upper({})
        optimum=max(score for score,_ in legal_boards(p))
        self.assertTrue(r['complete']);self.assertLess(r['clique_size'],4)
        self.assertLess(r['upper_bound'],4);self.assertGreaterEqual(r['upper_bound'],optimum)

    def test_completed_prefix_score_exact(self):
        p=tiny_problem(True);reference=min(legal_boards(p))[1];model=RegionBounds(p,reference,range(4))
        for method in (model.assignment_upper,model.clique_upper):
            r=method(dict(enumerate(reference)))
            self.assertEqual(r['upper_bound'],evaluate_board(p,reference)['score'])

    def test_interruptions_and_limits_supply_no_bound(self):
        p=tiny_problem(True);reference=min(legal_boards(p))[1];model=RegionBounds(p,reference,range(4))
        for method in (model.assignment_upper,model.clique_upper):
            for kwargs in ({'max_work':0},{'max_work':3},{'should_stop':lambda:True}):
                r=method({},**kwargs)
                self.assertFalse(r['complete']);self.assertIsNone(r['upper_bound']);self.assertFalse(r['structurally_infeasible'])
        r=model.clique_upper({},max_vertices=1)
        self.assertEqual(r['termination'],'vertex_limit');self.assertIsNone(r['upper_bound'])

    def test_setup_reports_raw_selected_state_scan(self):
        p=tiny_problem();reference=min(legal_boards(p))[1];model=RegionBounds(p,reference,[0,1])
        self.assertEqual(model.setup_state_evaluations,len(p.domains[0])+len(p.domains[1]))
        self.assertEqual(model.setup_edges,3*len(grid_edges(p.size)))

    def test_snapshot_and_argument_guards(self):
        p=tiny_problem();reference=min(legal_boards(p))[1];model=RegionBounds(p,reference,[0,1])
        reference[0]=-1
        self.assertTrue(model.assignment_upper({})['complete'])
        for prefix in ({9:0},{0:True},{0:-1},{0:model.reference[2]},[] ):
            with self.assertRaises(ValueError):model.assignment_upper(prefix)
        for kwargs in ({'max_work':True},{'max_work':-1},{'should_stop':3}):
            with self.assertRaises(ValueError):model.clique_upper({},**kwargs)
        with self.assertRaises(ValueError):model.clique_upper({},max_vertices=0)
        with self.assertRaises(ValueError):RegionBounds(replace(p,edges=()),model.reference,[0])
        with self.assertRaises(ValueError):RegionBounds(p,model.reference,[0,0])

    def test_integrated_modes_match_all_small_targets(self):
        for p in (tiny_problem(),tiny_problem(True)):
            legal=legal_boards(p);reference=min(legal)[1];optimum=max(s for s,_ in legal)
            for algorithm in ('dfs','frontier'):
                for flags in ({'assignment_bound':True},{'clique_bound':True},{'assignment_bound':True,'clique_bound':True}):
                    for target in (optimum,optimum+1):
                        r=solve_region(p,reference,range(4),target_score=target,algorithm=algorithm,max_work=50000,**flags)
                        self.assertEqual(r['target_reached'],optimum>=target)
                        self.assertEqual(r['proven_unreachable'],optimum<target)
                        self.assertEqual(r['work'],r['candidate_placements']+r['bound_assignments']+r['assignment_work']+r['clique_work'])
                        self.assertTrue(evaluate_board(p,r['board'])['valid'])

    def test_complete_clique_witness_is_used_without_repeating_dfs(self):
        p=tiny_problem();reference=min(legal_boards(p))[1]
        r=solve_region(p,reference,range(4),target_score=4,algorithm='dfs',clique_bound=True)
        self.assertTrue(r['target_reached']);self.assertEqual(r['clique_completions'],1)
        self.assertEqual(r['candidate_placements'],0);self.assertGreater(r['clique_nodes'],0)
        self.assertGreater(r['work'],0)
        self.assertEqual(evaluate_board(p,r['board'])['score'],4)

    def test_integrated_limit_never_cached_as_proof(self):
        p=tiny_problem(True);reference=min(legal_boards(p))[1]
        for flag in ('assignment_bound','clique_bound'):
            for algorithm in ('dfs','frontier'):
                cache=RegionFailureCache()
                r=solve_region(p,reference,range(4),target_score=4,algorithm=algorithm,max_work=2,cache=cache,**{flag:True})
                self.assertFalse(r['proven_unreachable']);self.assertEqual(len(cache),0)
                self.assertLessEqual(r['work'],2)
                r=solve_region(p,reference,range(4),target_score=4,algorithm=algorithm,should_stop=lambda:True,cache=cache,**{flag:True})
                self.assertEqual(r['termination'],'stopped');self.assertEqual(len(cache),0)

    def test_clues_and_frame_preserved_on_three_by_three(self):
        p,answer=generated_problem(3,3,41,clue_count=1);reference=scramble(p,answer)
        for flag in ('assignment_bound','clique_bound'):
            r=solve_region(p,reference,[c for c in range(9) if c not in p.fixed],target_score=12,
                           algorithm='dfs',max_work=100000,**{flag:True})
            self.assertTrue(r['target_reached'],r)
            self.assertTrue(evaluate_board(p,r['board'])['valid'])
            self.assertTrue(all(r['board'][c]==s for c,s in p.fixed.items()))

if __name__=='__main__':unittest.main()
