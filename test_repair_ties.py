"""Independent finite checks for seeded equal-primary-score repair choices."""
from dataclasses import replace
import itertools
import random
import unittest
from unittest.mock import patch

import numpy as np

from exact_model import check_solution, frame_ok, generated_problem, grid_edges
from gpu_assignment import AssignmentEvaluator, assignment_arc_costs, cpu_cost_matrix
from smart_repair import independent_cells
from test_gpu_assignment import scrambled


def objective(problem, board):
    return sum(problem.faces[board[a]][sa] == problem.faces[board[b]][sb]
               for a,b,sa,sb in grid_edges(problem.size))


def match(matrix):
    from ortools.graph.python import linear_sum_assignment
    costs = assignment_arc_costs(matrix)
    n = len(matrix['cells'])
    solver = linear_sum_assignment.SimpleLinearSumAssignment()
    for row in range(n):
        for column in range(n):
            if matrix['states'][row,column] >= 0:
                solver.add_arc_with_cost(row,column,int(costs[row,column]))
    if solver.solve() != solver.OPTIMAL:
        raise AssertionError('Feasible fixture was lost')
    return tuple(solver.right_mate(row) for row in range(n))


class RepairTieTests(unittest.TestCase):
    def test_seeded_rotation_choices_preserve_every_primary_matrix_entry(self):
        changes = 0
        for problem_seed in range(4):
            p,answer = generated_problem(5,3,problem_seed,clue_count=2)
            board = scrambled(p,answer,problem_seed+101)
            p = replace(p,domains=tuple(tuple(s for s in d if s == board[c] or s % 3 != 0)
                                        for c,d in enumerate(p.domains)))
            cells = independent_cells(p,problem_seed)
            baseline = cpu_cost_matrix(p,board,cells)
            for seed in (0,1,17,2**32-1):
                tied = cpu_cost_matrix(p,board,cells,tie_seed=seed)
                np.testing.assert_array_equal(tied['weights'],baseline['weights'])
                repeated = cpu_cost_matrix(p,board,cells,tie_seed=seed)
                np.testing.assert_array_equal(tied['states'],repeated['states'])
                changes += np.count_nonzero(tied['states'] != baseline['states'])
                for i,c in enumerate(cells):
                    for j,source in enumerate(cells):
                        state = int(tied['states'][i,j])
                        allowed = [s for s in p.domains[c]
                                   if s//4 == board[source]//4 and frame_ok(p.size,c,p.faces[s])]
                        if not allowed:
                            self.assertEqual(state,-1)
                            continue
                        self.assertIn(state,allowed)
                        # Score the actual incident edges, independently of the
                        # matrix implementation's neighbor construction.
                        scores = []
                        for candidate in allowed:
                            trial = list(board);trial[c] = candidate
                            scores.append(sum(p.faces[trial[a]][sa] == p.faces[trial[b]][sb]
                                              for a,b,sa,sb in grid_edges(p.size) if c in (a,b)))
                        self.assertEqual(int(tied['weights'][i,j]),max(scores))
        self.assertGreater(changes,0)

    def test_secondary_objective_cannot_replace_a_primary_match(self):
        rng = random.Random(561)
        for n in range(2,6):
            matrix = {'board':[4*c for c in range(n)],'cells':list(range(n)),
                      'states':np.tile(np.arange(n,dtype=np.int32)*4,(n,1)),
                      'weights':np.asarray([[rng.randrange(5) for _ in range(n)] for _ in range(n)],dtype=np.int32)}
            permutations = list(itertools.permutations(range(n)))
            primary = {perm:sum(int(matrix['weights'][i,j]) for i,j in enumerate(perm))
                       for perm in permutations}
            for seed in (0,1,17,2**32-1):
                matrix['tie_seed'] = seed
                costs = assignment_arc_costs(matrix)
                totals = {perm:sum(int(costs[i,j]) for i,j in enumerate(perm)) for perm in permutations}
                chosen = match(matrix)
                self.assertEqual(primary[chosen],max(primary.values()))
                self.assertEqual(totals[chosen],min(totals.values()))
                # Check all feasible pairs, not just the chosen optimum.
                for better in permutations:
                    for worse in permutations:
                        if primary[better] > primary[worse]:
                            self.assertLess(totals[better],totals[worse])

    def test_equal_score_assignments_explore_multiple_permutations(self):
        n=4
        matrix={'board':[4*c for c in range(n)],'cells':list(range(n)),
                'states':np.tile(np.arange(n,dtype=np.int32)*4,(n,1)),
                'weights':np.ones((n,n),dtype=np.int32)}
        choices=[]
        for seed in range(16):
            matrix['tie_seed']=seed
            chosen=match(matrix)
            self.assertEqual(chosen,match(matrix))
            choices.append(chosen)
        self.assertGreater(len(set(choices)),1)

    def test_seeded_actual_board_repair_matches_exhaustive_primary_optimum(self):
        p,answer=generated_problem(3,3,42,clue_count=1)
        board=scrambled(p,answer,21)
        cells=independent_cells(p,13)[:4]
        structure=replace(p,edges=())
        optimum=-1
        for pieces in itertools.permutations([board[c]//4 for c in cells]):
            for states in itertools.product(*[[s for s in p.domains[c] if s//4==piece]
                                               for c,piece in zip(cells,pieces)]):
                trial=list(board)
                for c,s in zip(cells,states):trial[c]=s
                if check_solution(structure,trial):optimum=max(optimum,objective(p,trial))
        for seed in (0,1,17,2**32-1):
            matrix=cpu_cost_matrix(p,board,cells,tie_seed=seed)
            chosen=match(matrix);trial=list(board)
            for i,c in enumerate(cells):trial[c]=int(matrix['states'][i,chosen[i]])
            self.assertTrue(check_solution(structure,trial))
            self.assertEqual(objective(p,trial),optimum)
            self.assertTrue(all(trial[c]==s for c,s in p.fixed.items()))

    def test_illegal_arcs_and_bounded_signed_integer_costs(self):
        n=256
        matrix={'board':[4*c for c in range(n)],'cells':list(range(n)),
                'states':np.tile(np.arange(n,dtype=np.int32)*4,(n,1)),
                'weights':np.full((n,n),4,dtype=np.int32),'tie_seed':2**32-1}
        costs=assignment_arc_costs(matrix)
        self.assertEqual(costs.dtype,np.int64)
        self.assertLessEqual(int(np.abs(costs).max())*n,67000000)
        n=4
        matrix={'board':[4*c for c in range(n)],'cells':list(range(n)),
                'states':np.full((n,n),-1,dtype=np.int32),
                'weights':np.full((n,n),-1,dtype=np.int32),'tie_seed':17}
        for i in range(n):matrix['states'][i,i]=4*i;matrix['weights'][i,i]=0
        self.assertEqual(match(matrix),tuple(range(n)))
        matrix['tie_seed']=None
        np.testing.assert_array_equal(assignment_arc_costs(matrix),-matrix['weights'])

    def test_batch_ties_match_scalar_and_stay_independent_of_batch_partition(self):
        p,answer=generated_problem(5,3,7,clue_count=2)
        jobs=[(scrambled(p,answer,seed),independent_cells(p,seed)) for seed in range(4)]
        for seed in (0,17,2**32-1):
            for batch_size in (1,3,16):
                evaluator=AssignmentEvaluator(p,backend='cpu',batch_size=batch_size)
                actual=evaluator.evaluate(jobs,tie_seed=seed)
                for (board,cells),got in zip(jobs,actual):
                    want=cpu_cost_matrix(p,board,cells,tie_seed=seed)
                    np.testing.assert_array_equal(got['weights'],want['weights'])
                    np.testing.assert_array_equal(got['states'],want['states'])
                    np.testing.assert_array_equal(assignment_arc_costs(got),assignment_arc_costs(want))

    def test_invalid_tie_seeds_and_malformed_matrices_rejected(self):
        p,board=generated_problem(3,3,7)
        evaluator=AssignmentEvaluator(p,backend='cpu')
        matrix=cpu_cost_matrix(p,board,[0])
        for seed in (-1,2**32,True,1.0,'1'):
            with self.assertRaises(ValueError):cpu_cost_matrix(p,board,[0],tie_seed=seed)
            with self.assertRaises(ValueError):evaluator.evaluate([(board,[0])],tie_seed=seed)
            with self.assertRaises(ValueError):assignment_arc_costs(dict(matrix,tie_seed=seed))
        for weights in (np.asarray([[5]]),np.asarray([[-2]]),np.asarray([[1.0]]),np.empty((0,0))):
            with self.assertRaises(ValueError):assignment_arc_costs(dict(matrix,weights=weights))

    def test_reused_structure_checks_each_board_without_rebuilding_rules(self):
        p,board=generated_problem(4,4,61,clue_count=2)
        evaluator=AssignmentEvaluator(p,backend='cpu')
        with patch('gpu_assignment.replace',side_effect=AssertionError('No rebuild')):
            self.assertTrue(evaluator.structurally_valid(board))
            repeated=list(board);repeated[0]=repeated[1]
            self.assertFalse(evaluator.structurally_valid(repeated))
            wrong_frame=list(board);wrong_frame[0]=(wrong_frame[0]//4)*4+(wrong_frame[0]+1)%4
            self.assertFalse(evaluator.structurally_valid(wrong_frame))
            alias=list(board);alias[0]=float(alias[0])
            self.assertFalse(evaluator.structurally_valid(alias))
        # Original mutable clues can change, but the detached operation's rules
        # never change silently, and explicit compatibility detects mutation.
        fixed=next(iter(p.fixed));p.fixed[fixed]=(p.fixed[fixed]+1)%len(p.faces)
        self.assertFalse(evaluator.matches_problem(p))
        self.assertTrue(evaluator.structurally_valid(board))


if __name__=='__main__':unittest.main()
