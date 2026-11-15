from collections import deque
import itertools
import unittest

from exact_model import generated_problem
from exact_optimize import evaluate_board
from repair_plateau import explore


class PlateauControls(unittest.TestCase):
    def test_complete_reachability_and_exits(self):
        checked = set()
        for seed in (3,5,21):
            problem,_ = generated_problem(2,3,seed)
            scores={b:ans['score'] for b in itertools.product(*problem.domains)
                    if (ans:=evaluate_board(problem,b))['valid']}
            for board,score in scores.items():
                todo=deque([board])
                reached={board}
                exit_found=False
                while todo:
                    b=todo.popleft()
                    for other,value in scores.items():
                        if not 1<=sum(x!=y for x,y in zip(b,other))<=2:
                            continue
                        if value>score:
                            exit_found=True
                        elif value==score and other not in reached:
                            reached.add(other)
                            todo.append(other)
                res=explore(problem,board,max_boards=100)
                self.assertEqual(exit_found,res['improved'])
                if not exit_found:
                    self.assertEqual(('exhausted',len(reached)),
                                      (res['termination'],res['boards_fully_expanded']))
                checked.add(exit_found)
        self.assertEqual({True,False},checked)

    def test_cutoff_and_stop(self):
        p,b=generated_problem(3,2,5)
        self.assertEqual('work_limit',explore(p,b,max_work=0)['termination'])
        self.assertEqual('stopped',explore(p,b,should_stop=lambda:True)['termination'])
