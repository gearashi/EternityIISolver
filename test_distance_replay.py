import contextlib
from dataclasses import replace
import io
import itertools
import json
import unittest
from unittest.mock import Mock,patch

from exact_model import generated_problem,constraint_edges
from perfect_distance_cli import main
from regional_distance_replay import verify_exact_packing


class ExactDistanceReplayTests(unittest.TestCase):
    def test_rejects_false_minimum_duplicate_cell_and_overweight_overlap(self):
        p,solution=generated_problem(2,2,9)
        for rows in ([([0,1],1,16)],[([0,0],0,1)],
                     [([0,1],0,9),([1,2],0,8)],[([4],0,1)],[([0],0,-1)]):
            with self.assertRaises(ValueError):
                verify_exact_packing(p,solution,rows,16)
        good=verify_exact_packing(p,solution,[([0,1],0,16)],16)
        self.assertEqual(good['lower_bound'],0)

    def test_missing_edge_invalidates_its_distance_row(self):
        p,_=generated_problem(2,3,47)
        for board in itertools.product(*p.domains):
            if len({s//4 for s in board})!=4:continue
            edge=next((e for e in constraint_edges(p)
                       if p.faces[board[e[0]]][e[2]]!=p.faces[board[e[1]]][e[3]]),None)
            if edge:break
        a,b,_,_=edge
        rows=[((a,b),1,16)]
        self.assertEqual(verify_exact_packing(p,board,rows,16)['lower_bound'],1)
        with self.assertRaises(ValueError):
            verify_exact_packing(replace(p,edges=()),board,rows,16)

    def test_cli_default_and_original_method_dispatch(self):
        p,solution=generated_problem(2,2,41)
        source=Mock()
        source.read_text.return_value=json.dumps({'board':solution})
        with patch('perfect_distance_cli.Path',return_value=source), \
             patch('perfect_distance_cli.load_bundle',return_value=object()), \
             patch('perfect_distance_cli.local_board_hash',return_value='abc123') as digest, \
             patch('perfect_distance_cli.validate_board',return_value=dict(valid=True,score=4)), \
             patch('perfect_distance_cli.from_bundle',return_value=p):
            for args,mode in (([],'edges'),(['--method','independent'],'independent')):
                stream=io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(main(['board.json',*args]),0)
                row=json.loads(stream.getvalue())
                self.assertEqual((row['method'],row['lower_bound']),(mode,0))
                self.assertEqual(row['parent_local_sha256'],'abc123')
                self.assertEqual(tuple(digest.call_args.args[0]),tuple(solution))
