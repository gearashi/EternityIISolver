from dataclasses import replace
import random
import unittest
from exact_model import make_problem,generated_problem,grid_edges,check_solution
from test_refined_ordering import independent_oracle
from regional_master import solve


def diagonal_color_shortage():
    # Both copies of color 3 belong to pieces confined to diagonal cells.
    # Every edge has unary support for color 1 but none can match color 3.
    base=[(0,1,3,0),(0,3,1,0),(0,1,1,0),(0,1,1,0)]
    faces=tuple(tuple(f[(side-rot)%4] for side in range(4)) for f in base for rot in range(4))
    p=make_problem(2,faces)
    return replace(p,domains=tuple(tuple(s for s in d if c in (0,3) or s//4>=2) for c,d in enumerate(p.domains)))


class SharedInventoryTests(unittest.TestCase):
    def test_total_and_per_edge_supports_differ(self):
        p=diagonal_color_shortage()
        modes={}
        for mode in ('none','counts','edge_colors'):
            r=solve(p,regions=((0,1,2,3),),minimize_interfaces=True,allocation_cores=True,hall_capacity=True,
                    color_conservation=True,color_coupling=mode,seconds=3)
            self.assertEqual(r['outcome'],'infeasible')
            modes[mode]=r['local_queries']
        self.assertGreater(modes['counts'],0)
        self.assertEqual(modes['edge_colors'],0)
        self.assertIsNone(independent_oracle(p))

    def test_partial_edges_and_arbitrary_state_faces_preserve_solutions(self):
        for seed in range(18):
            p,known=generated_problem(2,1+seed%3,seed,clue_count=seed%2)
            rng=random.Random(seed+51)
            faces=list(p.faces)
            if seed%2:
                x=known[0];faces[x]=tuple(9 if color else 0 for color in faces[x])
            q=replace(p,faces=tuple(faces),domains=tuple(tuple(s for s in d if rng.random()<.8) for d in p.domains),
                      edges=tuple(e for e in grid_edges(2) if rng.random()<.7))
            want=independent_oracle(q)
            for mode in ('counts','edge_colors'):
                r=solve(q,color_coupling=mode,color_conservation=True,allocation_cores=True,hall_capacity=True,
                        minimize_interfaces=True,max_master_queries=300,seconds=3)
                self.assertEqual(r['outcome'],'solved' if want else 'infeasible',(seed,mode))
                if want:self.assertTrue(check_solution(q,r['board']))
