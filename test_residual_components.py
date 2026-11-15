"""Independent residual-component oracles. No production puzzle runs."""
from dataclasses import replace
import itertools
import unittest
from exact_model import generated_problem, grid_edges, check_solution
from residual_components import solve_components


def masks(p):return [sum(1<<s for s in set(d)) for d in p.domains]


def binary_components(negative=True, groups=4):
    """Constructed exact-CSP fixture, not the Eternity II rotation inventory.

    Each square encodes equality around a cycle; final optional flipped edge
    contradicts the cycle. Physical identities are distinct and permanently
    assigned; these synthetic state tables need not be rotations of one tile.
    """
    p,answer=generated_problem(8,1,731)
    squares=[[9,10,17,18],[12,13,20,21],[33,34,41,42],[36,37,44,45]][:groups]
    faces=list(p.faces);domains=[(s,) for s in answer]
    for i,square in enumerate(squares):
        inside=[e for e in grid_edges(8) if e[0] in square and e[1] in square]
        for c in square:
            piece=answer[c]//4
            for bit in (0,1):
                f=[1]*4
                for a,b,sa,sb in inside:
                    if c==a:f[sa]=2+bit
                    if c==b:f[sb]=2+bit
                if negative and i==len(squares)-1 and c==square[-1]:f[0]=2+(1-bit)
                faces[4*piece+bit]=tuple(f)
            domains[c]=(4*piece,4*piece+1)
    return replace(p,faces=tuple(faces),domains=tuple(domains)),squares


def exhaustive(p):
    for b in itertools.product(*p.domains):
        if check_solution(p,list(b)):return list(b)
    return None


class ComponentTests(unittest.TestCase):
    def test_small_component_oracle(self):
        for negative in (False,True):
            for groups in (2,3):
                p,_=binary_components(negative,groups)
                r=solve_components(p,masks(p),max_work=1000000)
                wanted=exhaustive(p)
                self.assertEqual(r['outcome'],'solved' if wanted else 'infeasible')
                self.assertEqual(r['component_count'],groups)
                if wanted:self.assertTrue(check_solution(p,r['board']))

    def test_decomposition_avoids_cartesian_repetition(self):
        p,_=binary_components(True)
        split=solve_components(p,masks(p),max_work=1000000)
        mono=solve_components(p,masks(p),max_work=1000000,decompose=False)
        self.assertEqual(split['outcome'],mono['outcome'])
        self.assertEqual(split['outcome'],'infeasible')
        self.assertLess(split['candidate_placements'],mono['candidate_placements'])
        self.assertLess(split['work'],mono['work'])

    def test_shared_physical_piece_prevents_split(self):
        p,answer=generated_problem(6,1,12)
        a,b=7,28;piece_a,piece_b=answer[a]//4,answer[b]//4
        domains=[(s,) for s in answer]
        for c in (a,b):domains[c]=tuple(range(4*piece_a,4*piece_a+4))+tuple(range(4*piece_b,4*piece_b+4))
        p=replace(p,domains=tuple(domains))
        r=solve_components(p,masks(p),max_work=100000)
        self.assertEqual(r['outcome'],'not_decomposable')
        self.assertEqual(r['component_count'],1)

    def test_cancellation_and_caps_are_unknown_and_unchanged(self):
        p,_=binary_components(False);ds=masks(p);before=ds.copy()
        for cap in (0,5,100):
            r=solve_components(p,ds,max_work=cap)
            self.assertEqual(r['outcome'],'work_limit');self.assertFalse(r['complete'])
            self.assertLessEqual(r['work'],cap);self.assertIsNone(r['board'])
        r=solve_components(p,ds,should_stop=lambda:True)
        self.assertEqual(r['outcome'],'stopped');self.assertEqual(before,ds)

    def test_original_full_grid_clues_frames_and_inventory(self):
        p,answer=generated_problem(6,1,97)
        p=replace(p,fixed={0:answer[0],5:answer[5],35:answer[35]})
        free=[c for c in (7,8,21,22) if c not in p.fixed]
        domains=[(s,) for s in answer]
        for c in free:domains[c]=tuple(range((answer[c]//4)*4,(answer[c]//4)*4+4))
        p=replace(p,domains=tuple(domains))
        r=solve_components(p,masks(p),max_work=100000)
        self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(p,r['board']))

    def test_restricted_edge_task_scope(self):
        p,_=binary_components(True,2)
        p=replace(p,edges=())
        r=solve_components(p,masks(p),max_work=100000)
        self.assertEqual(r['outcome'],'solved');self.assertTrue(check_solution(p,r['board']))
        self.assertEqual(r['proof_scope'],'supplied_problem_and_exact_residual_domains')

    def test_invalid_and_singleton_failure(self):
        p,answer=generated_problem(2,3,42)
        with self.assertRaises(ValueError):solve_components(p,[True]*4)
        with self.assertRaises(ValueError):solve_components(p,masks(p),max_work=True)
        d=[1<<s for s in answer];d[0]=0
        r=solve_components(p,d)
        self.assertEqual(r['outcome'],'infeasible')

if __name__=='__main__':unittest.main()
