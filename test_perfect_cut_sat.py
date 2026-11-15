import itertools
from dataclasses import replace
import unittest

from exact_model import check_solution, generated_problem
from perfect_cut_sat import certificate, solve
from perfect_distance import DistanceBounds, verify_packing


def changed_parent(p, known):
    # Try swaps in deterministic order, retaining the original frame and clues.
    for i,j in itertools.combinations(range(len(known)),2):
        if i in p.fixed or j in p.fixed:
            continue
        for s in p.domains[i]:
            if s//4 != known[j]//4:
                continue
            for t in p.domains[j]:
                if t//4 != known[i]//4:
                    continue
                parent = list(known)
                parent[i],parent[j] = s,t
                if not check_solution(p,parent):
                    return parent
    raise ValueError('No broken legal parent among tested swaps')


class VerifiedCuts(unittest.TestCase):
    def setUp(self):
        self.p, self.known = generated_problem(2,4,6011)
        self.parent = changed_parent(self.p,self.known)
        self.k = DistanceBounds(self.p,self.parent).local_minimum(range(4))
        self.rows = [(tuple(range(4)),self.k,1)]
        self.cert = certificate(self.p,self.parent,self.rows,1)

    def test_all_radii_match_independent_enumeration(self):
        p,parent = self.p,self.parent
        distances = [sum(a!=b for a,b in zip(candidate,parent)) for candidate in itertools.product(*p.domains)
                     if check_solution(p,candidate)]
        for rad in [None,*range(5)]:
            expected = 'solved' if rad is None or min(distances)<=rad else 'infeasible'
            for cert in (None,self.cert):
                r = solve(p,parent,radius=rad,cert=cert)
                self.assertEqual(expected,r['outcome'])
                if cert:
                    self.assertEqual(r['lower_bound'],verify_packing(p,parent,self.rows,1)['lower_bound'])
                if r['board'] is not None:
                    self.assertTrue(check_solution(p,r['board']))
        self.assertGreater(self.k,0)

    def test_wrong_scope_invalid_weight_or_unproved_cut(self):
        p,b,cert = self.p,self.parent,self.cert
        for q,other,changed in ((replace(p,edges=()),b,cert),(p,self.known,cert),
                (replace(p,domains=tuple(tuple(reversed(d)) for d in p.domains)),b,cert),
                (p,b,dict(cert,objective='score-467')), (p,b,dict(cert,rows=self.rows*2)),
                (p,b,dict(cert,rows=[([1],1,1)])), (p,b,dict(cert,rows=[([[]],1,1)])),
                (p,b,dict(cert,rows=[([0,0],1,1)])), (p,b,dict(cert,scale=False)),
                (p,b,dict(cert,schema=True))):
            with self.assertRaises(ValueError):
                solve(q,other,cert=changed)
        with self.assertRaises(ValueError):
            solve(p,[b[0]]*4)

    def test_changed_neighbors_and_independent_exact_checker(self):
        from regional_distance_replay import verify_exact_packing
        for seed in range(5999,6006):
            p,known=generated_problem(2,4,seed)
            parent=changed_parent(p,known)
            best=min(sum(a!=b for a,b in zip(candidate,parent))
                     for candidate in itertools.product(*p.domains) if check_solution(p,candidate))
            for k in range(5):
                rows=[(tuple(range(4)),k,1)]
                cert=certificate(p,parent,rows,1)
                if k<=best:
                    raw=verify_exact_packing(p,parent,rows,1)
                    r=solve(p,parent,cert=cert)
                    self.assertEqual(r['lower_bound'],raw['lower_bound'])
                    self.assertEqual(r['outcome'],'solved')
                else:
                    with self.assertRaises(ValueError):
                        verify_exact_packing(p,parent,rows,1)
                    with self.assertRaises(ValueError):
                        solve(p,parent,cert=cert)

    def test_stop_timeout_never_prove_radius(self):
        for opt in (dict(seconds=0),dict(should_stop=lambda:True)):
            r=solve(self.p,self.parent,radius=0,cert=self.cert,**opt)
            self.assertIn(r['outcome'],('timeout','stopped'))
            self.assertFalse(r['native_started'])


if __name__=='__main__':
    unittest.main()
