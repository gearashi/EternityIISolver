from dataclasses import replace
from itertools import combinations
from math import comb
import os
from pathlib import Path
import unittest

from exact_model import generated_problem
from repair_signed_cover import decode, plans


def eight_fixture():
    p,b=generated_problem(7,4,19)
    cells=(16,17)
    pads=(12,26,30,32,38,40)
    before=[list(p.faces[s]) for s in b]
    before[16],before[17]=[2,1,3,1],[3,1,2,1]
    for c,side,color in ((9,2,1),(23,0,4),(15,1,4),(10,2,4),(18,3,4),(24,0,4)):
        before[c][side]=color
    for c in pads:before[c]=[4]*4
    faces=list(p.faces)
    for state,face in zip(b,before):
        for r in range(4):
            faces[state//4*4+(state+r)%4]=tuple(face[(i-r)%4] for i in range(4))
    ds=[(s,) for s in b]
    for c in cells:
        s=b[c]
        ds[c]=(s,s//4*4+(s+1)%4)
    for i,c in enumerate(pads):ds[c]=(b[c],b[pads[(i+1)%len(pads)]])
    return replace(p,faces=tuple(faces),domains=tuple(ds)),b


class SignedRadiusTests(unittest.TestCase):
    def test_plan_multiplicities_and_negative_cost_group(self):
        hist={(1,-1):3,(1,0):3,(2,2):2,(3,0):1}
        groups,entries=plans(hist,4)
        n=0
        for ids in entries:
            terms=1
            for i in set(ids):terms*=comb(hist[groups[i]],ids.count(i))
            n+=terms
        flat=[key for key,count in hist.items() for _ in range(count)]
        expected=sum(sum(t[0] for t in row)==4 and sum(t[1] for t in row)>0
                     for k in range(1,5) for row in combinations(flat,k))
        self.assertEqual(n,expected)

    def test_planner_and_native_work_limits(self):
        from repair_signed_cover import PlanCutoff
        seen=0
        def charge(amount=1):
            nonlocal seen
            if seen+amount>20:raise PlanCutoff('work_limit')
            seen+=amount
        with self.assertRaisesRegex(PlanCutoff,'work_limit'):
            plans({(1,-1):3,(1,0):3,(2,2):2,(3,0):1},6,charge)
        self.assertLessEqual(seen,20)

    def test_all_eight_orientations_and_transfers(self):
        # Exercise the positive exact-eight worker directly: the full-ball API
        # legitimately stops at an earlier two-cell repair of the same parent.
        from subprocess import run
        from exact_model import initial_domains
        from repair_transfer_bound import certificate
        from repair_signed_cover import prepare
        helper = 'eternity-radius.exe' if os.name == 'nt' else 'eternity-radius'
        exe = Path(__file__).resolve().parent / 'build' / 'native-helper' / helper
        if not exe.is_file():self.skipTest('local native helper unavailable')
        p,b=eight_fixture()
        prices=certificate(p,b)['potentials']
        info,suffix=prepare(p,b,prices,8)
        self.assertEqual(info['status'],'complete')
        cap=20000000
        data=[f'{p.size} 8 {cap}',*(' '.join(map(str,f)) for f in p.faces),
              ' '.join(map(str,b)),' '.join(map(str,prices))]
        data+=[' '.join(map(str,(len(d),*d))) for d in initial_domains(p)]
        response=run([str(exe),'--signed-family'],input='\n'.join(data)+'\n'+suffix,
                     capture_output=True,text=True,check=True)
        status,counts,child=decode(response.stdout,p,b,8,cap,info)
        self.assertEqual(status,'family_complete')
        self.assertEqual(sum(s!=t for s,t in zip(child,b)),8)
        self.assertEqual(counts['gain'],1)
        data[0]=f'{p.size} 8 1'
        limited=run([str(exe),'--signed-family'],input='\n'.join(data)+'\n'+suffix,
                    capture_output=True,text=True,check=True)
        state,counts,child=decode(limited.stdout,p,b,8,1,info)
        self.assertEqual((state,counts['work'],counts['nominal_covered'],child),
                         ('work_limit',1,0,b))

    def test_incomplete_signed_receipt(self):
        p,b=eight_fixture()
        header='family_complete 0 18 11 0 0 0 0 7\n'
        info=dict(status='complete',patterns=4,shape_rows=31,nominal=149)
        for raw in (header,header+'signed_rows 31 3 3 149',
                    header+'signed_rows 30 4 4 149',header+'signed_rows 31 4 4 148'):
            with self.assertRaises(RuntimeError):
                decode(raw,p,b,8,1000,info)


if __name__=='__main__':
    unittest.main()
