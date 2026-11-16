"""Safety and checkpoint compatibility for optional pivot proofs on small puzzles."""
from dataclasses import replace
from itertools import product
from pathlib import Path
import unittest
from unittest.mock import patch

import dashboard_server
import exact_dfs as dfs
import exact_parallel
import exact_worker
import impact_dfs
import launcher
from dfs_profiles import PRUNING_PROFILES, dfs_profile_options
from exact_model import check_solution, generated_problem, grid_edges, make_problem
from pivot_cut import prove_pivot

def ring_problem(odd=True):
    size=8
    base,board=generated_problem(size,3,314)
    faces=list(base.faces)
    rows=[[state] for state in board]
    inner={c for c in range(size**2) if 0<c//size<size-1 and 0<c%size<size-1}
    ring={r*size+x for r in (1,2,3) for x in (1,2,3) if (r,x)!=(2,2)}
    wide=2*size+2
    dummy=inner-ring-{wide}
    for c in inner:
        s=board[c]
        b=(1,1,2,2) if c in ring else (3,3,3,3)
        if odd and c==size+1:b=(1,2,2,1)
        for k in range(4):
            faces[s//4*4+(s+k)%4]=tuple(b[(side-k)%4] for side in range(4))
        rows[c]=[s,(s//4)*4+(s+2)%4]
    rows[wide].append((board[wide]//4)*4+(board[wide]+1)%4)
    edges=[e for e in grid_edges(size) if (e[0] in ring and e[1] in ring) or (e[0] in dummy and e[1] in dummy)]
    return make_problem(size,faces,domains=rows,edges=edges),ring,dummy,wide


def masks(problem):
    return tuple(sum(1<<s for s in set(d)) for d in problem.domains)


def scalar_ring(problem, ring):
    cells=sorted(ring)
    edges=[(cells.index(a),cells.index(b),x,y) for a,b,x,y in problem.edges if a in ring and b in ring]
    return [v for v in product(*(problem.domains[c] for c in cells))
            if len({s//4 for s in v})==len(v) and
            all(problem.faces[v[a]][x]==problem.faces[v[b]][y] for a,b,x,y in edges)]


class PivotGraphTests(unittest.TestCase):
    def test_consistent_and_inconsistent_eight_cycles_and_unrelated_pivot(self):
        for odd in (True,False):
            p,ring,dummy,wide=ring_problem(odd)
            d=masks(p)
            witnesses=scalar_ring(p,ring)
            self.assertEqual(len(witnesses),0 if odd else 2)
            for target in ring|dummy:
                ans=prove_pivot(p,d,target,edges=p.edges)
                self.assertEqual(ans['work'],sum(ans['counts'].values()))
                self.assertEqual(ans['outcome'],'contradiction' if target in ring and odd else 'unchanged')
                if ans['outcome']=='contradiction':
                    self.assertFalse(witnesses)
            for fixed in (0,wide,-1,len(d)):
                self.assertEqual(prove_pivot(p,d,fixed,edges=p.edges)['outcome'],'ineligible')
            self.assertEqual(prove_pivot(p,d,min(ring),edges=())['outcome'],'unchanged')

    def test_all_restricted_masks_and_edges_accept_no_false_pivot_proofs(self):
        p, board=generated_problem(2,3,41)
        # Wide fourth cell. Each of three other cells chooses one of two
        # physical pieces; enumerate all of their nonempty restricted masks.
        candidates=[]
        for c,d in enumerate(p.domains):
            other=next(s for s in d if s//4==board[(c+1)%4]//4)
            candidates.append([board[c],other])
        for edge_bits in range(16):
            edges=[e for i,e in enumerate(grid_edges(2)) if edge_bits&(1<<i)]
            for selector in product((1,2,3),repeat=3):
                rows=[[s for i,s in enumerate(candidates[c]) if selector[c]&(1<<i)] for c in range(3)]+[p.domains[3][:3]]
                task=replace(p,domains=tuple(map(tuple,rows)),edges=tuple(edges))
                valid=[v for v in product(*rows) if check_solution(task,v)]
                d=masks(task)
                for pivot in range(3):
                    before=tuple(d)
                    ans=prove_pivot(task,d,pivot,edges=edges)
                    self.assertEqual(before,tuple(d))
                    if ans['outcome']=='contradiction':
                        self.assertFalse(valid)

    def test_one_impossible_orientation_cannot_eliminate_two(self):
        p, ring, dummy, wide=ring_problem(False)
        ring=sorted(ring)
        # Keep three cells on two required edges in a line. Couple the outer
        # alternatives by giving them a common physical piece. Only one
        # choice of the middle pivot is impossible, so both-way reach fails.
        a,b,c=9,10,11
        rows=list(p.domains)
        faces=list(p.faces)
        A,B,C,P=(rows[i][0] for i in (a,b,c,wide))
        def setface(state,face):
            for k in range(4):
                faces[state//4*4+(state+k)%4]=tuple(face[(s-k)%4] for s in range(4))
        # B in its first orientation accepts A.left end and C.right end.
        # B half-turned requires the same physical P at both ends.
        for state,face in ((A,(1,1,1,1)),(C,(2,2,2,2)),(P,(2,2,1,1)),(B,(1,2,1,1))):
            setface(state,face)
        rows[a],rows[b],rows[c]=(A,P),(B,B//4*4+(B+2)%4),(C,P)
        task=replace(p,faces=tuple(faces),domains=tuple(rows),edges=((a,b,1,3),(b,c,1,3)))
        local=scalar_ring(task,{a,b,c})
        self.assertEqual({v[1] for v in local},{B})
        for pivot in (a,b,c):
            self.assertEqual(prove_pivot(task,masks(task),pivot,edges=task.edges)['outcome'],'unchanged')

    def test_work_caps_and_last_poll_never_publish_a_partial_proof(self):
        p,ring,_,_=ring_problem()
        d=masks(p); c=min(ring)
        calls=[]
        complete=prove_pivot(p,d,c,edges=p.edges,check=lambda:calls.append(True))
        self.assertEqual(complete['outcome'],'contradiction')
        for cap in (0,1,50,complete['work']//2,complete['work']-1):
            res=prove_pivot(p,d,c,edges=p.edges,max_work=cap)
            self.assertEqual(res['outcome'],'work_limit')
            self.assertLessEqual(res['work'],cap)
        self.assertEqual(prove_pivot(p,d,c,edges=p.edges,max_work=complete['work'])['outcome'],'contradiction')
        for count in (1,2,len(calls)//2,len(calls)):
            remaining=count
            def interrupt():
                nonlocal remaining
                remaining-=1
                if remaining==0:raise dfs._Interrupted('stopped')
            with self.assertRaises(dfs._Interrupted):
                prove_pivot(p,d,c,edges=p.edges,check=interrupt)



class ProfileTests(unittest.TestCase):
    def test_selection_is_explicit_and_disabled_options_have_legacy_hash(self):
        full=dfs_profile_options('full')
        self.assertEqual(dfs.normalize_options(dict(full,pivot_cut_v1=False)),dfs.normalize_options(full))
        enabled=dfs.normalize_options(dfs_profile_options('full-pivot'))
        self.assertEqual(enabled,dict(dfs.normalize_options(full),pivot_cut_v1=True))
        self.assertEqual(dfs.normalize_options(enabled),enabled)
        for value in (0,None,'true'):
            with self.assertRaises(ValueError):dfs.normalize_options(dict(full,pivot_cut_v1=value))
        with self.assertRaisesRegex(ValueError,'Maximum filtering'):
            dfs.normalize_options(dict(pivot_cut_v1=True))
        for profile in PRUNING_PROFILES:
            self.assertEqual(dfs.normalize_options(dfs_profile_options(profile)).get('pivot_cut_v1',False),
                             profile=='full-pivot')

    def test_dashboard_launcher_worker_and_unsupported_modes(self):
        command=['--method','exact','--exact-engine','dfs','--pruning','full-pivot']
        for entry in ('start','run'):
            parsed=launcher.parser().parse_args([entry]+command)
            self.assertIn('full-pivot',launcher.run_arguments(parsed))
        chosen=dict(method='exact',exact_engine='dfs',pruning='full-pivot',workers=1)
        settings=dashboard_server.validate_settings(chosen)
        self.assertIn('full-pivot',dashboard_server.worker_command(Path.cwd(),8765,settings))
        for engine in ('partitioned','hybrid','impact-dfs','cp-sat'):
            with self.assertRaisesRegex(ValueError,'standard DFS'):
                exact_worker._settings(engine,10,0,1,8765,pruning='full-pivot')
            with self.assertRaisesRegex(ValueError,'standard DFS'):
                dashboard_server.validate_settings(dict(chosen,exact_engine=engine))
        with self.assertRaises(ValueError):
            dashboard_server.validate_settings(dict(chosen,method='gpu'))
        p,board=generated_problem(2,3,6)
        enabled=dfs_profile_options('full-pivot')
        with self.assertRaisesRegex(ValueError,'without ordering hints'):
            dfs.solve(p,seconds=0,options=enabled,hints=[board])
        with self.assertRaisesRegex(ValueError,'standard DFS'):
            impact_dfs.solve(p,seconds=10,options=enabled)
        with self.assertRaisesRegex(ValueError,'partitioned'):
            exact_parallel.solve(p,seconds=0,workers=1,options=enabled)
        with self.assertRaises(ValueError):
            dfs.solve(p,seconds=0,workers=2,options=enabled)

class PivotFrontierTests(unittest.TestCase):
    def setUp(self):
        self.problem,self.ring,_,_=ring_problem(True)

    def search(self,profile,stop=None,checkpoint=None):
        return dfs._Search(self.problem,10,4666,stop,None,checkpoint,None,1,None,dfs_profile_options(profile))

    def pending(self,profile):
        solver=self.search(profile)
        original=solver.choose_partition
        def choose(domains):
            kind,cell,options=original(domains)
            if kind=='cell' and cell in self.ring:
                self.assertGreater(len(solver.stack)-1,8)
                raise dfs._Interrupted('stopped')
            return kind,cell,options
        with patch.object(solver,'choose_partition',choose),self.assertRaises(dfs._Interrupted):
            solver.visit(solver.build())
        return solver.checkpoint()

    def through_pivot(self,profile,checkpoint,first_child_only=False):
        depth=len(checkpoint['stack'])-1
        observing=checkpoint['stack'][-1]['phase']=='branch'
        calls=[]
        def stop():
            if s.restoring or not observing or not s.stack:return False
            last=s.stack[-1]
            if first_child_only:
                return len(s.stack)==depth+1 and last['phase']=='branch' and last['next']==1
            return len(s.stack)<depth+1
        s=self.search(profile,stop,checkpoint)
        original=s.choose_partition
        def selected(domains):
            nonlocal observing
            part=original(domains)
            self.assertEqual(part[0],'cell')
            self.assertIn(part[1],self.ring)
            observing=True
            return part
        def record(*args,**kwargs):
            res=prove_pivot(*args,**kwargs)
            calls.append(res['outcome'])
            return res
        with patch.object(s,'choose_partition',selected),patch.object(dfs,'prove_pivot',record),self.assertRaises(dfs._Interrupted):
            s.visit(s.build())
        return s.checkpoint(),calls

    def test_release_full_checkpoint_and_enabled_profile_each_resume_their_own_frontier(self):
        base=self.pending('full')
        # SHA-256 of the same generated pending checkpoint written by v0.5.3.
        self.assertEqual(base['checksum'],'a59c2e839554b4e9ef720563cd498cebffe2c863b59835d7244ee39a2e034448')
        self.assertEqual(base['options_hash'],'739ff8cab675a666324a2fb3b53a8ece0620d331261710e311880d453f4e858a')
        enabled=self.pending('full-pivot')
        for profile,wrong in (('full',enabled),('full-pivot',base)):
            with self.assertRaises(ValueError):
                self.search(profile,checkpoint=wrong)
        half,calls=self.through_pivot('full',base,True)
        self.assertEqual(calls,[])
        self.assertEqual(half['checksum'],'6a1c5f977e7782d821b9460a28e61a06bc979429192b485461457e81a11b6eaa')
        self.assertEqual(half['stack'][-1]['next'],1)
        finish_old,old_calls=self.through_pivot('full',half)
        finish_new,new_calls=self.through_pivot('full-pivot',enabled)
        self.assertEqual(old_calls,[])
        self.assertEqual(new_calls,['contradiction'])
        self.assertEqual(finish_old['stack'],finish_new['stack'])
        self.assertEqual(finish_old['rng_state'],finish_new['rng_state'])
        self.assertEqual(finish_old['counters']['branches']-finish_new['counters']['branches'],2)
        self.assertEqual(finish_old['counters']['probe_checks'],finish_new['counters']['probe_checks'])

    def test_cutoff_in_proof_rejoins_the_pending_choice_and_fresh_sat_witness(self):
        enabled=self.pending('full-pivot')
        solver=self.search('full-pivot',checkpoint=enabled)
        def interrupted(problem,domains,target,**kwargs):
            self.assertNotEqual(solver.committed_rng,solver.rng.getstate())
            checks=0
            def stop():
                nonlocal checks
                checks+=1
                if checks==2:raise dfs._Interrupted('stopped')
            return prove_pivot(problem,domains,target,**dict(kwargs,check=stop))
        with patch.object(dfs,'prove_pivot',interrupted),self.assertRaises(dfs._Interrupted):
            solver.visit(solver.build())
        stopped=solver.checkpoint()
        self.assertEqual(stopped['stack'],enabled['stack'])
        self.assertEqual(stopped['rng_state'],enabled['rng_state'])
        self.assertEqual(stopped['counters']['branches'],enabled['counters']['branches'])
        resumed,calls=self.through_pivot('full-pivot',stopped)
        uninterrupted,expected=self.through_pivot('full-pivot',enabled)
        self.assertEqual(calls,expected)
        self.assertEqual(resumed['stack'],uninterrupted['stack'])
        self.assertEqual(resumed['rng_state'],uninterrupted['rng_state'])
        self.assertEqual(resumed['counters']['branches'],uninterrupted['counters']['branches'])
        good,_,_,_=ring_problem(False)
        base=dfs.solve(good,seconds=10,seed=4666,options=dfs_profile_options('full'))
        candidate=dfs.solve(good,seconds=10,seed=4666,options=dfs_profile_options('full-pivot'))
        self.assertEqual(base['board'],candidate['board'])
        self.assertTrue(check_solution(good,candidate['board']))


if __name__=='__main__':
    unittest.main()
