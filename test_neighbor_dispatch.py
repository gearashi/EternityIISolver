"""Focused proof, dispatch ownership, interruption, and profile checks on the B009 overlay."""
from pathlib import Path
import itertools
import unittest
from unittest.mock import patch

import exact_dfs
import exact_parallel
from dfs_profiles import dfs_profile_options
from exact_model import generated_problem, grid_edges, make_problem, check_solution
from neighbor_cut import build_neighbor_graph, query_neighbor_graph

OLD = dfs_profile_options('full-dispatch')
OPTIONS = dfs_profile_options('full-neighbor')


def fixture():
    p, board = generated_problem(8,3,91)
    c, left, right, d, e = 27,26,28,45,46
    p0,q0,a,b,z = (board[i] for i in (c,left,right,d,e))
    cores = {c,left,right,d,e}
    dummy = sorted(i for i in range(64) if 0<i//8<7 and 0<i%8<7 and i not in cores)[:9]
    rows=[[s] for s in board]
    faces=list(p.faces)
    for state,color in ((p0,1),(q0,1),(a,2),(b,2),(z,2)):
        for rot in range(4):
            faces[4*(state//4)+rot]=(color,)*4
    for i in dummy:
        v=board[i]
        rows[i]=[v,v//4*4+(v+1)%4]
    rows[c]=[p0,a,b,z]
    rows[left]=rows[right]=[q0,a,b,z]
    rows[d]=rows[e]=[p0,q0,a,b,z]
    edges=[t for t in grid_edges(8) if t[:2] in ((left,c),(c,right))]
    return make_problem(8,faces,domains=rows,edges=edges),dummy,p0,(c,d,e)


def sequence(dummy,p0,dest,kind,good_first=False):
    original=exact_dfs._Search.choose_partition
    def choose(search,domains):
        # Use only legal release partitions. Pick the nine dummy binary branches
        # first so all generated cases reach the declared depth cheaply.
        for i in dummy:
            if domains[i].bit_count()>1:
                choices=search.partition_choices(domains,'cell',i)
                search.rng.shuffle(choices)
                return 'cell',i,choices
        c,d,e=dest
        target=p0//4 if kind=='piece' else c
        choices=search.partition_choices(domains,kind,target)
        expected=(3 if kind=='piece' else 4)
        if len(choices)==expected and domains[c]&(1<<p0):
            search.rng.shuffle(choices)
            first=(d if good_first else c) if kind=='piece' else (next(x for x in choices if x!=p0) if good_first else p0)
            choices.remove(first);choices.insert(0,first)
            return kind,target,choices
        return original(search,domains)
    return choose


class ProofTests(unittest.TestCase):
    def test_exhaustive_two_cell_edge(self):
        faces = [(1,1,1,1),(2,2,2,2)] * 4
        rows = [(0,1),(0,4),(0,4,5),(4,5,6)]
        colors = [{value:sum(1<<s for s,face in enumerate(faces) if face[side]==value) for value in (1,2)} for side in range(4)]
        for left,right in itertools.product(rows,repeat=2):
            masks = [sum(1<<v for v in cell) for cell in (left,right)]
            g = build_neighbor_graph(faces,colors,[[(1,1,3)],[(0,3,1)]],masks,lambda:None)
            for state in left:
                expected = not any(state//4!=other//4 and faces[state][1]==faces[other][3] for other in right)
                self.assertEqual(query_neighbor_graph(g,0,[state],lambda:None),expected)

    def test_partial_proofs_cutoffs_and_interruption(self):
        faces=[(1,1,1,1)]*5
        faces[0]=(1,2,1,1)
        colors=[{value:sum(1<<s for s,face in enumerate(faces) if face[side]==value) for value in (1,2)} for side in range(4)]
        geo=colors,[[(1,1,3)],[(0,3,1)]]
        g=build_neighbor_graph(faces,*geo,[3,16],lambda:None)
        audit={}
        self.assertFalse(query_neighbor_graph(g,0,[0,1],lambda:None,audit=audit))
        self.assertEqual([r['outcome'] for r in audit['roots']],['proved','closure'])
        for cap in ({'checks':0},{'literals':0},{'steps':0}):
            self.assertFalse(query_neighbor_graph(g,0,[0],lambda:None,**cap))
        def stopped():raise exact_dfs._Interrupted('stopped')
        with self.assertRaises(exact_dfs._Interrupted):query_neighbor_graph(g,0,[0],stopped)
        with self.assertRaises(exact_dfs._Interrupted):build_neighbor_graph(faces,*geo,[3,16],stopped)


class DispatchTests(unittest.TestCase):
    def test_profile_packaging_and_worker_command(self):
        import ast
        import tomllib
        import dashboard_server
        import exact_worker
        import launcher
        self.assertEqual(OPTIONS,dict(OLD,neighbor_dispatch_v1=True))
        root=Path(__file__).resolve().parent
        modules=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['tool']['setuptools']['py-modules']
        self.assertIn('neighbor_cut',modules)
        spec=ast.parse((root/'EternityIISolver.spec').read_text(encoding='utf-8'))
        native=next(n.value for n in spec.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='hiddenimports' for v in n.targets))
        self.assertIn('neighbor_cut',ast.literal_eval(native))
        args=launcher.parser().parse_args(['start','--method','exact','--pruning','full-neighbor','--no-browser'])
        self.assertEqual(args.pruning,'full-neighbor')
        settings=dashboard_server.validate_settings({'method':'exact','exact_engine':'dfs','workers':1,'pruning':args.pruning})
        cmd=dashboard_server.worker_command(root,8765,settings)
        self.assertEqual(cmd[cmd.index('--pruning')+1],args.pruning)
        for engine in ('hybrid','partitioned','impact-dfs','cp-sat'):
            with self.assertRaises(ValueError):exact_worker._settings(engine,1,3,1,8765,pruning=args.pruning)
        with self.assertRaises(ValueError):dashboard_server.validate_settings(dict(settings,workers=2))

    def test_worker_checkpoint_identity_and_resume(self):
        import json
        import tempfile
        import exact_worker
        with tempfile.TemporaryDirectory(prefix='eternity-neighbor-worker-') as temp:
            home=Path(temp)
            for resume in (False,True):
                self.assertEqual(exact_worker.run_task(home,engine='dfs',workers=1,seconds=0,pruning='full-neighbor'),0)
                status=json.loads((home/'runtime/status.json').read_text())
                checkpoint=json.loads((home/'runtime/exact-dfs-checkpoint.json').read_text())
                self.assertIs(status['resumed'],resume)
                self.assertEqual(status['pruning'],'full-neighbor')
                self.assertNotEqual(checkpoint['options_hash'],'96b7d389565f2263d159565226b9c21261b594b9e89f4139164dfe84980f20c1')
                if resume:self.assertEqual(checkpoint['options_hash'],previous)
                previous=checkpoint['options_hash']

    def test_profile_version_and_restrictions(self):
        puzzle,dummy,p0,dest=fixture()
        old=exact_dfs.solve(puzzle,seconds=0,options=OLD)['checkpoint']
        new=exact_dfs.solve(puzzle,seconds=0,options=OPTIONS)['checkpoint']
        self.assertEqual(old['options_hash'],'96b7d389565f2263d159565226b9c21261b594b9e89f4139164dfe84980f20c1')
        self.assertNotEqual(old['options_hash'],new['options_hash'])
        for op,cp in ((OLD,new),(OPTIONS,old)):
            with self.assertRaises(ValueError):exact_dfs.solve(puzzle,seconds=0,options=op,checkpoint=cp)
        self.assertEqual(exact_dfs.normalize_options(dict(OLD,neighbor_dispatch_v1=False)),exact_dfs.normalize_options(OLD))
        for bad in ({'neighbor_dispatch_v1':True},dict(OLD,neighbor_dispatch_v1=1)):
            with self.assertRaises(ValueError):exact_dfs.normalize_options(bad)
        with self.assertRaises(ValueError):exact_parallel.solve(puzzle,seconds=0,options=OPTIONS)

    def test_skip_only_actual_cell_and_positive_sibling(self):
        problem,dummy,p0,dest=fixture()
        original=exact_dfs._Search.skip_neighbor_choice
        for kind in ('piece','cell'):
            for good in (False,True):
                records=[]
                def observe(s,f,choice):
                    records.append((f['kind'],f['cell'],choice,f['next']))
                    return original(s,f,choice)
                with patch.object(exact_dfs._Search,'choose_partition',sequence(dummy,p0,dest,kind,good)):
                    before=exact_dfs.solve(problem,seconds=10,seed=11,options=OLD)
                    with patch.object(exact_dfs._Search,'skip_neighbor_choice',observe):
                        after=exact_dfs.solve(problem,seconds=10,seed=11,options=OPTIONS)
                self.assertEqual(after['outcome'],'solved')
                self.assertTrue(check_solution(problem,after['board']))
                self.assertEqual(before['board'],after['board'])
                self.assertEqual(before['probe_checks'],after['probe_checks'])
                self.assertEqual(before['branches']-after['branches'],len(records))
                expected=(kind,p0//4,dest[0],0) if kind=='piece' else (kind,dest[0],p0,0)
                self.assertEqual(records, [] if good else [expected])
                if kind=='piece':self.assertIn(after['board'].index(p0),dest[1:])

    def test_stop_and_restore_actual_remaining_branch_frames(self):
        puzzle,dummy,p0,dest=fixture()
        real_build=exact_dfs.build_neighbor_graph
        real_query=exact_dfs.query_neighbor_graph
        real_skip=exact_dfs._Search.skip_neighbor_choice
        with patch.object(exact_dfs._Search,'choose_partition',sequence(dummy,p0,dest,'piece')):
            fresh=exact_dfs.solve(puzzle,seconds=10,seed=11,options=OPTIONS)
            pre_cp=None
            for phase in ('build','query','after_skip'):
                state={'stop':False}
                def stop_build(*a,**kw):
                    if phase=='build':raise exact_dfs._Interrupted('stopped')
                    return real_build(*a,**kw)
                def stop_query(g,c,legal,check,**kw):
                    if phase=='query':
                        calls=0
                        def interrupt():
                            nonlocal calls
                            calls+=1
                            if calls==2:raise exact_dfs._Interrupted('stopped')
                            return check()
                        return real_query(g,c,legal,interrupt,**kw)
                    return real_query(g,c,legal,check,**kw)
                def stop_skip(s,frame,chosen):
                    real_skip(s,frame,chosen)
                    if phase=='after_skip':state['stop']=True
                with patch.object(exact_dfs,'build_neighbor_graph',stop_build),patch.object(exact_dfs,'query_neighbor_graph',stop_query),patch.object(exact_dfs._Search,'skip_neighbor_choice',stop_skip):
                    r=exact_dfs.solve(puzzle,seconds=10,seed=11,options=OPTIONS,should_stop=lambda:state['stop'])
                self.assertEqual(r['outcome'],'stopped')
                cp=r['checkpoint'];last=cp['stack'][-1]
                self.assertEqual((last['kind'],last['cell'],last['next']),('piece',p0//4,int(phase=='after_skip')))
                self.assertEqual(set(last['candidates']),set(dest))
                self.assertNotIn('neighbor_graph',last)
                if pre_cp:
                    self.assertEqual(cp['rng_state'],pre_cp['rng_state'])
                    self.assertEqual(cp['counters'],pre_cp['counters'])
                    self.assertEqual(last['candidates'],pre_cp['stack'][-1]['candidates'])
                else:pre_cp=cp
                # A deadline while reconstructing an ancestor retains the
                # original saved suffix. No wide graph belongs to that replay.
                def stop_restore(search,*args,**kwargs):
                    self.assertTrue(search.restoring)
                    raise exact_dfs._Interrupted('stopped')
                with patch.object(exact_dfs._Search,'prepare',stop_restore),patch.object(exact_dfs,'build_neighbor_graph',side_effect=AssertionError('ancestor graph')):
                    early=exact_dfs.solve(puzzle,seconds=10,seed=11,options=OPTIONS,checkpoint=cp)
                self.assertEqual(early['checkpoint'],cp)
                paths=exact_parallel._frontier_paths(cp)
                owned=[path[-1][2] for path in paths if path and path[-1][:2]==['piece',p0//4]]
                self.assertEqual(owned,last['candidates'][last['next']:])
                resumed_queries=[]
                def watch(g,c,legal,check,**kw):
                    resumed_queries.append((c,list(legal)))
                    return real_query(g,c,legal,check,**kw)
                with patch.object(exact_dfs,'query_neighbor_graph',watch):
                    resumed=exact_dfs.solve(puzzle,seconds=10,seed=11,options=OPTIONS,checkpoint=cp)
                self.assertEqual((resumed['outcome'],resumed['board'],resumed['branches'],resumed['probe_checks']),
                                 (fresh['outcome'],fresh['board'],fresh['branches'],fresh['probe_checks']))
                self.assertEqual(resumed_queries[0],(last['candidates'][last['next']],[p0]))
                if phase=='after_skip':self.assertNotIn((dest[0],[p0]),resumed_queries)


if __name__=='__main__':unittest.main(verbosity=2)
