"""Bounded positive patch witnesses, rescored under the current complement."""
from collections import OrderedDict
from dataclasses import replace
from types import MappingProxyType

from exact_model import constraint_edges, frame_ok, grid_edges
from exact_optimize import evaluate_board
from repair_multisource import _cells


class PatchWitnesses:
    def __init__(self,problem,capacity=256):
        if type(capacity) is not int or not 1<=capacity<=4096:
            raise ValueError('capacity must be 1..4096')
        if set(constraint_edges(problem))!=set(grid_edges(problem.size)):
            raise ValueError('full-grid objective required')
        self.problem=replace(problem,faces=tuple(tuple(f) for f in problem.faces),
                             domains=tuple(tuple(d) for d in problem.domains),
                             fixed=MappingProxyType(dict(problem.fixed)),edges=None)
        self.capacity=capacity
        self.rows=OrderedDict()

    def remember(self,board,cells):
        checked=evaluate_board(self.problem,board)
        cells=tuple(_cells(self.problem,cells))
        if not checked['valid']:
            raise ValueError('witness must be a verified legal board')
        states=tuple(board[c] for c in cells)
        pool=tuple(sorted(s//4 for s in states))
        key=(cells,pool,states)
        self.rows[key]=None
        self.rows.move_to_end(key)
        if len(self.rows)>self.capacity:
            self.rows.popitem(last=False)

    def lookup(self,board,cells,*,should_stop=None):
        base=evaluate_board(self.problem,board)
        cells=tuple(_cells(self.problem,cells))
        if not base['valid']:
            raise ValueError('invalid lookup parent')
        if should_stop is not None and not callable(should_stop):
            raise ValueError('invalid should_stop')
        pool=tuple(sorted(board[c]//4 for c in cells))
        probes=visits=0
        for key in reversed(self.rows):
            if should_stop and should_stop():
                break
            visits+=1
            shape,required,states=key
            if shape!=cells or required!=pool:
                continue
            probes+=1
            candidate=list(board)
            for c,s in zip(cells,states):
                candidate[c]=s
            observed=evaluate_board(self.problem,candidate)
            if not observed['valid']:
                raise RuntimeError('scoped positive patch witness illegally combined')
            if observed['score']>base['score']:
                return dict(board=candidate,score=observed['score'],hit=True,probes=probes,index_visits=visits)
        return dict(board=list(board),score=base['score'],hit=False,probes=probes,index_visits=visits)

    def export_state(self):
        return [[list(cells),list(states)] for cells,_,states in self.rows]

    def restore_state(self,entries):
        if type(entries) is not list or len(entries)>self.capacity:
            raise ValueError('invalid witness cache size')
        fresh=OrderedDict()
        for row in entries:
            if type(row) is not list or len(row)!=2:
                raise ValueError('invalid witness cache row')
            cells,states=row
            if type(cells) is not list or list(_cells(self.problem,cells))!=cells or type(states) is not list or len(states)!=len(cells):
                raise ValueError('invalid witness scope or states')
            if any(type(s) is not int or s not in self.problem.domains[c] or
                   not frame_ok(self.problem.size,c,self.problem.faces[s]) for c,s in zip(cells,states)):
                raise ValueError('witness state outside legal cell domain')
            pool=tuple(sorted(s//4 for s in states))
            if len(set(pool))!=len(cells):
                raise ValueError('duplicate physical piece in witness')
            key=tuple(cells),pool,tuple(states)
            if key in fresh:
                raise ValueError('duplicate saved witness')
            fresh[key]=None
        self.rows=fresh


class WitnessRegionSession:
    def __init__(self,problem,capacity=32):
        from repair_regions import RegionRepairSession
        self.base=RegionRepairSession(problem)
        self.witnesses=PatchWitnesses(self.base.problem,capacity)
        self.positive_hits=0
        self.lookup_count=0
        self.lookup_validations=0
        self.index_visits=0

    @property
    def cache_hits(self):
        return self.base.cache_hits

    def solve(self,board,cells,seconds=.5,seed=0,should_stop=None):
        import math
        import time
        from repair_regions import _validate
        started=time.monotonic()
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
            raise ValueError('invalid region time budget')
        p=self.base.problem
        base=_validate(p,board,seed,should_stop)
        selected=_cells(p,cells)
        def ended():
            return (should_stop is not None and should_stop()) or time.monotonic()-started>=seconds
        lookup=self.witnesses.lookup(board,selected,should_stop=ended)
        self.lookup_count+=1
        self.lookup_validations+=lookup['probes']
        self.index_visits+=lookup['index_visits']
        if lookup['hit'] and not ended():
            self.positive_hits+=1
            return dict(board=lookup['board'],score=lookup['score'],reference_score=base['score'],
                        max_score=base['max_score'],target_score=min(base['score']+1,base['max_score']),
                        improved=True,outcome='solved' if lookup['score']==base['max_score'] else 'improved',
                        termination='positive_witness',optimal=lookup['score']==base['max_score'],
                        proof_scope='verified-positive-witness-only',backend='witness-cache',
                        workers=1,branches=0,conflicts=0,table_rows=0,region_cells=selected,
                        witness_checks=lookup['probes'],witness_index_visits=lookup['index_visits'])
        answer=self.base.solve(board,selected,seconds=max(0.,seconds-(time.monotonic()-started)),
                               seed=seed,should_stop=should_stop)
        if answer['improved']:
            self.witnesses.remember(answer['board'],selected)
        return dict(answer,witness_checks=lookup['probes'],witness_index_visits=lookup['index_visits'])

    def stats(self):
        return dict(positive_patch_hits=self.positive_hits,patch_queries=self.lookup_count,
                    patch_validations=self.lookup_validations,patch_index_visits=self.index_visits)

    def export_state(self):
        return dict(witnesses=self.witnesses.export_state(),stats=self.stats())

    def restore_state(self,state):
        if type(state) is not dict or set(state)!={'witnesses','stats'}:
            raise ValueError('invalid saved patch state')
        stats=state['stats']
        if (type(stats) is not dict or set(stats)!=set(self.stats()) or
                any(type(n) is not int or not 0<=n<2**53 for n in stats.values()) or
                not stats['positive_patch_hits']<=stats['patch_queries'] or
                not stats['positive_patch_hits']<=stats['patch_validations']<=stats['patch_index_visits']):
            raise ValueError('invalid positive patch counters')
        self.witnesses.restore_state(state['witnesses'])
        self.positive_hits=stats['positive_patch_hits']
        self.lookup_count=stats['patch_queries']
        self.lookup_validations=stats['patch_validations']
        self.index_visits=stats['patch_index_visits']
