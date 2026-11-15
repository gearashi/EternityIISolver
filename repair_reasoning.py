"""Optional regional reasoning adapters for the matching portfolio.

Each run owns its caches. A score target failure never proves full-puzzle
impossibility. Native and explicit-search work remain separate counters.
"""
import math
import itertools
import time
from repair_regions import RegionRepairSession
from repair_exact_regions import RegionFailureCache, solve_region
from repair_certificates import GeneralizedFailureCache


class ReasoningRegionSession:
    def __init__(self, problem, mode):
        if mode not in ('bounded','learned','partition-bounds','price-bounds'):raise ValueError('Invalid regional reasoning mode')
        self.mode=mode;self.closed=False;self.seed=None
        self.exact_cache=RegionFailureCache();self.fallback=RegionRepairSession(problem)
        self.problem=self.fallback.problem
        self.learning=GeneralizedFailureCache(self.problem) if mode=='learned' else None
        self.counters=dict(calls=0,candidate_placements=0,bound_work=0,bound_setup_work=0,
                           native_queries=0,native_branches=0,native_conflicts=0,native_propagations=0,
                           proved_targets=0,unknown_queries=0,price_simplex_iterations=0,price_fixings=0,encoding_clauses=0,model_builds=0,
                           build_seconds=0.,lookup_literal_checks=0,minimization_queries=0,table_rows=0)

    @property
    def cache_hits(self):
        return self.exact_cache.hits+self.fallback.cache_hits+(self.learning.cache_hits if self.learning else 0)

    def close(self):
        self.closed=True
        if self.learning is not None:self.learning.close()

    def stats(self):return {'region_reasoning_'+key:value for key,value in self.counters.items()}

    def solve(self, board, cells, *, seconds=.25, seed=0, should_stop=None):
        if self.closed:raise RuntimeError('Regional reasoning session is closed')
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid region seconds')
        if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid region seed')
        if should_stop is not None and not callable(should_stop):raise ValueError('Invalid stop callback')
        # Stable first seed retains the same native model across regional calls.
        if self.seed is None:self.seed=seed
        try: chosen=list(itertools.islice(iter(cells),65))
        except TypeError as error: raise ValueError('Invalid region cells') from error
        if (len(chosen)>64 or any(type(c) is not int or not 0<=c<self.problem.size**2 for c in chosen)
                or len(set(chosen))!=len(chosen) or set(chosen)&self.problem.fixed.keys()):
            raise ValueError('Region requires at most 64 distinct mutable cells')
        self.counters['calls']+=1
        if self.learning is not None:
            r=self.learning.solve(board,chosen,seconds=seconds,seed=self.seed,should_stop=should_stop)
            for source,target in (('native_queries','native_queries'),('branches','native_branches'),
                                  ('conflicts','native_conflicts'),('propagations','native_propagations')):
                self.counters[target]+=r.get(source,0)
            for key in ('encoding_clauses','model_builds','build_seconds','lookup_literal_checks','minimization_queries'):
                self.counters[key]+=r.get(key,0)
        elif len(chosen)>24:
            r=self.fallback.solve(board,chosen,seconds=seconds,seed=seed,should_stop=should_stop)
            self.counters['native_queries']+=int(not r.get('cache_hit',False) and r.get('solver_status','NOT_RUN')!='NOT_RUN')
            self.counters['table_rows']+=r.get('table_rows',0)
            self.counters['build_seconds']+=r.get('build_seconds',0)
            self.counters['native_branches']+=r.get('branches',0);self.counters['native_conflicts']+=r.get('conflicts',0)
        else:
            deadline=time.monotonic()+seconds
            def stopped():return bool((should_stop and should_stop()) or time.monotonic()>=deadline)
            r=solve_region(self.problem,board,chosen,max_work=50000,algorithm='dfs',ordering='mrv',
                           assignment_bound=self.mode in ('bounded', 'price-bounds'),
                           clique_bound=self.mode in ('bounded', 'price-bounds'),
                           cost_partition_bound=self.mode=='partition-bounds',certified_prices=self.mode=='price-bounds',cache=self.exact_cache,should_stop=stopped)
            self.counters['candidate_placements']+=r['candidate_placements']
            self.counters['price_simplex_iterations']+=r.get('price_simplex_iterations',0)
            self.counters['price_fixings']+=r.get('price_fixings',0)
            self.counters['bound_work']+=r['search_work']-r['candidate_placements']
            self.counters['bound_setup_work']+=r.get('partition_setup_work',0)+sum(v for k,v in r.items() if k.startswith('bounds_setup_') and type(v) is int)
        proven=r.get('proven_unreachable',False) or r.get('termination')=='no_improvement_in_region'
        self.counters['proved_targets']+=int(proven)
        self.counters['unknown_queries']+=int(not proven and not r.get('improved',False) and not r.get('target_reached',False) and r.get('termination') not in ('already_perfect','perfect_board_validated','target_already_reached'))
        # Explicit caps/timeouts are diagnostic only; never add a proof here.
        return r
