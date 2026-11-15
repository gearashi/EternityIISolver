"""Bounded exact regional search and reusable, narrowly scoped failure proofs.

Search work counts placements, bound tuples and counted assignment/graph operations; never BOINC nodes or GPU moves. A
work/memory interruption returns unknown, never a negative puzzle conclusion.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import replace
from fractions import Fraction
import hashlib
import itertools
import json
import math

from exact_model import constraint_edges, frame_ok, grid_edges, initial_domains, problem_hash
from exact_optimize import evaluate_board

MAX_CELLS = 24
SCOPE = 'selected-cells-and-piece-pool-with-frozen-complement-only'


def _integer(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value


class _Limit(Exception):
    pass


class _Witness(Exception):
    pass


class RegionFailureCache:
    """Bounded in-memory proofs produced only by completed regional searches.

    No persistence/import API accepts untrusted claims of exhaustion. Keys bind
    the full problem, physical pool, frozen complement, selected cells and target.
    """
    def __init__(self, capacity=256):
        self.capacity = _integer('capacity', capacity, 1, 4096)
        self._entries = OrderedDict()
        self.hits = 0

    def __len__(self):
        return len(self._entries)

    def _get(self, key):
        item = self._entries.get(key)
        if item is not None:
            self._entries.move_to_end(key)
            self.hits += 1
            return list(item)
        return None

    def _record_completed(self, key, board):
        self._entries[key] = tuple(board)
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)


def _region(problem, board, cells):
    if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
        raise ValueError('Regional score search requires the full grid')
    checked = evaluate_board(problem, board)
    if not checked['valid']:
        raise ValueError('Illegal regional reference')
    chosen = list(itertools.islice(iter(cells), MAX_CELLS + 1))
    if (len(chosen) > MAX_CELLS or any(type(c) is not int or not 0 <= c < len(board) for c in chosen)
            or len(set(chosen)) != len(chosen) or set(chosen) & problem.fixed.keys()):
        raise ValueError('Region must contain at most 24 distinct mutable cells')
    chosen.sort()
    pool = sorted(board[c] // 4 for c in chosen)
    domains = {c: tuple(s for s in dict.fromkeys(problem.domains[c])
                        if s // 4 in pool and frame_ok(problem.size, c, problem.faces[s])) for c in chosen}
    return list(board), chosen, pool, domains, checked


def region_signature(problem, board, cells, target_score):
    reference, chosen, pool, _, checked = _region(problem, board, cells)
    _integer('target_score', target_score, 0, checked['max_score'] + 1)
    selected = set(chosen)
    data = [problem_hash(problem), chosen, pool,
            [(c, s) for c, s in enumerate(reference) if c not in selected], target_score]
    return hashlib.sha256(json.dumps(data, separators=(',', ':')).encode()).hexdigest()


def solve_region(problem, board, cells, *, target_score=None, max_work=50000,
                 max_states=20000, algorithm='frontier', ordering='mrv', cache=None,
                 should_stop=None, patch_bounds=False, assignment_bound=False, clique_bound=False, cost_partition_bound=False, certified_prices=False):
    """Find a legal target board or prove only this frozen regional target absent.

    Frontier DP merges states with identical used-piece masks and live boundary
    placements, retaining the highest score. DFS is a bounded comparison route;
    failure ordering changes branch order only. Search work counts attempted
    placements plus tuples checked by optional 2x3 bounds; both are also reported
    separately. Optional assignment/clique reasoning work is included in the
    same cap, with detailed counters. Static model setup is reported separately.
    """
    _integer('max_work', max_work, 0, 1000000)
    _integer('max_states', max_states, 1, 100000)
    if algorithm not in ('frontier', 'dfs') or ordering not in ('natural', 'mrv', 'failure'):
        raise ValueError('Unknown regional search algorithm or ordering')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    if cache is not None and not isinstance(cache, RegionFailureCache):
        raise ValueError('cache must be RegionFailureCache')
    for name,value in (('patch_bounds',patch_bounds),('assignment_bound',assignment_bound),('clique_bound',clique_bound),('cost_partition_bound',cost_partition_bound),('certified_prices',certified_prices)):
        if type(value) is not bool: raise ValueError(name+' must be boolean')
    reference, chosen, pool, domains, initial = _region(problem, board, cells)
    maximum = initial['max_score']
    target = min(maximum, initial['score'] + 1) if target_score is None else target_score
    _integer('target_score', target, 0, maximum + 1)
    # Detach all mutable problem constituents before any callback can run.
    problem = replace(problem, fixed=dict(problem.fixed), domains=tuple(tuple(d) for d in problem.domains),
                      faces=tuple(tuple(f) for f in problem.faces))
    selected = set(chosen)
    key = region_signature(problem, reference, chosen, target)
    best, best_score = reference.copy(), initial['score']
    work = placements = bound_assignments = bound_calls = patch_bound_prunes = 0
    expanded = dominated = bound_prunes = peak_states = 0
    cache_hit = False
    extra_counts=dict(price_calls=0,price_work=0,price_simplex_iterations=0,price_fixings=0,price_prunes=0,partition_calls=0,partition_work=0,partition_setup_work=0,partition_prunes=0,partition_skips=0,assignment_calls=0,assignment_prunes=0,assignment_work=0,assignment_matrix_states=0,
                      assignment_relaxations=0,clique_calls=0,clique_prunes=0,clique_work=0,
                      clique_graph_states=0,clique_pair_checks=0,clique_color_steps=0,clique_nodes=0,
                      bound_prefix_edge_checks=0,bound_skips=0,clique_completions=0)
    score_bounds=None
    if assignment_bound or clique_bound:
        from repair_bounds import RegionBounds
        score_bounds=RegionBounds(problem,reference,chosen)
    edges = grid_edges(problem.size)
    internal = [(a, b, sa, sb) for a, b, sa, sb in edges if a in selected and b in selected]
    changing = [(a, b, sa, sb) for a, b, sa, sb in edges if a in selected or b in selected]
    fixed_score = sum(problem.faces[reference[a]][sa] == problem.faces[reference[b]][sb]
                      for a, b, sa, sb in edges if a not in selected and b not in selected)
    neighbors = {c: [] for c in chosen}
    boundary = {c: [] for c in chosen}
    for a, b, sa, sb in changing:
        if a in selected and b in selected:
            neighbors[a].append((b, sa, sb)); neighbors[b].append((a, sb, sa))
        elif a in selected:
            boundary[a].append((sa, problem.faces[reference[b]][sb]))
        else:
            boundary[b].append((sb, problem.faces[reference[a]][sa]))
    unary = {c: {s: sum(problem.faces[s][side] == color for side, color in boundary[c])
                 for s in domains[c]} for c in chosen}
    bits = {piece: 1 << i for i, piece in enumerate(pool)}

    def check():
        if should_stop is not None and should_stop():
            raise _Limit('stopped')

    def consume():
        nonlocal work, placements
        check()
        if work >= max_work:
            raise _Limit('work_limit')
        work += 1; placements += 1

    def admissible_prefix(assignments, mask):
        nonlocal work, bound_assignments, bound_calls, patch_bound_prunes
        if assignment_bound or clique_bound:
            for kind,enabled in (('assignment',assignment_bound),('clique',clique_bound)):
                if not enabled:continue
                check();extra_counts[kind+'_calls']+=1
                method=score_bounds.assignment_upper if kind=='assignment' else score_bounds.clique_upper
                bound=method(assignments,max_work=min(20000,max_work-work),should_stop=should_stop)
                extra_counts[kind+'_work']+=bound['work'];work+=bound['work']
                extra_counts['bound_prefix_edge_checks']+=bound['prefix_edge_checks']
                if kind=='assignment':
                    extra_counts['assignment_matrix_states']+=bound['matrix_state_evaluations']
                    extra_counts['assignment_relaxations']+=bound['matching_relaxations']
                else:
                    extra_counts['clique_graph_states']+=bound['graph_state_evaluations']
                    extra_counts['clique_pair_checks']+=bound['graph_pair_checks']
                    extra_counts['clique_color_steps']+=bound['graph_color_steps']
                    extra_counts['clique_nodes']+=bound['clique_nodes']
                if not bound['complete']:
                    if bound['termination']=='stopped':raise _Limit('stopped')
                    extra_counts['bound_skips']+=1
                    if work>=max_work:raise _Limit('work_limit')
                    continue
                if bound['structurally_infeasible'] or bound['upper_bound']<target:
                    extra_counts[kind+'_prunes']+=1
                    return False
                if kind=='clique' and bound.get('completion') is not None:
                    if accept({**assignments,**bound['completion']}):
                        extra_counts['clique_completions']+=1
                        raise _Witness()
        if cost_partition_bound:
            from overlap_bounds import lower_bound
            check();extra_counts['partition_calls']+=1
            restricted=[]
            for cell,original in enumerate(reference):
                if cell in assignments:restricted.append((assignments[cell],))
                elif cell in selected:restricted.append(tuple(s for s in domains[cell] if not bits[s//4]&mask))
                else:restricted.append((original,))
            candidate=replace(problem,domains=tuple(restricted))
            groups=[]
            for row in range(problem.size-1):
                for col in range(problem.size-1):
                    a=row*problem.size+col;group=(a,a+1,a+problem.size,a+problem.size+1)
                    if selected.intersection(group):groups.append(group)
            groups.sort(key=lambda g:(math.prod(len(restricted[c]) for c in g),g))
            extra_counts['partition_setup_work']+=len(restricted)+len(groups)+len(edges)
            groups=groups[:6]
            bound=lower_bound(candidate,regions=groups,max_work=min(12000,max_work-work),
                              max_tuples=2000,strategy='uniform',should_stop=should_stop)
            work+=bound['work'];extra_counts['partition_work']+=bound['work']
            extra_counts['partition_setup_work']+=bound['setup_state_visits']
            if not bound['complete']:
                extra_counts['partition_skips']+=1
                if bound['termination']=='stopped':raise _Limit('stopped')
                if work>=max_work:raise _Limit('work_limit')
            else:
                covered={e for group_edges in bound['region_edges'] for e in group_edges}
                outside=sum(len(restricted[a])==len(restricted[b])==1 and
                            problem.faces[restricted[a][0]][sa]!=problem.faces[restricted[b][0]][sb]
                            for e,(a,b,sa,sb) in enumerate(edges) if e not in covered)
                if bound['structurally_infeasible'] or maximum-bound['lower_bound']-outside<target:
                    extra_counts['partition_prunes']+=1
                    return False
        if not patch_bounds:
            return True
        check(); restricted = []
        for cell, original in enumerate(reference):
            if cell in assignments:
                restricted.append((assignments[cell],))
            elif cell in selected:
                restricted.append(tuple(s for s in domains[cell] if not mask & bits[s // 4]))
            else:
                restricted.append((original,))
        bound_calls += 1
        bound = patch_lower_bound(problem, domains=restricted,
                                  max_work=min(2000, max_work - work), should_stop=should_stop)
        bound_assignments += bound['work']; work += bound['work']
        if not bound['complete']:
            raise _Limit('stopped')
        if bound['structurally_infeasible'] or maximum - bound['lower_bound'] < target:
            patch_bound_prunes += 1
            return False
        return True

    def accept(assignments):
        nonlocal best, best_score
        candidate = reference.copy()
        for c, s in assignments.items():
            candidate[c] = s
        tested = evaluate_board(problem, candidate)
        if (not tested['valid'] or any(candidate[c] != reference[c] for c in range(len(reference)) if c not in selected)
                or sorted(candidate[c] // 4 for c in chosen) != pool):
            raise RuntimeError('Regional witness failed independent validation')
        if tested['score'] > best_score:
            best, best_score = candidate, tested['score']
        return tested['score'] >= target

    def result(termination, completed=False):
        reached = best_score >= target
        proven = completed and not reached
        return dict(board=best.copy(), score=best_score, reference_score=initial['score'],
                    target_score=target, target_reached=reached, proven_unreachable=proven,
                    complete=completed, termination=termination, improved=best_score > initial['score'],
                    operator='exact-region-' + algorithm, proof_scope=SCOPE, region_cells=chosen.copy(),
                    work=work, search_work=work, candidate_placements=placements,
                    bound_assignments=bound_assignments, bound_calls=bound_calls,
                    patch_bound_prunes=patch_bound_prunes, patch_bounds=patch_bounds, expanded_states=expanded,
                    dominated_states=dominated, bound_prunes=bound_prunes,
                    peak_states=peak_states, cache_hit=cache_hit, ordering=ordering,
                    assignment_bound=assignment_bound,clique_bound=clique_bound,cost_partition_bound=cost_partition_bound,
                    bounds_setup_states=score_bounds.setup_state_evaluations if score_bounds else 0,
                    bounds_setup_edges=score_bounds.setup_edges if score_bounds else 0,**extra_counts)

    try:
        check()
        if best_score >= target:
            return result('target_already_reached')
        if cache is not None:
            saved = cache._get(key)
            if saved is not None:
                accept({c: saved[c] for c in chosen})
                if best_score >= target:
                    raise RuntimeError('Cached negative proof contradicts its witness')
                cache_hit = True
                return result('cached_proven_unreachable', True)
        if not chosen or target > maximum:
            if cache is not None: cache._record_completed(key, best)
            return result('proven_unreachable', True)
        if certified_prices:
            from regional_dual import bound as price_bound
            # The entire exterior is fixed to the parent. Presolve may exclude
            # states only for this exact pool, this target and this exterior.
            restricted=tuple(domains[c] if c in selected else (state,) for c,state in enumerate(reference))
            setup=len(reference)+sum(len(d) for d in restricted)
            if setup>max_work-work:raise _Limit('work_limit')
            work+=setup;extra_counts['price_work']+=setup
            narrowed=replace(problem,domains=restricted)
            priced=price_bound(narrowed,seconds=.25,max_work=min(50000,(max_work-work)//2),should_stop=should_stop)
            used=priced['work'];work+=used;extra_counts['price_work']+=used
            extra_counts['price_calls']+=1;extra_counts['price_simplex_iterations']+=priced['simplex_iterations']
            if priced['termination']=='stopped':raise _Limit('stopped')
            if priced['complete'] and priced['by_state'] is not None:
                allowed_loss=maximum-target
                for c in chosen:
                    count=len(domains[c])
                    if work+count>max_work:raise _Limit('work_limit')
                    work+=count;extra_counts['price_work']+=count
                    domains[c]=tuple(s for s in domains[c] if priced['by_state'][c][s]<=allowed_loss)
                    extra_counts['price_fixings']+=count-len(domains[c])
                if priced['lower_bound']>allowed_loss or any(not domains[c] for c in chosen):
                    extra_counts['price_prunes']+=1
                    if cache is not None:cache._record_completed(key,best)
                    return result('proven_unreachable',True)
        if not admissible_prefix({},0):
            if cache is not None: cache._record_completed(key,best)
            return result('proven_unreachable',True)
        if algorithm == 'frontier':
            # Choose row/column traversal by exact live-boundary width, independent
            # of scores. Used physical identities remain in every key after retirement.
            orders = (chosen, sorted(chosen, key=lambda c: (c % problem.size, c // problem.size)))
            def frontiers(order):
                return [tuple(c for c in order[:i] if any(other in set(order[i:]) for other, _, _ in neighbors[c]))
                        for i in range(len(order) + 1)]
            order = min(orders, key=lambda o: (max(map(len, frontiers(o))), tuple(o)))
            live = frontiers(order)
            # Entries retain full prefix witnesses for bounded reconstruction.
            states = {(0, ()): (0, ())}
            for i, cell in enumerate(order):
                check(); next_states = {}; expanded += len(states)
                before = order[:i]
                scored_count = sum(len(boundary[c]) for c in before) + sum(a in before and b in before for a, b, _, _ in internal)
                scored_next = scored_count + len(boundary[cell]) + sum(other in before for other, _, _ in neighbors[cell])
                for (mask, _), (value, prefix) in states.items():
                    assigned = dict(zip(before, prefix))
                    for state in domains[cell]:
                        bit = bits[state // 4]
                        if mask & bit:
                            continue
                        consume()
                        gain = unary[cell][state] + sum(problem.faces[state][side] == problem.faces[assigned[other]][other_side]
                                                        for other, side, other_side in neighbors[cell] if other in assigned)
                        total = value + gain
                        if fixed_score + total + len(changing) - scored_next < target:
                            bound_prunes += 1; continue
                        extended = prefix + (state,)
                        if i + 1 == len(order):
                            if accept(dict(zip(order, extended))):
                                return result('target_reached')
                        assigned[cell] = state
                        if not admissible_prefix(assigned, mask | bit):
                            del assigned[cell]
                            continue
                        state_key = (mask | bit, tuple(assigned[c] for c in live[i + 1]))
                        old = next_states.get(state_key)
                        if old is not None and old[0] >= total:
                            dominated += 1
                            del assigned[cell]
                            continue
                        if old is not None: dominated += 1
                        next_states[state_key] = (total, extended)
                        peak_states = max(peak_states, len(next_states))
                        if len(next_states) > max_states:
                            raise _Limit('state_limit')
                        del assigned[cell]
                states = next_states
                if not states:
                    break
        else:
            assigned = {}; failures = dict.fromkeys(chosen, 0)
            def dfs(mask, value, scored):
                nonlocal expanded, bound_prunes
                check(); expanded += 1
                if len(assigned) == len(chosen):
                    return accept(assigned)
                options = {c: [s for s in domains[c] if not mask & bits[s // 4]] for c in chosen if c not in assigned}
                if ordering == 'natural': cell = min(options)
                elif ordering == 'failure': cell = min(options, key=lambda c: (Fraction(len(options[c]), 1 + failures[c]),
                                                                                  -sum(o in assigned for o, _, _ in neighbors[c]), c))
                else: cell = min(options, key=lambda c: (len(options[c]), -sum(o in assigned for o, _, _ in neighbors[c]), c))
                values = options[cell]
                def gain(s):
                    return unary[cell][s] + sum(problem.faces[s][side] == problem.faces[assigned[o]][os]
                                               for o, side, os in neighbors[cell] if o in assigned)
                if ordering != 'natural': values = sorted(values, key=lambda s: (-gain(s), s))
                next_scored = scored + len(boundary[cell]) + sum(o in assigned for o, _, _ in neighbors[cell])
                for state in values:
                    consume(); total = value + gain(state)
                    if fixed_score + total + len(changing) - next_scored < target:
                        bound_prunes += 1; continue
                    assigned[cell] = state
                    found = (admissible_prefix(assigned, mask | bits[state // 4])
                             and dfs(mask | bits[state // 4], total, next_scored))
                    del assigned[cell]
                    if found: return True
                failures[cell] += 1
                return False
            if dfs(0, 0, 0):
                return result('target_reached')
        check()
        if cache is not None:
            cache._record_completed(key, best)
        return result('proven_unreachable', True)
    except _Witness:
        return result('target_reached')
    except _Limit as exc:
        return result(str(exc))


def patch_lower_bound(problem, *, domains=None, max_work=20000, should_stop=None):
    """Admissible mismatch bound from edge-disjoint 2x3 patches and single edges.

    Patch enumeration is complete or discarded. Competition between patches is
    relaxed; no edge is counted twice. The bound can safely be max-combined with
    a separate 2x2 bound, never added to it.
    """
    _integer('max_work', max_work, 0, 1000000)
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    base = initial_domains(problem)
    if domains is None: domains = base
    else:
        if len(domains) != len(base): raise ValueError('One domain per cell is required')
        domains = tuple(tuple(dict.fromkeys(d)) for d in domains)
        if any(any(type(s) is not int for s in d) or not set(d) <= set(allowed) for d, allowed in zip(domains, base)):
            raise ValueError('Bound domains must be legal subsets')
    def check():
        if should_stop is not None and should_stop(): raise _Limit('stopped')
    edges = grid_edges(problem.size); groups = []; work = skipped = 0
    try:
        check()
        if any(not d for d in domains):
            return dict(lower_bound=None, structurally_infeasible=True, complete=True, groups=[], work=0, skipped_patches=0)
        support = []
        for domain in domains:
            check(); sides = [{} for _ in range(4)]
            for s in domain:
                for side, color in enumerate(problem.faces[s]):
                    sides[side][color] = sides[side].get(color, 0) | (1 << (s // 4))
            support.append(sides)
        edge_min = []
        for a, b, sa, sb in edges:
            check(); left, right = support[a][sa], support[b][sb]
            possible = any(color in right and (mask | right[color]).bit_count() >= 2 for color, mask in left.items())
            edge_min.append(0 if possible else 1)
        remaining = set(range(len(edges)))
        for row in range(0, problem.size - 1, 2):
            for col in range(0, problem.size - 2, 3):
                check(); cells = tuple((row + dr) * problem.size + col + dc for dr in range(2) for dc in range(3))
                indices = [i for i, (a, b, _, _) in enumerate(edges) if a in cells and b in cells]
                total = math.prod(len(domains[c]) for c in cells)
                if total > max_work - work:
                    skipped += 1; continue
                best = None; offsets = {c: i for i, c in enumerate(cells)}; floor = sum(edge_min[i] for i in indices)
                for values in itertools.product(*(domains[c] for c in cells)):
                    work += 1
                    if work % 32 == 0: check()
                    if len({s // 4 for s in values}) != 6: continue
                    cost = sum(problem.faces[values[offsets[a]]][sa] != problem.faces[values[offsets[b]]][sb]
                               for i in indices for a, b, sa, sb in (edges[i],))
                    best = cost if best is None else min(best, cost)
                    if best == floor: break
                check()
                if best is None:
                    return dict(lower_bound=None, structurally_infeasible=True, complete=True, groups=groups,
                                work=work, skipped_patches=skipped)
                groups.append(dict(edges=indices, minimum_mismatches=best, kind='2x3-relaxation'))
                remaining.difference_update(indices)
        groups.extend(dict(edges=[i], minimum_mismatches=edge_min[i], kind='edge-relaxation') for i in sorted(remaining))
        return dict(lower_bound=sum(g['minimum_mismatches'] for g in groups), structurally_infeasible=False,
                    complete=True, groups=groups, work=work, skipped_patches=skipped)
    except _Limit:
        return dict(lower_bound=None, structurally_infeasible=False, complete=False, groups=[], work=work,
                    skipped_patches=skipped, termination='stopped')
