"""Exact bounded fast path for integer binary pairwise objectives.

Separable pair terms become unary costs. Independent interaction components
are optimized separately. Forests use tree dynamic programming; small cyclic
components use complete Gray-code enumeration. Unsupported or interrupted work
returns no optimal assignment, leaving the caller's general solver available.
"""
from __future__ import annotations
import math
import time


def maximize(unary, pairs, *, seconds=.05, should_stop=None, max_component_bits=16):
    started = time.monotonic()
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if type(max_component_bits) is not int or not 1 <= max_component_bits <= 20:
        raise ValueError('max_component_bits must be 1..20')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    if not isinstance(unary, (list, tuple)) or len(unary) > 256:
        raise ValueError('Expected at most256 binary variables')
    original = [tuple(row) for row in unary]
    if any(len(row) != 2 or any(type(v) is not int for v in row) for row in original):
        raise ValueError('Unary costs must be integer pairs')
    n = len(original)
    if not isinstance(pairs, dict) or len(pairs) > n*(n-1)//2:
        raise ValueError('Pair costs must be a canonical edge dictionary')
    source_pairs = {}
    for key, row in pairs.items():
        if not isinstance(key, tuple) or len(key) != 2 or any(type(v) is not int for v in key) or not 0 <= key[0] < key[1] < n:
            raise ValueError('Pair key must be distinct ordered variable indices')
        row = tuple(row)
        if len(row) != 4 or any(type(v) is not int for v in row):
            raise ValueError('Pair costs must be four exact integers')
        source_pairs[key] = row
    deadline = started + seconds
    trees = enumerated = removed = visited = 0
    def result(termination, assignment=None, value=None):
        return dict(optimal=termination == 'optimal', assignment=assignment, value=value,
                    termination=termination, tree_components=trees, enumerated_components=enumerated,
                    removed_separable_pairs=removed, assignments_evaluated=visited,
                    elapsed_seconds=time.monotonic()-started)
    def reason():
        if should_stop is not None and should_stop(): return 'stopped'
        if time.monotonic() >= deadline: return 'timeout'
        return None
    why = reason()
    if why: return result(why)
    costs = [list(row) for row in original]
    neighbors = [dict() for _ in range(n)]
    constant = 0
    for (a, b), (w00, w01, w10, w11) in source_pairs.items():
        why = reason()
        if why: return result(why)
        constant += w00
        costs[a][1] += w10-w00
        costs[b][1] += w01-w00
        coupling = w11-w10-w01+w00
        if coupling:
            neighbors[a][b] = coupling
            neighbors[b][a] = coupling
        else:
            removed += 1
    unseen = set(range(n)); components = []
    while unseen:
        first = min(unseen); stack = [first]; unseen.remove(first); nodes = []
        while stack:
            why = reason()
            if why: return result(why)
            node = stack.pop(); nodes.append(node)
            for other in sorted(neighbors[node], reverse=True):
                if other in unseen: unseen.remove(other); stack.append(other)
        edges = sum(len(neighbors[v]) for v in nodes)//2
        if edges != len(nodes)-1 and len(nodes) > max_component_bits:
            return result('unsupported_component')
        components.append((sorted(nodes), edges == len(nodes)-1))
    assignment = [0]*n; total = constant
    for nodes, forest in components:
        why = reason()
        if why: return result(why)
        if forest:
            trees += 1
            first = nodes[0]; parent = {first: None}; order = [first]
            for node in order:
                for other in sorted(neighbors[node]):
                    if other != parent[node]: parent[other] = node; order.append(other)
            dp = {}
            for node in reversed(order):
                why = reason()
                if why: return result(why)
                row = costs[node].copy()
                for other, coupling in neighbors[node].items():
                    if parent.get(other) == node:
                        row[0] += max(dp[other])
                        row[1] += max(dp[other][0], dp[other][1]+coupling)
                dp[node] = row
            assignment[first] = int(dp[first][1] > dp[first][0])
            total += dp[first][assignment[first]]
            for node in order[1:]:
                ancestor = parent[node]
                assignment[node] = int(dp[node][1]+neighbors[node][ancestor]*assignment[ancestor] > dp[node][0])
        else:
            enumerated += 1
            local = {v:i for i,v in enumerate(nodes)}
            bits = [0]*len(nodes)
            current = sum(costs[v][0] for v in nodes)
            best = current; best_mask = 0; previous = 0; visited += 1
            for step in range(1, 1 << len(nodes)):
                if step % 128 == 0:
                    why = reason()
                    if why: return result(why)
                mask = step ^ (step >> 1)
                changed = (mask ^ previous).bit_length()-1
                node = nodes[changed]
                direction = 1 if bits[changed] == 0 else -1
                current += direction*(costs[node][1]-costs[node][0]+sum(weight*bits[local[other]] for other,weight in neighbors[node].items()))
                bits[changed] ^= 1
                visited += 1
                if current > best: best=current; best_mask=mask
                previous=mask
            for i,node in enumerate(nodes): assignment[node]=(best_mask >> i)&1
            total += best
    why = reason()
    if why: return result(why)
    # Independently re-evaluate the original unnormalized objective.
    actual = sum(original[i][bit] for i,bit in enumerate(assignment)) + sum(row[2*assignment[a]+assignment[b]] for (a,b),row in source_pairs.items())
    if actual != total: raise RuntimeError('Binary fast path objective mismatch')
    return result('optimal', assignment, actual)
