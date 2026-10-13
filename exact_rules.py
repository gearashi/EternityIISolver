"""Optional, solution-preserving filters for oriented edge-matching domains.

No I/O, solver runs, or global caches. All budgets are deterministic work caps;
a skipped or interrupted check never uses partially enumerated supports to prune.
Whole-board identities require a complete framed grid and invariant per-piece
color multiplicities. Regional problems deliberately bypass those identities.
"""
from __future__ import annotations

from collections import deque
import time

from exact_model import constraint_edges, grid_edges

RULES_VERSION = 'additional-rules-v1'
DEFAULT_OPTIONS = {
    'hall': True, 'color_balance': True, 'directional_balance': True,
    'blocks': True, 'color_flow': True,
    'balance_work': 12000, 'block_pairs': 1024, 'block_total_pairs': 8192,
    'flow_colors': 2, 'flow_max_edges': 160,
}


def normalize_options(options=None):
    if options is None:
        return DEFAULT_OPTIONS.copy()
    if not isinstance(options, dict) or set(options) - set(DEFAULT_OPTIONS):
        raise ValueError('Invalid additional-rule options')
    result = DEFAULT_OPTIONS.copy()
    for key, value in options.items():
        default = DEFAULT_OPTIONS[key]
        if type(default) is bool:
            if type(value) is not bool:
                raise ValueError('Rule switches must be booleans')
        elif type(value) is not int or not 0 <= value <= 1000000:
            raise ValueError('Rule work caps must be bounded nonnegative integers')
        result[key] = value
    return result


def _bits(mask):
    while mask:
        flag = mask & -mask
        mask ^= flag
        yield flag.bit_length() - 1


class AdditionalRules:
    def __init__(self, problem, check=lambda: None, options=None):
        self.problem, self.check = problem, check
        self.options = normalize_options(options)
        self.n = problem.size ** 2
        self.faces = problem.faces
        self.piece_mask = sum(1 << (4 * p) for p in range(self.n))
        self.black = tuple((c // problem.size + c % problem.size) % 2 == 0 for c in range(self.n))
        self.black_count = sum(self.black)
        self.edges = tuple(constraint_edges(problem))
        self.full_grid = set(self.edges) == set(grid_edges(problem.size))
        self.invariant = all(len({tuple(sorted(f)) for f in self.faces[4*p:4*p+4]}) == 1 for p in range(self.n))
        self.frame_masks = []
        self.colors = [{} for _ in range(4)]
        for state, face in enumerate(self.faces):
            if state % 64 == 0:
                self.check()
            for side, color in enumerate(face):
                self.colors[side][color] = self.colors[side].get(color, 0) | (1 << state)
        all_states = (1 << len(self.faces)) - 1
        for cell in range(self.n):
            self.check()
            outside = (cell < problem.size, cell % problem.size == problem.size - 1,
                       cell >= self.n - problem.size, cell % problem.size == 0)
            mask = all_states
            for side, border in enumerate(outside):
                gray = self.colors[side].get(0, 0)
                mask &= gray if border else all_states ^ gray
            self.frame_masks.append(mask)
        palette = sorted({c for f in self.faces for c in f} - {0})
        self.weights = {c: tuple(self.faces[4*p].count(c) for p in range(self.n)) for c in palette}
        self.totals = {c: sum(w) for c, w in self.weights.items()}
        self.palette = tuple(sorted(palette, key=lambda c: (self.totals[c], c)))
        self.degree_masks = {c: tuple(sum(1 << s for s, f in enumerate(self.faces) if f.count(c) == k) for k in range(5)) for c in self.palette} if self.options['color_flow'] else {}
        self.direction_masks = {}
        for color in (self.palette if self.options['directional_balance'] else ()):
            self.direction_masks[color] = tuple(tuple(sum(1 << s for s, f in enumerate(self.faces)
                if int(f[a] == color) - int(f[b] == color) == value) for value in (-1, 0, 1))
                for a, b in ((1, 3), (0, 2)))
        edge_pairs = {frozenset((a, b)) for a, b, _, _ in self.edges}
        self.blocks = []
        for row in range(problem.size - 1):
            for col in range(problem.size - 1):
                a = row * problem.size + col
                cells = (a, a + 1, a + 1 + problem.size, a + problem.size)
                if all(frozenset((cells[i], cells[(i+1) % 4])) in edge_pairs for i in range(4)):
                    self.blocks.append(cells)
        self.counters = {'checks': 0, 'failures': 0, 'removed_states': 0, 'seconds': 0.0}
        for name in ('hall', 'balance', 'direction', 'blocks', 'flow'):
            for suffix in ('checks', 'failures', 'skips', 'seconds'):
                self.counters[name + '_' + suffix] = 0
        self.counters.update(block_pairs=0, balance_work=0, flow_edges=0)

    def pieces(self, domain):
        return (domain | domain >> 1 | domain >> 2 | domain >> 3) & self.piece_mask

    def _reduce(self, domains, cell, keep, changed):
        after = domains[cell] & keep
        if after != domains[cell]:
            self.counters['removed_states'] += domains[cell].bit_count() - after.bit_count()
            domains[cell] = after
            changed.add(cell)
        return bool(after)

    def apply(self, domains):
        """Mutate state-bitset domains; return (consistent, changed_cells).

        Call after ordinary arc consistency and repeat propagation after changes.
        The caller must supply frame/fixed-filtered domains. Unframed input is
        tolerated, but disables every whole-board identity as a safety measure.
        """
        self.check()
        if len(domains) != self.n or any(type(d) is not int or d < 0 or d >> len(self.faces) for d in domains):
            raise ValueError('Expected one bounded state bitset per cell')
        self.counters['checks'] += 1
        started = time.perf_counter()
        changed = set()
        try:
            if any(not d for d in domains):
                self.counters['failures'] += 1
                return False, changed
            framed = all(not (d & ~m) for d, m in zip(domains, self.frame_masks))
            identities = self.full_grid and framed
            steps = (
                ('hall', self.options['hall'], self._hall),
                ('balance', self.options['color_balance'] and identities and self.invariant, self._balance),
                ('direction', self.options['directional_balance'] and identities, self._direction),
                ('blocks', self.options['blocks'], self._blocks),
                ('flow', self.options['color_flow'] and identities and self.invariant, self._flow),
            )
            for name, enabled, method in steps:
                self.check()
                if not enabled:
                    self.counters[name + '_skips'] += 1
                    continue
                self.counters[name + '_checks'] += 1
                rule_started = time.perf_counter()
                try:
                    consistent = method(domains, changed)
                finally:
                    self.counters[name + '_seconds'] += time.perf_counter() - rule_started
                if not consistent:
                    self.counters[name + '_failures'] += 1
                    self.counters['failures'] += 1
                    return False, changed
            return True, changed
        finally:
            self.counters['seconds'] += time.perf_counter() - started

    def _hall(self, domains, changed):
        groups = {}
        for cell, domain in enumerate(domains):
            groups.setdefault(self.pieces(domain), []).append(cell)
        for pieces, cells in groups.items():
            self.check()
            if len(cells) > pieces.bit_count():
                return False
            if len(cells) != pieces.bit_count() or len(cells) == self.n:
                continue
            reserved = set(cells)
            forbidden = pieces * 15
            for cell in range(self.n):
                if cell not in reserved and not self._reduce(domains, cell, ~forbidden, changed):
                    return False
        return True

    def _balance(self, domains, changed):
        eligible = [0] * self.n
        for cell, domain in enumerate(domains):
            for offset in _bits(self.pieces(domain)):
                eligible[offset // 4] |= 1 if self.black[cell] else 2
        if any(not mask for mask in eligible):
            return False
        fixed_black = [p for p, mask in enumerate(eligible) if mask == 1]
        flexible = [p for p, mask in enumerate(eligible) if mask == 3]
        count = self.black_count - len(fixed_black)
        if not 0 <= count <= len(flexible):
            return False
        remaining_work = self.options['balance_work']
        for color in self.palette:
            self.check()
            total = self.totals[color]
            if total % 2:
                return False
            weight = self.weights[color]
            target = total // 2 - sum(weight[p] for p in fixed_black)
            counts = [0] * 5
            for piece in flexible:
                counts[weight[piece]] += 1
            ordered = [w for w in range(5) for _ in range(counts[w])]
            minimum = sum(ordered[:count])
            maximum = sum(ordered[len(ordered)-count:]) if count else 0
            if not minimum <= target <= maximum:
                return False
            if count == 0 or count == len(flexible):
                continue
            # Exact bounded-cardinality bitset knapsack, grouping equal weights.
            chunks = []
            for w, quantity in enumerate(counts):
                step = 1
                while quantity:
                    take = min(step, quantity)
                    chunks.append((take, take * w))
                    quantity -= take
                    step *= 2
            cost = sum(max(0, count - take + 1) for take, _ in chunks)
            if cost > remaining_work:
                self.counters['balance_skips'] += 1
                continue
            remaining_work -= cost
            dp = [0] * (count + 1)
            dp[0] = 1
            limit = (1 << (target + 1)) - 1
            for take, shift in chunks:
                self.check()
                for used in range(count, take - 1, -1):
                    dp[used] |= (dp[used-take] << shift) & limit
            self.counters['balance_work'] += cost
            if not (dp[count] >> target) & 1:
                return False
        return True

    def _direction(self, domains, changed):
        for color in self.palette:
            for masks in self.direction_masks[color]:
                self.check()
                ranges = []
                for domain in domains:
                    values = [k-1 for k, mask in enumerate(masks) if domain & mask]
                    ranges.append((min(values), max(values)))
                lower = sum(x[0] for x in ranges)
                upper = sum(x[1] for x in ranges)
                if lower > 0 or upper < 0:
                    return False
                for cell, (lo, hi) in enumerate(ranges):
                    allowed = 0
                    for k, mask in enumerate(masks):
                        value = k - 1
                        if lower - lo + value <= 0 <= upper - hi + value:
                            allowed |= mask
                    if not self._reduce(domains, cell, allowed, changed):
                        return False
        return True

    def _side(self, a, b):
        delta = b - a
        return 1 if delta == 1 else 3 if delta == -1 else 2 if delta == self.problem.size else 0

    def _blocks(self, domains, changed):
        budget = self.options['block_total_pairs']
        examined_pairs = 0  # Local cadence keeps adaptive work stable on replay.
        for cells in self.blocks:
            self.check()
            a, b, c, d = cells
            if domains[b].bit_count() * domains[d].bit_count() < domains[a].bit_count() * domains[c].bit_count():
                a, b, c, d = b, c, d, a
            cost = domains[a].bit_count() * domains[c].bit_count()
            if cost > self.options['block_pairs'] or cost > budget:
                self.counters['blocks_skips'] += 1
                continue
            budget -= cost
            supports = {cell: 0 for cell in (a, b, c, d)}
            ab, ba = self._side(a, b), self._side(b, a)
            cb, bc = self._side(c, b), self._side(b, c)
            ad, da = self._side(a, d), self._side(d, a)
            cd, dc = self._side(c, d), self._side(d, c)
            for sa in _bits(domains[a]):
                self.check()
                for sc in _bits(domains[c]):
                    self.counters['block_pairs'] += 1
                    examined_pairs += 1
                    if examined_pairs % 64 == 0:
                        self.check()
                    if sa // 4 == sc // 4:
                        continue
                    excluded = (15 << (4 * (sa // 4))) | (15 << (4 * (sc // 4)))
                    bs = domains[b] & self.colors[ba].get(self.faces[sa][ab], 0) & self.colors[bc].get(self.faces[sc][cb], 0) & ~excluded
                    ds = domains[d] & self.colors[da].get(self.faces[sa][ad], 0) & self.colors[dc].get(self.faces[sc][cd], 0) & ~excluded
                    if not bs or not ds:
                        continue
                    bp, dp = self.pieces(bs), self.pieces(ds)
                    valid_b = bs & ~(15 * dp) if dp & (dp - 1) == 0 else bs
                    valid_d = ds & ~(15 * bp) if bp & (bp - 1) == 0 else ds
                    if valid_b and valid_d:
                        supports[a] |= 1 << sa
                        supports[c] |= 1 << sc
                        supports[b] |= valid_b
                        supports[d] |= valid_d
            # No support-based deletion occurs until this block is complete.
            for cell, support in supports.items():
                if not self._reduce(domains, cell, support, changed):
                    return False
        return True

    def _flow(self, domains, changed):
        for color in self.palette[:self.options['flow_colors']]:
            self.check()
            total = self.totals[color]
            if total % 2:
                return False
            target = total // 2
            degree = []
            for domain in domains:
                possible = [k for k, mask in enumerate(self.degree_masks[color]) if domain & mask]
                degree.append((min(possible), max(possible)))
            possible_edges = []
            adjacency = [[] for _ in range(self.n)]
            for a, b, sa, sb in self.edges:
                aa = domains[a] & self.colors[sa].get(color, 0)
                bb = domains[b] & self.colors[sb].get(color, 0)
                if not aa or not bb or (self.pieces(aa) | self.pieces(bb)).bit_count() < 2:
                    continue
                possible_edges.append((a, b) if self.black[a] else (b, a))
                adjacency[a].append(b)
                adjacency[b].append(a)
            self.counters['flow_edges'] += len(possible_edges)
            # A cheap component balance is useful even if exact flow is skipped.
            seen = set()
            min_total = max_total = 0
            for root in range(self.n):
                if root in seen:
                    continue
                self.check()
                seen.add(root)
                todo = [root]
                bl = bh = wl = wh = edge_count = 0
                while todo:
                    cell = todo.pop()
                    lo, hi = degree[cell]
                    if self.black[cell]:
                        bl += lo
                        bh += hi
                        edge_count += len(adjacency[cell])
                    else:
                        wl += lo
                        wh += hi
                    for other in adjacency[cell]:
                        if other not in seen:
                            seen.add(other)
                            todo.append(other)
                low, high = max(bl, wl), min(bh, wh, edge_count)
                if low > high:
                    return False
                min_total += low
                max_total += high
            if not min_total <= target <= max_total:
                return False
            if len(possible_edges) > self.options['flow_max_edges']:
                self.counters['flow_skips'] += 1
                continue
            if not self._bounded_flow(degree, possible_edges, target):
                return False
        return True

    def _bounded_flow(self, degree, edges, target):
        source, sink, supersource, supersink = self.n, self.n + 1, self.n + 2, self.n + 3
        graph = [[] for _ in range(self.n + 4)]
        demands = [0] * (self.n + 2)
        def edge(a, b, capacity):
            graph[a].append([b, capacity, len(graph[b])])
            graph[b].append([a, 0, len(graph[a])-1])
        def bounded(a, b, lower, upper):
            edge(a, b, upper-lower)
            demands[a] -= lower
            demands[b] += lower
        for cell, (lower, upper) in enumerate(degree):
            if self.black[cell]:
                bounded(source, cell, lower, upper)
            else:
                bounded(cell, sink, lower, upper)
        for a, b in edges:
            edge(a, b, 1)
        bounded(sink, source, target, target)
        required = 0
        for vertex, demand in enumerate(demands):
            if demand > 0:
                edge(supersource, vertex, demand)
                required += demand
            elif demand < 0:
                edge(vertex, supersink, -demand)
        sent = 0
        while sent < required:
            self.check()
            parent = [None] * len(graph)
            parent[supersource] = (-1, -1)
            todo = deque([supersource])
            while todo and parent[supersink] is None:
                vertex = todo.popleft()
                for index, (other, capacity, _) in enumerate(graph[vertex]):
                    if capacity and parent[other] is None:
                        parent[other] = (vertex, index)
                        todo.append(other)
            if parent[supersink] is None:
                return False
            amount = required - sent
            vertex = supersink
            while vertex != supersource:
                previous, index = parent[vertex]
                amount = min(amount, graph[previous][index][1])
                vertex = previous
            vertex = supersink
            while vertex != supersource:
                previous, index = parent[vertex]
                entry = graph[previous][index]
                entry[1] -= amount
                graph[vertex][entry[2]][1] += amount
                vertex = previous
            sent += amount
        return True


