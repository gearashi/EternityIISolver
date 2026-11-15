"""Replay distance certificates with exact finite regional enumeration."""
from itertools import combinations

from exact_model import constraint_edges, initial_domains


def _has_completion(problem, domains, parent, neighbors, region, changed, counts):
    assigned = {cell: parent[cell] for cell in region if cell not in changed}
    owners = {state // 4 for state in assigned.values()}

    def agrees(cell, state):
        for other, side, opposite in neighbors[cell]:
            if other in assigned and problem.faces[state][side] != problem.faces[assigned[other]][opposite]:
                return False
        return True

    if any(not agrees(cell, state) for cell, state in assigned.items()):
        return False
    choices = {}
    for cell in changed:
        available = []
        for state in domains[cell]:
            counts["verified_state_visits"] += 1
            if state != parent[cell] and state // 4 not in owners and agrees(cell, state):
                available.append(state)
        if not available:
            return False
        choices[cell] = available

    order = sorted(changed, key=lambda cell: (len(choices[cell]), cell))

    def visit(index):
        if index == len(order):
            return True
        cell = order[index]
        for state in choices[cell]:
            counts["verified_placements"] += 1
            piece = state // 4
            if piece in owners or not agrees(cell, state):
                continue
            owners.add(piece)
            assigned[cell] = state
            if visit(index + 1):
                return True
            owners.remove(piece)
            del assigned[cell]
        return False

    return visit(0)


def verify_exact_packing(problem, parent, rows, scale):
    """Check every smaller subset and the packing without a native optimizer.

    An exterior cell may always change, so a surviving local assignment is an
    admissible relaxation. A claimed lower bound must exclude every such
    assignment using fewer changed cells in that region.
    """
    if type(scale) is not int or scale <= 0:
        raise ValueError("Invalid packing scale")
    parent = tuple(parent)
    domains = initial_domains(problem)
    size = len(domains)
    if (len(parent) != size
            or any(type(s) is not int or s not in domains[c] for c, s in enumerate(parent))
            or len({s // 4 for s in parent}) != size):
        raise ValueError("Invalid parent board")
    loads = [0] * size
    numerator = 0
    counts = dict(verified_subsets=0, verified_state_visits=0, verified_placements=0)
    edges = constraint_edges(problem)

    for raw_region, minimum, weight in rows:
        region = tuple(raw_region)
        if (not 1 <= len(region) <= 12
                or any(type(cell) is not int or not 0 <= cell < size for cell in region)
                or len(set(region)) != len(region)
                or type(minimum) is not int or not 0 <= minimum <= len(region)
                or type(weight) is not int or not 1 <= weight <= scale):
            raise ValueError("Invalid regional certificate row")
        for cell in region:
            loads[cell] += weight
            if loads[cell] > scale:
                raise ValueError("Regional packing exceeds cell capacity")
        neighbors = {cell: [] for cell in region}
        for a, b, sa, sb in edges:
            if a in neighbors and b in neighbors:
                neighbors[a].append((b, sa, sb))
                neighbors[b].append((a, sb, sa))

        for count in range(minimum):
            for moves in combinations(region, count):
                counts["verified_subsets"] += 1
                if _has_completion(problem, domains, parent, neighbors, region, set(moves), counts):
                    raise ValueError("The regional lower bound admits a smaller changed subset")
        numerator += minimum * weight

    return dict(lower_bound=(numerator + scale - 1) // scale, numerator=numerator, scale=scale, **counts)
