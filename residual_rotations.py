"""Merge observationally identical orientations in an exact feasibility problem."""
from dataclasses import replace

from exact_model import constraint_edges, initial_domains


def quotient_rotations(problem):
    """Return (representative problem, deleted rotations, signature comparisons).

    For a given cell and piece, two legal orientations with identical colors
    on every constrained incident side can be exchanged in any witness. Fixed
    clue states are filtered before choosing representatives. Symmetry reduction
    preserves existence, not the number of labeled solutions.
    """
    sides = [[] for _ in problem.domains]
    for a, b, sa, sb in constraint_edges(problem):
        sides[a].append(sa)
        sides[b].append(sb)
    domains = []
    removed = work = 0
    for cell, legal in enumerate(initial_domains(problem)):
        seen = set()
        kept = []
        for state in legal:
            work += 1 + len(sides[cell])
            key = (state // 4, *(problem.faces[state][side] for side in sides[cell]))
            if key not in seen:
                kept.append(state)
                seen.add(key)
            else:
                removed += 1
        domains.append(tuple(kept))
    return replace(problem, domains=tuple(domains)), removed, work
