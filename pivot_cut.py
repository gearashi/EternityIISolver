"""Bounded implication proof for the two states of a selected DFS branch cell."""
from collections import Counter, defaultdict


class _WorkLimit(Exception):
    pass


def prove_pivot(problem, domains, target, *, edges, max_work=1_000_000, check=None):
    """Return a cut only if the selected states imply each other's negation.

    Binary cells contribute complementary literals. A forbidden pair of
    oriented states gives two implications. We omit wider and singleton
    cells, which can only weaken the relaxation. The caller supplies the
    current, post-propagation masks and the task's actual required edges.
    The outer search's interrupt exception passes through unchanged.
    """
    if type(max_work) is not int or max_work < 0:
        raise ValueError('Expected a nonnegative integer work cap')
    counts = Counter()
    work = 0
    next_poll = 128

    def charge(kind, amount=1):
        nonlocal work, next_poll
        if work + amount > max_work:
            raise _WorkLimit
        counts[kind] += amount
        work += amount
        if check is not None and work >= next_poll:
            check()
            next_poll = work + 128

    def finish(outcome):
        if check is not None:
            check()
        return dict(outcome=outcome, work=work, counts=dict(counts))

    try:
        if check is not None:
            check()
        if type(target) is not int or not 0 <= target < len(domains):
            return finish('ineligible')
        charge('pivot_width_checks')
        if domains[target].bit_count() != 2:
            return finish('ineligible')

        rows = {}
        wide = False
        for cell, mask in enumerate(domains):
            charge('cell_width_reads')
            width = mask.bit_count()
            if width == 2:
                first = mask & -mask
                rows[cell] = (first.bit_length()-1, (mask ^ first).bit_length()-1, 2*len(rows))
                charge('binary_states', 2)
            elif width > 2:
                wide = True
        if len(rows) < 2 or not wide:
            return finish('ineligible')

        charge('literal_slots', 2*len(rows))
        graph = [[] for _ in range(2*len(rows))]

        def forbid(a, b):
            charge('forbidden_pairs')
            charge('implications', 2)
            graph[a].append(b ^ 1)
            graph[b].append(a ^ 1)

        seen_piece = defaultdict(list)
        for cell, (first, second, variable) in rows.items():
            for offset, state in enumerate((first, second)):
                charge('identity_state_reads')
                occurrences = seen_piece[state//4]
                for other_cell, other_literal in occurrences:
                    charge('identity_pair_tests')
                    if cell != other_cell:
                        forbid(variable+offset, other_literal)
                occurrences.append((cell, variable+offset))

        for a, b, sa, sb in edges:
            charge('edge_scope_checks')
            if a not in rows or b not in rows:
                continue
            a0, a1, la = rows[a]
            b0, b1, lb = rows[b]
            for i, state in enumerate((a0, a1)):
                for j, other in enumerate((b0, b1)):
                    charge('edge_state_tests')
                    if problem.faces[state][sa] != problem.faces[other][sb]:
                        forbid(la+i, lb+j)

        def implies(start, goal):
            visited = {start}
            stack = [start]
            while stack:
                charge('reach_visits')
                current = stack.pop()
                for other in graph[current]:
                    charge('reach_edges')
                    if other == goal:
                        return True
                    if other not in visited:
                        visited.add(other)
                        stack.append(other)
            return False

        pivot = rows[target][2]
        if implies(pivot, pivot ^ 1) and implies(pivot ^ 1, pivot):
            return finish('contradiction')
        return finish('unchanged')
    except _WorkLimit:
        return finish('work_limit')
