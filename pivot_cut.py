"""Bounded implication proof for the two states of a selected DFS branch cell."""
from collections import Counter, defaultdict


class _WorkLimit(Exception):
    pass


def prove_pivot(problem, domains, target, *, edges, max_work=1_000_000, check=None, one_sided=False, work_counts=None):
    """Return a cut only if the selected states imply each other's negation.

    Binary cells contribute complementary literals. A forbidden pair of
    oriented states gives two implications. We omit wider and singleton
    cells, which can only weaken the relaxation. The caller supplies the
    current, post-propagation masks and the task's actual required edges.
    In one-sided mode an implication from one state to its own complement
    rules out just that orientation. Only completed proofs publish exclusions.
    The outer search's interrupt exception passes through unchanged.
    """
    if type(max_work) is not int or max_work < 0:
        raise ValueError('Expected a nonnegative integer work cap')
    if work_counts is not None and (not isinstance(work_counts, Counter) or work_counts):
        raise ValueError('Expected a fresh counter for this proof')
    counts = Counter() if work_counts is None else work_counts
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

    def finish(outcome, excluded_mask=0):
        if check is not None:
            check()
        result = dict(outcome=outcome, work=work, counts=dict(counts))
        if one_sided:
            result['excluded_mask'] = excluded_mask
        return result

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

        def implies(start, goal, side):
            visits = side + '_reach_visits' if one_sided else 'reach_visits'
            edges_tested = side + '_reach_edges' if one_sided else 'reach_edges'
            visited = {start}
            stack = [start]
            while stack:
                charge(visits)
                current = stack.pop()
                for other in graph[current]:
                    charge(edges_tested)
                    if other == goal:
                        return True
                    if other not in visited:
                        visited.add(other)
                        stack.append(other)
            return False

        low_state, high_state, pivot = rows[target]
        low_fails = implies(pivot, pivot ^ 1, 'lower')
        # The released rule can stop after a failed first proof. One-sided
        # dispatch must also consider the opposite orientation.
        high_fails = (low_fails or one_sided) and implies(pivot ^ 1, pivot, 'upper')
        if low_fails and high_fails:
            return finish('contradiction', domains[target])
        if one_sided and (low_fails or high_fails):
            return finish('one_sided', 1 << (low_state if low_fails else high_state))
        return finish('unchanged')
    except _WorkLimit:
        return finish('work_limit')
