"""Bounded consequences of immutable neighboring-cell supports in an exact DFS frame."""
from collections import Counter, deque
from dataclasses import dataclass


@dataclass(slots=True)
class NeighborGraph:
    faces: tuple
    neighbors: list
    arcs: dict
    zeros: dict


def build_neighbor_graph(faces, colors, neighbors, domains, check, *, audit=None):
    """Build the whole graph before any query. Neither its scope nor masks depend on a choice."""
    arcs = {}
    zeros = {}
    buckets = {}
    counts = None
    if audit is not None:
        counts = Counter()
        audit['counts'] = counts
        audit['bitset_work_bounds'] = bounds = Counter()
    for cell, mask in enumerate(domains):
        check()
        edges = neighbors[cell]
        if counts is not None:
            counts['parent_cell_reads'] += 1
        if not edges:
            continue
        visited = 0
        while mask:
            flag = mask & -mask
            mask ^= flag
            state = flag.bit_length() - 1
            piece = 15 << (state // 4 * 4)
            face = faces[state]
            visited += 1
            if visited % 64 == 0:
                check()
            if counts is not None:
                counts['parent_oriented_state_visits'] += 1
            for other, side, far in edges:
                color = face[side]
                key = other, far, color
                allowed = buckets.get(key)
                if allowed is None:
                    color_mask = colors[far].get(color, 0)
                    allowed = domains[other] & color_mask
                    buckets[key] = allowed
                    if counts is not None:
                        counts['neighbor_color_bucket_intersections'] += 1
                        bounds['cached_domain_intersection_64bit_upper'] += (max(domains[other].bit_length(), color_mask.bit_length()) + 63) // 64
                support = allowed & ~piece
                if not support:
                    # Only one zero witness is needed. Preserve original edge iteration order.
                    zeros.setdefault((cell, state), (cell, other, side, far))
                elif support & (support - 1) == 0:
                    arcs.setdefault((cell, state), []).append((other, support.bit_length() - 1))
                if counts is not None:
                    counts['directed_source_adjacencies'] += 1
                    counts['source_color_reads'] += 1
                    counts['support_cache_lookups'] += 1
                    counts['single_source_piece_exclusions'] += 1
                    counts['support_zero_tests'] += 1
                    bounds['exclude_piece_64bit_upper'] += (max(allowed.bit_length(), piece.bit_length()) + 63) // 64
                    if support:
                        counts['support_singleton_tests'] += 1
                        counts['arcs' if support & (support - 1) == 0 else 'ambiguous_supports'] += 1
                    else:
                        counts['zero_marks'] += 1
    check()
    return NeighborGraph(faces, neighbors, arcs, zeros)


class _Limit(Exception):
    pass


def query_neighbor_graph(graph, center, legal, check, *, audit=None, steps=4, literals=32, checks=8192):
    """A piece-location dispatch is invalid only if every surviving orientation is refuted."""
    work = reached = 0
    total = None
    if audit is not None:
        total = Counter()
        audit.update(counts=total, roots=[], center=center, legal_states=list(legal), tentative_placements=0)

    def charge(key, amount=1):
        nonlocal work
        if work + amount > checks:
            raise _Limit('work_limit')
        work += amount
        if total is not None:
            total[key] += amount

    def trace(root):
        nonlocal reached
        seen = {}
        pending = deque()
        cells = {}
        pieces = {}
        boundary = False

        def enter(node, previous, depth):
            nonlocal reached
            charge('literal_seen_lookups')
            if node in seen:
                return
            if reached >= literals:
                raise _Limit('literal_limit')
            reached += 1
            seen[node] = previous
            charge('new_literal_records')
            pending.append((node, depth))
            charge('queue_pushes')

        def proof(reason, endpoints, edge=None):
            charge('proof_records')
            paths = []
            for node in endpoints:
                path = []
                while node is not None:
                    charge('proof_path_literals')
                    if audit is not None:
                        path.append(node)
                    node = seen[node]
                if audit is not None:
                    paths.append(list(reversed(path)))
            if edge is not None:
                charge('proof_edge_records')
            if audit is None:
                return 'proved', None
            return 'proved', dict(reason=reason, paths=paths, edge=edge)

        charge('root_queries')
        enter(root, None, 0)
        while pending:
            check()
            charge('queue_pops')
            node, depth = pending.popleft()
            cell, state = node
            piece = state // 4
            charge('reached_cell_checks')
            if cell in cells and cells[cell] != node:
                return proof('cell', (cells[cell], node))
            charge('reached_piece_checks')
            if piece in pieces and pieces[piece] != node:
                return proof('piece', (pieces[piece], node))
            for other, side, far in graph.neighbors[cell]:
                charge('existing_endpoint_lookups')
                if other in cells:
                    charge('required_color_comparisons')
                    if graph.faces[state][side] != graph.faces[cells[other][1]][far]:
                        return proof('color', (node, cells[other]), (cell, other, side, far))
            cells[cell] = pieces[piece] = node
            charge('endpoint_registrations', 2)
            charge('zero_list_lookups')
            if node in graph.zeros:
                charge('zero_mark_reads')
                if depth + 1 <= steps:
                    return proof('zero', (node,), graph.zeros[node])
                boundary = True
            charge('arc_list_lookups')
            for other in graph.arcs.get(node, ()):
                charge('graph_arc_visits')
                if depth == steps:
                    charge('depth_boundary_seen_checks')
                    if other not in seen:
                        boundary = True
                else:
                    enter(other, node, depth+1)
        return ('depth_limit' if boundary else 'closure'), None

    outcomes = []
    for state in legal:
        check()
        start_work, start_seen = work, reached
        before = total.copy() if total is not None else None
        try:
            result, certificate = trace((center, state))
        except _Limit as limit:
            result, certificate = str(limit), None
        outcomes.append(result)
        if audit is not None:
            audit['roots'].append(dict(root=(center,state), outcome=result, certificate=certificate,
                                       reached=reached-start_seen, work=work-start_work, counts=dict(total-before)))
        if result in ('work_limit','literal_limit'):
            break
    check()
    proved = bool(outcomes) and len(outcomes) == len(legal) and all(o == 'proved' for o in outcomes)
    if audit is not None:
        status = outcomes[-1] if outcomes and outcomes[-1] in ('work_limit','literal_limit') else (
            'proved' if proved else 'depth_limit' if 'depth_limit' in outcomes else 'closure')
        audit.update(outcome=status, work=work, reached_total=reached)
    return proved

