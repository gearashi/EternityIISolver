"""Local Hall failures at required neighbors of a prepared native DFS alternative."""
from collections import Counter
from itertools import combinations


def _bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _ids(mask):
    # Internal physical ID masks have one low bit for each group of four rotations.
    return sum(1 << (index // 4) for index in _bits(mask))


def hall_proves_choice(search, frame, choice, *, audit=None):
    """True only after all valid rotations at one endpoint have failed.

    Takes immutable domains after ordinary DFS prepare. A parent singleton tile
    is already unavailable to every other cell, including the other neighbors.
    """
    domains = frame['domains']
    kind, target = frame['kind'], frame['cell']
    counts = Counter() if audit is not None else None
    if audit is not None:
        audit.update(counts=counts, parts=[], excluded=False)
        counts['dispatched_alternatives_checked'] += 1
    if kind == 'cell':
        parts = ((target, domains[target] & (1 << choice)),)
    elif kind == 'piece':
        parts = ((choice, domains[choice] & (15 << (4 * target))),)
    elif kind == 'edge':
        a, b, side, far = search.edges[target]
        parts = ((a, domains[a] & search.colors[side].get(choice, 0)),
                 (b, domains[b] & search.colors[far].get(choice, 0)))
    else:
        raise ValueError('Unknown DFS partition kind')

    for cell, roots in parts:
        if counts is not None:
            counts['endpoint_scopes'] += 1
            detail = dict(cell=cell, states=list(_bits(roots)), roots=[], refuted=False)
            audit['parts'].append(detail)
        if not roots or len(search.neighbors[cell]) < 2:
            continue
        neighbors = []
        for other, side, far in search.neighbors[cell]:
            if counts is not None:
                counts['neighbor_domain_reads'] += 1
            d = domains[other]
            if not d:
                # A genuine prepared frame cannot contain an empty domain.
                return False
            if d & ~(15 << ((d.bit_length() - 1) & ~3)):
                neighbors.append((other, side, far))
        degree = len(neighbors)
        if degree < 2:
            continue
        # Retain only a single prepared parent, with a bounded number of buckets.
        if search._hall_domains is not domains:
            search._hall_domains = domains
            search._hall_buckets = {}
            if counts is not None:
                counts['new_parent_caches'] += 1
        buckets = search._hall_buckets
        for state in _bits(roots):
            search.check()
            pools = []
            source = 1 << (4 * (state // 4))
            face = search.faces[state]
            if counts is not None:
                counts['source_orientations'] += 1
                root_info = dict(state=state, pools=[], certificate=None)
                detail['roots'].append(root_info)
            for other, side, far in neighbors:
                color = face[side]
                key = other, far, color
                ids = buckets.get(key)
                if counts is not None:
                    counts['bucket_lookups'] += 1
                if ids is None:
                    ids = search.pieces(domains[other] & search.colors[far].get(color, 0))
                    if len(buckets) == 128:
                        buckets.clear()
                        if counts is not None:
                            counts['cache_capacity_resets'] += 1
                    buckets[key] = ids
                    if counts is not None:
                        counts['color_domain_intersections_and_id_projections'] += 1
                mask = ids & ~source
                size = mask.bit_count()
                if counts is not None:
                    counts['source_id_removals_and_cardinalities'] += 1
                    root_info['pools'].append((other, hex(_ids(mask))))
                # A pool this large cannot belong to a deficient subset.
                if size < degree:
                    pools.append((other, mask, size))
            failed = False
            # Check all subset sizes still possible. A singleton shortage is
            # already excluded by native distinct-tile arc consistency.
            for k in range(2, len(pools) + 1):
                small = [row for row in pools if row[2] < k]
                if counts is not None:
                    counts['subset_sizes_scanned'] += 1
                    counts['cardinality_comparisons'] += len(pools)
                if len(small) < k:
                    continue
                for group in combinations(small, k):
                    union = 0
                    for _, mask, _ in group:
                        union |= mask
                    if counts is not None:
                        counts['subset_tests'] += 1
                        counts['id_unions'] += k
                    if union.bit_count() < k:
                        failed = True
                        if counts is not None:
                            root_info['certificate'] = dict(neighbors=[r[0] for r in group],
                                                            union=hex(_ids(union)), size=k)
                        break
                if failed:
                    break
            if not failed:
                break
        else:
            search.check()
            if counts is not None:
                detail['refuted'] = audit['excluded'] = True
            return True
    search.check()
    return False
