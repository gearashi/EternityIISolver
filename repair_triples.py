"""Exact strict-improvement search for at most three changed cells.

After proving that every single and pair fails, improving triples must be
connected by changed-cell adjacency or movement of physical pieces. Component
gains add when their edge sets and physical-piece pools are disjoint.
"""
from __future__ import annotations

from itertools import combinations, product

from exact_model import frame_ok, grid_edges
from exact_optimize import evaluate_board
from repair_radius import search as search_pairs


def search(problem, board, *, should_stop=None, max_work=None):
    reference = tuple(board)
    pairs = search_pairs(problem, reference, should_stop=should_stop, max_work=max_work)
    start = pairs['reference_score']
    work = pairs['work']
    counts = dict(pair_phase_work=work, setup_domain_states=0, support_edges=0,
                  triple_visits=0, allocation_checks=0, state_checks=0, edge_tests=0,
                  bounded_allocations=0, orientation_tuples=0, validation_calls=0)

    def result(status, exhausted, best=None, score=start):
        return dict(termination=status, score=score, improved=score>start, reference_score=start,
                    radius=3, max_exhausted_radius=exhausted, board=list(reference if best is None else best),
                    work=work, counts=counts.copy(), pair_counts=pairs['counts'],
                    proof_scope='strict improvement from supplied board within changed-cell radius only')

    if pairs['improved']:
        return result('improved', pairs['max_exhausted_radius'], pairs['board'], pairs['score'])
    if pairs['termination'] == 'already_perfect':
        return result('already_perfect', 3)
    if pairs['termination'] != 'exhausted':
        return result(pairs['termination'], pairs['max_exhausted_radius'])

    class Stopped(Exception):
        pass

    def charge(key, amount=1):
        nonlocal work
        if max_work is not None and work + amount > max_work:
            raise Stopped('work_limit')
        if work // 256 != (work + amount) // 256 and should_stop and should_stop():
            raise Stopped('stopped')
        work += amount
        counts[key] += amount

    def accept(cells, states, delta):
        candidate = list(reference)
        for c,s in zip(cells,states):
            candidate[c]=s
        actual = evaluate_board(problem,candidate)
        counts['validation_calls'] += 1
        if (not actual['valid'] or actual['score'] != start + delta or delta <= 0
                or sum(a!=b for a,b in zip(candidate,reference)) != 3):
            raise RuntimeError('three-cell witness failed independent validation')
        return result('improved',2,candidate,actual['score'])

    def profile(c, origin, hidden):
        # hidden is a mask of edges with both endpoints changed. All other
        # edges have fixed neighbors. Cache only complete tables.
        key = (c,origin,hidden)
        if key not in profiles:
            values = []
            for s in range(4 * (reference[origin]//4), 4 * (reference[origin]//4) + 4):
                charge('state_checks')
                if s == reference[c] or s not in domains[c] or not frame_ok(problem.size,c,faces[s]):
                    continue
                gain = 0
                for side,other,base in adj[c]:
                    if not hidden & (1<<side):
                        charge('edge_tests')
                        gain += int(faces[s][side] == other) - base
                values.append((gain,s))
            values.sort(reverse=True)
            profiles[key] = values
        return profiles[key]

    def try_allocation(cells, sources, masks, links, potential):
        charge('allocation_checks')
        options=[]
        for c,source,mask in zip(cells,sources,masks):
            row=profile(c,source,mask)
            if not row:
                return None
            options.append(row)
        upper=sum(row[0][0] for row in options)+potential
        if upper<=0:
            charge('bounded_allocations')
            return None
        if not links:
            return accept(cells,[row[0][1] for row in options],upper)
        for triple in product(*options):
            charge('orientation_tuples')
            gain=sum(item[0] for item in triple)
            if gain+potential<=0:
                continue
            states=tuple(item[1] for item in triple)
            for i,j,sc,sd,old in links:
                charge('edge_tests')
                gain+=int(faces[states[i]][sc]==faces[states[j]][sd])-old
            if gain>0:
                return accept(cells,states,gain)
        return None

    try:
        if should_stop and should_stop():
            return result('stopped',2)
        faces=problem.faces
        domains=[]
        for domain in problem.domains:
            charge('setup_domain_states',len(domain))
            domains.append(set(domain))
        adj=[[] for _ in reference]
        between={}
        anchors=set()
        for c,d,sc,sd in grid_edges(problem.size):
            charge('edge_tests')
            x,y=faces[reference[c]][sc],faces[reference[d]][sd]
            old=int(x==y)
            adj[c].append((sc,y,old))
            adj[d].append((sd,x,old))
            between[c,d]=(sc,sd,old)
            between[d,c]=(sd,sc,old)
            if not old:
                anchors.update((c,d))
        anchors.difference_update(problem.fixed)
        mutable=[c for c in range(len(reference)) if c not in problem.fixed]
        profiles={}
        for a in sorted(anchors):
            other=[c for c in mutable if c!=a and (c not in anchors or c>a)]
            for b,c in combinations(other,2):
                charge('triple_visits')
                cells=(a,b,c)
                masks=[0,0,0]
                links=[]
                for i,j in ((0,1),(0,2),(1,2)):
                    detail=between.get((cells[i],cells[j]))
                    if detail is not None:
                        sc,sd,old=detail
                        masks[i]|=1<<sc
                        masks[j]|=1<<sd
                        links.append((i,j,sc,sd,old))
                charge('support_edges',3)
                possible=sum(1-entry[-1] for entry in links)
                # A three-cycle connects all three cells through piece transfers.
                allocations=[(b,c,a),(c,a,b)]
                # One stationary rotating tile and a reciprocal swap form two
                # piece components, so the stationary tile needs a grid neighbor.
                for i,j,k in ((0,1,2),(0,2,1),(1,2,0)):
                    if masks[k]:
                        origin=list(cells)
                        origin[i],origin[j]=origin[j],origin[i]
                        allocations.append(origin)
                # Pure rotations need a connected induced three-cell grid.
                if len(links)>=2:
                    allocations.append(cells)
                for sources in allocations:
                    found=try_allocation(cells,sources,masks,links,possible)
                    if found:
                        return found
        return result('exhausted',3)
    except Stopped as exc:
        return result(str(exc),2)
