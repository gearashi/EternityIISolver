"""Certified lower bounds on the number of cells a perfect board must change.

A cell can keep its parent state or take an allowed rotation. For a candidate
region we relax all exterior cells and ignore competition between changed
cells. Every true completion satisfies this relaxation. A weighted sum of
checked regional lower bounds is a global lower bound.
"""
from exact_model import constraint_edges, initial_domains


def _bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


class DistanceBounds:
    def __init__(self, problem, board):
        self.n = problem.size ** 2
        self.size = problem.size
        self.domains = initial_domains(problem)
        if (len(board) != self.n or len({s // 4 for s in board}) != self.n or
                any(s not in self.domains[c] for c, s in enumerate(board))):
            raise ValueError('Parent must use all distinct physical pieces and legal frame/clue states')
        self.board = tuple(board)
        self.owner = {s // 4: c for c,s in enumerate(board)}
        self.neighbors = [[] for _ in board]
        self.breaks = []
        for a,b,sa,sb in constraint_edges(problem):
            ca,cb = problem.faces[board[a]][sa],problem.faces[board[b]][sb]
            self.neighbors[a].append((b,sa,cb))
            self.neighbors[b].append((a,sb,ca))
            if ca != cb:
                self.breaks.append((1 << a) | (1 << b))
        self.profiles = []
        self.setup_states = self.setup_edges = self.subsets = 0
        for c, domain in enumerate(self.domains):
            groups = {}
            for state in domain:
                self.setup_states += 1
                if state == board[c]:
                    continue
                needed = 0
                for d,side,target in self.neighbors[c]:
                    self.setup_edges += 1
                    if problem.faces[state][side] != target:
                        needed |= 1 << d
                origin = self.owner[state // 4]
                groups[needed] = groups.get(needed,0) | (1 << origin)
            self.profiles.append(groups)

    def local_minimum(self, cells):
        from itertools import combinations
        cells = tuple(sorted(set(cells)))
        if not cells or len(cells)>12 or any(c<0 or c>=self.n for c in cells):
            raise ValueError('Select 1..12 valid cells')
        region = sum(1 << c for c in cells)
        breaks = [e for e in self.breaks if e & region == e]
        if not breaks:
            return 0
        options = {}
        for c in cells:
            requirements = set()
            for neighbors, owners in self.profiles[c].items():
                inside = neighbors & region
                if owners & ~region or owners & (1 << c):
                    requirements.add(inside)
                else:
                    requirements.update(inside | 1 << d for d in _bits(owners))
            minimal = []
            for req in sorted(requirements,key=int.bit_count):
                if not any(prior & req == prior for prior in minimal):
                    minimal.append(req)
            options[c] = minimal
        for k in range(1,len(cells)+1):
            for group in combinations(cells,k):
                self.subsets += 1
                changed = sum(1 << c for c in group)
                if all(changed & edge for edge in breaks) and all(
                        any(need & changed == need for need in options[c]) for c in group):
                    return k
        # No support even with every cell changed: the relaxation itself has
        # no completion. Return a valid bound up to the region size here.
        return len(cells)

    def cuts(self):
        n=self.size
        regions = {tuple(_bits(edge)) for edge in self.breaks}
        for h,w in ((2,3),(3,2),(3,3),(2,4),(4,2),(3,4),(4,3)):
            for row in range(n-h+1):
                for col in range(n-w+1):
                    cells=tuple((row+i)*n+col+j for i in range(h) for j in range(w))
                    mask=sum(1 << c for c in cells)
                    if any(e & mask == e for e in self.breaks):
                        regions.add(cells)
        return [(r,self.local_minimum(r)) for r in sorted(regions)]


def packing(cuts,n,scale=16,seconds=3.0):
    """Maximize one certificate. Any returned feasible weights can be checked."""
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    weights = [model.new_int_var(0,scale,'') for _ in cuts]
    for c in range(n):
        relevant=[x for (r,_),x in zip(cuts,weights) if c in r]
        if relevant:
            model.add(sum(relevant)<=scale)
    model.maximize(sum(x*k for (_,k),x in zip(cuts,weights)))
    solver=cp_model.CpSolver()
    solver.parameters.num_search_workers=1
    solver.parameters.max_time_in_seconds=seconds
    status=solver.solve(model)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        return [],solver.num_branches,solver.status_name(status)
    rows=[(r,k,int(solver.value(x))) for (r,k),x in zip(cuts,weights) if solver.value(x)>0]
    return rows,solver.num_branches,solver.status_name(status)


def verify_packing(problem, board, rows, scale):
    """Separate direct enumerator, no optimized support tables or native solver."""
    from itertools import combinations
    if type(scale) is not int or scale<1:
        raise ValueError('Invalid weight scale')
    n=problem.size**2
    domains=initial_domains(problem)
    if len(board)!=n or len({s//4 for s in board})!=n or any(s not in domains[c] for c,s in enumerate(board)):
        raise ValueError('Invalid parent')
    loads=[0]*n
    owner={s//4:c for c,s in enumerate(board)}
    edges=constraint_edges(problem)
    total=subsets=states=0
    for region,minimum,weight in rows:
        cells=set(region)
        if (len(cells)!=len(region) or not 1<=len(cells)<=12 or any(type(c) is not int or not 0<=c<n for c in cells)
                or type(minimum) is not int or not 0<=minimum<=len(cells) or type(weight) is not int or not 1<=weight<=scale):
            raise ValueError('Invalid local certificate row')
        local=[(a,b,sa,sb) for a,b,sa,sb in edges if a in cells and b in cells]
        for size in range(minimum):
            for trial in combinations(region,size):
                subsets+=1
                changed=set(trial)
                fixed=cells-changed
                if any(problem.faces[board[a]][sa]!=problem.faces[board[b]][sb]
                       for a,b,sa,sb in local if a in fixed and b in fixed):
                    continue
                for c in changed:
                    tests=[(sa,problem.faces[board[b]][sb]) for a,b,sa,sb in local if a==c and b in fixed]
                    tests.extend((sb,problem.faces[board[a]][sa]) for a,b,sa,sb in local if b==c and a in fixed)
                    for s in domains[c]:
                        states+=1
                        if s!=board[c] and owner[s//4] not in fixed and all(problem.faces[s][i]==color for i,color in tests):
                            break
                    else:
                        break
                else:
                    raise ValueError('Local lower bound failed independent enumeration')
        for c in cells:
            loads[c]+=weight
        total+=weight*minimum
    if any(load>scale for load in loads):
        raise ValueError('Overlapping cuts exceed their cell budget')
    return dict(lower_bound=(total+scale-1)//scale,numerator=total,scale=scale,verified_subsets=subsets,
                verified_state_visits=states)
