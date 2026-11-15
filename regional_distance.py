"""Stronger necessary distance for a 480-edge solution.

For each changed subset, keep all unaffected tiles fixed inside a region.
Changed cells must have distinct physical donors and can optionally be
required to match each other. All exterior cells are relaxed.
"""
from itertools import combinations, islice
from exact_model import initial_domains,constraint_edges


class Cap(Exception):pass


def ids(mask):
    while mask:
        bit=mask&-mask;mask^=bit;yield bit.bit_length()-1


class DonorDistance:
    def __init__(self,p,board):
        self.p=p;self.board=tuple(board)
        ds=initial_domains(p)
        if len(ds)!=len(board) or len({s//4 for s in board})!=len(board) or any(s not in d for s,d in zip(board,ds)):
            raise ValueError('Invalid reference board')
        self.allowed=[sum(1<<s for s in d if s!=board[c]) for c,d in enumerate(ds)]
        self.donors=[15<<(4*(s//4)) for s in board]
        self.neighbors=[[] for _ in ds];self.broken=[]
        self.colors=[{} for _ in range(4)]
        for s,face in enumerate(p.faces):
            for side,color in enumerate(face):self.colors[side][color]=self.colors[side].get(color,0)|(1<<s)
        for a,b,sa,sb in constraint_edges(p):
            self.neighbors[a].append((b,sa,sb));self.neighbors[b].append((a,sb,sa))
            if p.faces[board[a]][sa]!=p.faces[board[b]][sb]:self.broken.append((a,b))
        self.setup=sum(map(len,p.domains))+sum(map(len,ds))+len(p.faces)*4+len(constraint_edges(p))
        self.work=self.subsets=self.placements=self.donor_tests=0

    def improve(self,region,base,*,level='edges',extra=3,max_work=300000):
        # The supplied base is an existing lower bound. The exported CLI
        # certificate independently rechecks all smaller cardinalities.
        try:
            region=tuple(islice(iter(region),13))
        except TypeError as exc:
            raise ValueError('Invalid region scope') from exc
        if (not 1<=len(region)<=12 or any(type(c) is not int or not 0<=c<len(self.board) for c in region)
                or len(set(region))!=len(region) or type(base) is not int or not 0<=base<=len(region)):
            raise ValueError('Invalid region scope')
        if level not in ('matching','edges') or type(extra) is not int or extra<0:
            raise ValueError('Invalid donor search options')
        if type(max_work) is not int or max_work<0:
            raise ValueError('Invalid donor search work limit')
        region=tuple(sorted(region))
        inside=set(region)
        breaks=[(a,b) for a,b in self.broken if a in inside and b in inside]
        start=self.work
        def take(n=1):
            if self.work-start+n>max_work:raise Cap
            self.work+=n
        def possible(changes):
            active=set(changes);fixed=inside-active
            take(len(region)+len(breaks))
            if any(a in fixed and b in fixed for a,b in breaks):return False
            banned=sum(self.donors[c] for c in fixed);dom={}
            for c in changes:
                take(1+len(self.neighbors[c]))
                states=self.allowed[c]&~banned
                for d,side,opp in self.neighbors[c]:
                    if d in fixed:
                        states&=self.colors[side].get(self.p.faces[self.board[d]][opp],0)
                if not states:return False
                dom[c]=states
            if level=='matching':
                # Match by physical piece, collapsing all allowed rotations.
                occupied={}
                def augment(c,seen):
                    vals=dom[c]
                    while vals:
                        flag=vals&-vals;piece=(flag.bit_length()-1)//4
                        vals &= ~(15<<(4*piece))
                        take();self.donor_tests+=1
                        if piece in seen:continue
                        seen.add(piece)
                        if piece not in occupied or augment(occupied[piece],seen):
                            occupied[piece]=c;return True
                    return False
                return all(augment(c,set()) for c in sorted(changes,key=lambda c:dom[c].bit_count()))
            def visit(domains):
                if not domains:return True
                take(len(domains))
                cell=min(domains,key=lambda c:domains[c].bit_count())
                for state in ids(domains[cell]):
                    take();self.placements+=1
                    rest={c:(mask&~(15<<(4*(state//4)))) for c,mask in domains.items() if c!=cell}
                    take(len(rest)+len(self.neighbors[cell]))
                    for d,side,opp in self.neighbors[cell]:
                        if d in rest:rest[d]&=self.colors[opp].get(self.p.faces[state][side],0)
                    if all(rest.values()) and visit(rest):return True
                return False
            return visit(dom)
        proved=base
        try:
            for k in range(base,min(base+extra,len(region))):
                for moves in combinations(region,k):
                    take();self.subsets+=1
                    if possible(moves):return proved,'witness'
                proved=k+1
        except Cap:return proved,'work_limit'
        return proved,'target_reached'
