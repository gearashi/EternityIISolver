"""Improve parent-union table rows without changing internal edge signatures."""
from exact_model import frame_ok, grid_edges


def lift_rows(problem, reference, cells, allowed, *, stop=None):
    selected=set(cells)
    inside={c:[] for c in cells}
    outside={c:[] for c in cells}
    counts=dict(grid_edges=0,domain_states=0,orientation_tests=0,color_tests=0,rows_before=0,rows_after=0)
    for a,b,sa,sb in grid_edges(problem.size):
        counts['grid_edges']+=1
        for c,d,side,other in ((a,b,sa,sb),(b,a,sb,sa)):
            if c in selected:
                if d in selected:
                    inside[c].append(side)
                else:
                    outside[c].append((side,problem.faces[reference[d]][other]))
    lifted={}
    model_parent=list(reference)
    for c in cells:
        if stop and stop():
            return None,None,counts
        domain=set(problem.domains[c])
        counts['domain_states']+=len(problem.domains[c])
        old=allowed[c]
        counts['rows_before']+=len(old)
        # Distinct physical pieces remain distinguishable in every table.
        cache={}
        for piece in sorted({s//4 for s in old}):
            for s in range(piece*4,piece*4+4):
                counts['orientation_tests']+=1
                if s not in domain or not frame_ok(problem.size,c,problem.faces[s]):
                    continue
                face=problem.faces[s]
                signature=(piece,*(face[side] for side in inside[c]))
                value=sum(face[side]==color for side,color in outside[c])
                counts['color_tests']+=len(inside[c])+len(outside[c])
                # Stable ties prefer the current parent state where possible.
                rank=(value,s==reference[c],-s)
                if signature not in cache or rank>cache[signature][0]:
                    cache[signature]=(rank,s)
        improved={}
        for s in old:
            key=(s//4,*(problem.faces[s][side] for side in inside[c]))
            counts['color_tests']+=len(inside[c])
            improved[s]=cache[key][1]
        lifted[c]=sorted(set(improved.values()))
        counts['rows_after']+=len(lifted[c])
        model_parent[c]=improved[reference[c]]
    return lifted,model_parent,counts
