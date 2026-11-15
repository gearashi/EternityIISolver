"""Select compatible legal patches using their exact pairwise score terms."""
from itertools import combinations, islice
from time import monotonic

from binary_pairwise import maximize
from exact_model import constraint_edges, grid_edges
from exact_optimize import evaluate_board


def compose(problem, parent, candidates, *, seconds=2.0, should_stop=None):
    start=monotonic()
    if type(seconds) not in (int,float) or not 0<=seconds<float('inf'):
        raise ValueError('seconds must be finite and nonnegative')
    edges=grid_edges(problem.size)
    if set(constraint_edges(problem))!=set(edges):
        raise ValueError('full-grid score objective required')
    best=list(parent)
    checked=evaluate_board(problem,best)
    if not checked['valid']:
        raise ValueError('illegal base board')
    initial=score=checked['score']
    boards=list(islice(candidates,65))
    if len(boards)>64:
        raise ValueError('at most 64 patch boards per search')
    patches=[]
    unary=[]
    interactions={}
    conflicts=[]
    edge_tests=overlap_tests=component_states=0
    def result(status):
        return dict(termination=status,optimal=status=='optimal',board=list(best),score=score,
                    reference_score=initial,improved=score>initial,patches=len(patches),
                    edge_tests=edge_tests,overlap_tests=overlap_tests,
                    binary_assignments_evaluated=component_states,elapsed=monotonic()-start)
    def stop():
        if should_stop and should_stop():
            return 'stopped'
        if monotonic()-start>=seconds:
            return 'timeout'
        return None

    parent=tuple(parent)
    for b in boards:
        why=stop()
        if why:
            return result(why)
        verified=evaluate_board(problem,b)
        if not verified['valid']:
            raise ValueError('every input must be a legal complete patch board')
        changed={c:s for c,s in enumerate(b) if s!=parent[c]}
        if not changed or changed in patches:
            continue
        patches.append(changed)
        unary.append((0,verified['score']-initial))
        if verified['score']>score:
            score,best=verified['score'],list(b)
    for i,j in combinations(range(len(patches)),2):
        why=stop()
        if why:
            return result(why)
        left,right=patches[i],patches[j]
        overlap_tests+=1
        if not left.keys().isdisjoint(right):
            conflicts.append((i,j))
            continue
        # Unaries score each patch against the parent. Only grid edges with
        # their endpoints in different selected patches need a correction.
        joint=0
        for a,b,sa,sb in edges:
            edge_tests+=1
            if a in left and b in right:
                s,t=left[a],right[b]
            elif a in right and b in left:
                s,t=right[a],left[b]
            else:
                continue
            x,y=problem.faces[parent[a]][sa],problem.faces[parent[b]][sb]
            u,v=problem.faces[s][sa],problem.faces[t][sb]
            joint+=int(u==v)+int(x==y)-int(u==y)-int(x==v)
        if joint:
            interactions[i,j]=(0,0,0,joint)
    # The empty choice scores zero and the sum of all positive terms bounds
    # every unpenalized choice, so any overlap loses to the empty choice.
    penalty=1+sum(max(0,p[1]) for p in unary)+sum(max(0,p[3]) for p in interactions.values())
    for key in conflicts:
        interactions[key]=(0,0,0,-penalty)
    native=maximize(unary,interactions,seconds=max(0,seconds-(monotonic()-start)),should_stop=should_stop)
    component_states=native['assignments_evaluated']
    if not native['optimal']:
        return result(native['termination'])
    chosen=[p for p,take in zip(patches,native['assignment']) if take]
    changed_cells=set()
    candidate=list(parent)
    for p in chosen:
        if not changed_cells.isdisjoint(p):
            raise RuntimeError('overlapping selections escaped exclusion')
        changed_cells.update(p)
        for c,s in p.items():
            candidate[c]=s
    actual=evaluate_board(problem,candidate)
    if not actual['valid'] or actual['score']!=initial+native['value'] or actual['score']<score:
        raise RuntimeError('patch composition failed independent validation')
    score,best=actual['score'],candidate
    return result('optimal')
