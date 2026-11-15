"""Check global grid symmetries, including bijections of the edge colors.

Colors are refined by their clockwise contexts in the physical tile multiset.
Unresolved color bijections are explicitly tried up to the stated cap. Every
surviving candidate must still preserve the tiles and the fixed, oriented clues.
"""
from collections import Counter, defaultdict
from itertools import permutations, product
from math import factorial, prod
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from exact_model import from_bundle
from validator import load_bundle


def orbit(face):
    return min(face[i:] + face[:i] for i in range(4))


def isometries(problem, cap=10000):
    if type(cap) is not int or cap < 1:
        raise ValueError('A positive permutation cap is required')
    original = [tuple(f) for f in problem.faces[::4]]
    colors = sorted({v for f in original for v in f})
    n = problem.size
    tiles = Counter(map(orbit, original))
    output = []
    for mirror in (False, True):
        # East and west switch under a left/right mirror.
        source = [(a,d,c,b) for a,b,c,d in original] if mirror else original
        labels = [{c:int(c != 0) for c in colors} for _ in range(2)]
        for _ in range(2*len(colors)):
            signatures=[]
            for faces, groups in zip((source, original), labels):
                contexts=defaultdict(list)
                for f in faces:
                    for i,c in enumerate(f):
                        contexts[c].append(tuple(groups[f[(i+j)%4]] for j in (1,2,3)))
                signatures.append({c:(groups[c],tuple(sorted(contexts[c]))) for c in colors})
            classes={signature:i for i,signature in enumerate(sorted(set(signatures[0].values()) | set(signatures[1].values())))}
            refined=[{c:classes[s] for c,s in item.items()} for item in signatures]
            stable=len(set(labels[0].values()) | set(labels[1].values())) == len(classes)
            labels=refined
            if stable:
                break
        groups=[]
        for partition in labels:
            group=defaultdict(list)
            for c in colors:
                group[partition[c]].append(c)
            groups.append(group)
        possible=0
        tested=0
        retained=[]
        if {key:len(g) for key,g in groups[0].items()} == {key:len(g) for key,g in groups[1].items()}:
            keys=sorted(groups[0])
            possible=prod(factorial(len(groups[0][key])) for key in keys)
            if possible <= cap:
                for choices in product(*(permutations(groups[1][key]) for key in keys)):
                    mapping={c:v for key,group in zip(keys,choices) for c,v in zip(groups[0][key],group)}
                    tested+=1
                    if Counter(orbit(tuple(mapping[c] for c in f)) for f in source) != tiles:
                        continue
                    for turns in range(4):
                        for index,state in problem.fixed.items():
                            r,c=divmod(index,n)
                            face=problem.faces[state]
                            if mirror:
                                c=n-1-c
                                face=(face[0],face[3],face[2],face[1])
                            for _ in range(turns):
                                r,c=c,n-1-r
                                face=(face[3],face[0],face[1],face[2])
                            other=problem.fixed.get(r*n+c)
                            if other is None or tuple(mapping[v] for v in face)!=tuple(problem.faces[other]):
                                break
                        else:
                            retained.append(dict(quarter_turns=turns,color_map=mapping))
        output.append(dict(mirror=mirror,possible_color_maps=possible,tested=tested,
                           complete=possible==tested,preserving=retained))
    return dict(distinct_orbits=len(tiles),physical_pieces=len(original),colors=len(colors),cases=output)


def main(output):
    puzzle=from_bundle(load_bundle(ROOT/'data'))
    report=isometries(puzzle)
    assert len(set(puzzle.faces))==len(puzzle.faces)
    assert all(x['complete'] for x in report['cases'])
    found=[(k['mirror'],t['quarter_turns'],t['color_map']) for k in report['cases'] for t in k['preserving']]
    assert len(found)==1 and found[0][:2]==(False,0) and all(a==b for a,b in found[0][2].items())
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main(Path(sys.argv[1]))
