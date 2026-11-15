"""Independently replay a five-clue distance certificate using raw printed tile sides."""
import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from struct import pack

ROOT = Path(__file__).resolve().parents[1]
CLUES = {34:831, 45:1019, 135:554, 210:723, 221:992}


def verify(pieces, parent, certificate):
    if (len(pieces) != 256 or any(len(tile) != 4 for tile in pieces)
            or len(parent) != 256 or any(type(s) is not int or not 0 <= s < 1024 for s in parent)
            or sorted(s//4 for s in parent) != list(range(256))
            or any(parent[cell] != state for cell,state in CLUES.items())):
        raise ValueError('Invalid tile set or five-clue parent')
    if sha256(pack('<256H', *parent)).hexdigest() != certificate['parent_local_sha256']:
        raise ValueError('Certificate belongs to another parent')
    if type(certificate['target_score']) is not int or certificate['target_score'] != 480:
        raise ValueError('Only perfect targets are supported')
    scale = certificate['scale']
    if type(scale) is not int or scale < 1:
        raise ValueError('Invalid packing denominator')

    # The printed order is up, down, left, right. The search traverses NESW.
    faces = [tuple(pieces[s//4][(0,3,1,2)[(side-s%4)%4]] for side in range(4)) for s in range(1024)]
    color_masks = [{} for _ in range(4)]
    for s, sides in enumerate(faces):
        for side, color in enumerate(sides):
            color_masks[side][color] = color_masks[side].get(color, 0) | (1 << s)
    neighbors, domains = [], []
    score = 0
    no_clues = ~sum(15 << (4*(s//4)) for s in CLUES.values())
    for cell, state in enumerate(parent):
        r,c = divmod(cell, 16)
        adjacent = {i:other for i,(other,on_board) in enumerate(
            ((cell-16,r>0), (cell+1,c<15), (cell+16,r<15), (cell-1,c>0))) if on_board}
        neighbors.append(adjacent)
        legal = (1 << CLUES[cell]) if cell in CLUES else ((1 << 1024)-1) & no_clues
        for side in range(4):
            gray = color_masks[side][0]
            legal &= ~gray if side in adjacent else gray
        if not legal & (1 << state):
            raise ValueError('Parent violates frame or clue')
        domains.append(legal ^ (1 << state))
        score += sum(faces[state][i] == faces[parent[other]][(i+2)%4]
                     for i,other in adjacent.items() if cell < other)
    if score != certificate['parent_score']:
        raise ValueError('Parent score is wrong')

    counts = dict(subsets=0, placements=0)
    def extend(pending):
        if not pending:
            return True
        cell = min(pending, key=lambda c: pending[c].bit_count())
        remaining = pending[cell]
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            state = bit.bit_length()-1
            counts['placements'] += 1
            same_tile = 15 << (4*(state//4))
            next_cells = {c:mask & ~same_tile for c,mask in pending.items() if c != cell}
            for i,other in neighbors[cell].items():
                if other in next_cells:
                    next_cells[other] &= color_masks[(i+2)%4].get(faces[state][i],0)
            if all(next_cells.values()) and extend(next_cells):
                return True
        return False

    loads = [0]*256
    total = 0
    for cells, lower, weight in certificate['regional_certificates']:
        if (not 1 <= len(cells) <= 12 or any(type(c) is not int or not 0 <= c < 256 for c in cells)
                or len(set(cells)) != len(cells) or type(lower) is not int or not 0 <= lower <= len(cells)
                or type(weight) is not int or not 1 <= weight <= scale):
            raise ValueError('Invalid certificate row')
        for cell in cells:
            loads[cell] += weight
            if loads[cell] > scale:
                raise ValueError('A cell exceeds its packing capacity')
        for k in range(lower):
            for movable in combinations(cells,k):
                counts['subsets'] += 1
                fixed = set(cells).difference(movable)
                if any(faces[parent[c]][i] != faces[parent[d]][(i+2)%4] for c in fixed
                       for i,d in neighbors[c].items() if d in fixed and c < d):
                    continue
                taken = sum(15 << (4*(parent[c]//4)) for c in fixed)
                pending = {c:domains[c] & ~taken for c in movable}
                for c in movable:
                    for side,d in neighbors[c].items():
                        if d in fixed:
                            pending[c] &= color_masks[side].get(faces[parent[d]][(side+2)%4],0)
                if all(pending.values()) and extend(pending):
                    raise ValueError(f'Regional lower bound fails on cells {cells}')
        total += lower*weight
    bound = (total + scale-1)//scale
    if bound != certificate['lower_bound'] or total != certificate['numerator']:
        raise ValueError('Incorrect claimed total')
    return dict(lower_bound=bound, parent_score=score, numerator=total, scale=scale, **counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('certificate', nargs='?', type=Path, default=ROOT/'docs'/'record466-distance.json')
    parser.add_argument('--board', type=Path, default=ROOT/'data'/'record466.json')
    args = parser.parse_args()
    tiles = [tuple(map(int,row.split())) for row in (ROOT/'data'/'pieces.txt').read_text().splitlines() if row.strip()]
    report = verify(tiles, json.loads(args.board.read_text(encoding='utf-8-sig'))['board'],
                    json.loads(args.certificate.read_text(encoding='utf-8-sig')))
    print(json.dumps(report))
