"""Necessary simultaneous checkerboard inventory equations over GF(2).

A bit in the solution vector means the physical piece is on a black square.
The filter only returns changes after elimination and recovery have completed.
It requires the complete framed grid and orientation-invariant piece inventories.
"""
from collections import Counter

from exact_model import constraint_edges, grid_edges


class _Incomplete(Exception):
    pass


def forced_variables(rows, count, charge):
    """Return fixed variable values in the affine space, or None if inconsistent.

    Variable bits are 0..count-1. The bit at count is the right-hand side.
    A pivot equation reduced against every other pivot gives a fixed value
    exactly when its only variable is the pivot.
    """
    variables = (1 << count) - 1
    basis = {}
    for row in rows:
        charge()
        while row & variables:
            bit = (row & -row).bit_length() - 1
            if bit not in basis:
                basis[bit] = row
                break
            charge()
            row ^= basis[bit]
        else:
            if row:
                return None
    forced = {}
    pivots = sorted(basis, reverse=True)
    for index, bit in enumerate(pivots):
        row = basis[bit]
        if row & variables == 1 << bit:
            forced[bit] = row >> count
        for other in pivots[index + 1:]:
            charge()
            if basis[other] & (1 << bit):
                basis[other] ^= row
    return forced


class CheckerboardParity:
    def __init__(self, problem):
        n = problem.size ** 2
        self.count = n
        self.black = tuple((cell // problem.size + cell % problem.size) % 2 == 0 for cell in range(n))
        self.rows = []
        self.setup = 0
        self.usable = set(constraint_edges(problem)) == set(grid_edges(problem.size))
        self.frame = []
        frames = {}
        totals = Counter()
        odd_pieces = Counter()
        for piece in range(n):
            faces = problem.faces[4 * piece:4 * piece + 4]
            counts = Counter(faces[0])
            for rotation, face in enumerate(faces):
                self.setup += 4
                if Counter(face) != counts:
                    self.usable = False
                shape = tuple(color == 0 for color in face)
                frames[shape] = frames.get(shape, 0) | (1 << (4 * piece + rotation))
            for color, amount in counts.items():
                self.setup += 1
                if color:
                    totals[color] += amount
                    if amount % 2:
                        odd_pieces[color] |= 1 << piece
        for cell in range(n):
            shape = (cell < problem.size, cell % problem.size == problem.size - 1,
                     cell >= n - problem.size, cell % problem.size == 0)
            self.frame.append(frames.get(shape, 0))
            self.setup += 1
        self.odd_total = any(value % 2 for value in totals.values())
        rhs = 1 << n
        self.rows = [(rhs - 1) | (sum(self.black) % 2) * rhs]
        self.rows.extend(odd_pieces[color] | ((value // 2) % 2) * rhs for color, value in totals.items())
        self.low_bits = sum(1 << (4 * piece) for piece in range(n))

    def apply(self, domains, *, limit=50000, should_stop=None):
        """Return narrowed domains, False for a proved failure, or None on skip/cap.

        Input domains are bounded bitsets from the solver. The caller owns them.
        Work counts scanned entries, bitset operations in elimination and
        projection operations. Python integer costs vary with bitset width.
        """
        if len(domains) != self.count or type(limit) is not int or limit < 0:
            raise ValueError('Invalid domain count or work limit')
        if any(type(d) is not int or d < 0 or d.bit_length() > 4 * self.count for d in domains):
            raise ValueError('Invalid domain bitset')
        spent = 0
        def charge():
            nonlocal spent
            if spent >= limit or (should_stop and should_stop()):
                raise _Incomplete
            spent += 1
        try:
            if not self.usable:
                return None, spent
            sides = [0, 0]
            for cell, domain in enumerate(domains):
                charge()
                if domain & ~self.frame[cell]:
                    return None, spent
                if not domain:
                    return False, spent
                sides[int(self.black[cell])] |= domain
            if self.odd_total:
                return False, spent
            for side, mask in enumerate(sides):
                sides[side] = (mask | (mask >> 1) | (mask >> 2) | (mask >> 3)) & self.low_bits
            rows = list(self.rows)
            for p in range(self.count):
                charge()
                here = 1 << (4 * p)
                white, black = bool(sides[0] & here), bool(sides[1] & here)
                if not white and not black:
                    return False, spent
                if white != black:
                    rows.append((1 << p) | (int(black) << self.count))
            fixed = forced_variables(rows, self.count, charge)
            if fixed is None:
                return False, spent
            forbidden = [0, 0]
            for p, side in fixed.items():
                charge()
                forbidden[1 - side] |= 15 << (4 * p)
            result = []
            for cell, d in enumerate(domains):
                charge()
                result.append(d & ~forbidden[int(self.black[cell])])
            return (result if all(result) else False), spent
        except _Incomplete:
            return None, spent
