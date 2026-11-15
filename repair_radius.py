"""Exact search for a strict score increase in a one- or two-cell neighborhood.

A changed cell includes a piece rotation. Results apply to this exact parent,
its fixed clues, frame and domains. No external services or production files.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from exact_model import constraint_edges, frame_ok, from_bundle, grid_edges
from exact_optimize import evaluate_board


def search(problem, board, *, radius=2, should_stop=None, max_work=None):
    """Search for the next score inside the specified Hamming ball.

    The cap includes setup state visits, edge comparisons, eligibility checks,
    pairs and orientation candidates. Validation is counted separately.
    """
    if type(radius) is not int or radius not in (1, 2):
        raise ValueError('radius must be one or two')
    if max_work is not None and (type(max_work) is not int or max_work < 0):
        raise ValueError('max_work must be a nonnegative integer')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    edges = grid_edges(problem.size)
    if set(constraint_edges(problem)) != set(edges):
        raise ValueError('full grid required')
    reference = tuple(board)
    initial = evaluate_board(problem, reference)
    if not initial['valid']:
        raise ValueError('invalid reference')
    start = initial['score']
    counts = dict(domain_states=0, edge_tests=0, cell_visits=0, pair_visits=0,
                  orientation_tests=0, domain_lookups=0, validation_calls=1)
    work = 0
    confirmed_radius = 0

    class Interrupted(Exception):
        pass

    def charge(key, count=1):
        nonlocal work
        if max_work is not None and work + count > max_work:
            raise Interrupted('work_limit')
        if (work == 0 or work // 128 != (work + count) // 128) and should_stop and should_stop():
            raise Interrupted('stopped')
        counts[key] += count
        work += count

    def result(status, candidate=reference, points=start):
        return dict(termination=status, score=points, reference_score=start, radius=radius,
                    max_exhausted_radius=confirmed_radius, board=list(candidate),
                    improved=points > start, work=work, counts=counts.copy(),
                    proof_scope='strict improvement from supplied board within changed-cell radius only')

    def witness(changes, delta):
        if delta <= 0:
            raise AssertionError('invalid internal score delta')
        candidate = list(reference)
        for c, state in changes:
            candidate[c] = state
        actual = evaluate_board(problem, candidate)
        counts['validation_calls'] += 1
        if (not actual['valid'] or actual['score'] != start + delta
                or sum(left != right for left, right in zip(reference, candidate)) > radius):
            raise RuntimeError('local-delta witness failed independent validation')
        return result('improved', candidate, actual['score'])

    neighbors = [[] for _ in reference]
    adjacent = {}
    anchors = set()
    domains = []
    faces = problem.faces
    gain_cache = {}

    def choices(c, piece):
        states = []
        for state in range(piece * 4, piece * 4 + 4):
            charge('domain_lookups')
            if state in domains[c] and frame_ok(problem.size, c, faces[state]):
                states.append(state)
        return states

    def gain(c, state):
        key = (c, state)
        if key not in gain_cache:
            value = 0
            for side, other_color, old in neighbors[c]:
                charge('edge_tests')
                value += int(faces[state][side] == other_color) - old
            gain_cache[key] = value
        return gain_cache[key]

    def consider_pair(c, d, left, right, sides):
        if not left or not right:
            return None
        if sides is None:
            # Each state's delta involves disjoint edges. Maximizing the two
            # deltas separately represents the entire product of orientations.
            charge('orientation_tests', len(left) + len(right))
            s = max(left, key=lambda item: gain(c, item))
            t = max(right, key=lambda item: gain(d, item))
            delta = gain(c, s) + gain(d, t)
            return witness(((c, s), (d, t)), delta) if delta > 0 else None
        sc, sd = sides
        x, y = faces[reference[c]][sc], faces[reference[d]][sd]
        for s, t in itertools.product(left, right):
            charge('orientation_tests')
            u, v = faces[s][sc], faces[t][sd]
            charge('edge_tests', 4)
            correction = int(u == v) + int(x == y) - int(u == y) - int(x == v)
            delta = gain(c, s) + gain(d, t) + correction
            if delta > 0:
                return witness(((c, s), (d, t)), delta)
        return None

    try:
        if should_stop and should_stop():
            return result('stopped')
        if start == len(edges):
            confirmed_radius = radius
            return result('already_perfect')
        for domain in problem.domains:
            charge('domain_states', len(domain))
            domains.append(set(domain))
        for c, d, sc, sd in edges:
            charge('edge_tests')
            x, y = faces[reference[c]][sc], faces[reference[d]][sd]
            same = int(x == y)
            neighbors[c].append((sc, y, same))
            neighbors[d].append((sd, x, same))
            adjacent[c, d] = (sc, sd)
            if not same:
                anchors.update((c, d))
        anchors.difference_update(problem.fixed)
        mutable = tuple(c for c in range(len(reference)) if c not in problem.fixed)
        for c in sorted(anchors):
            charge('cell_visits')
            for s in choices(c, reference[c] // 4):
                if s != reference[c]:
                    charge('orientation_tests')
                    delta = gain(c, s)
                    if delta > 0:
                        return witness(((c, s),), delta)
        confirmed_radius = 1
        if radius == 2:
            for a in sorted(anchors):
                for b in mutable:
                    charge('pair_visits')
                    if a == b or (b < a and b in anchors):
                        continue
                    c, d = sorted((a, b))
                    sides = adjacent.get((c, d))
                    # Nonadjacent rotations add two individually nonpositive gains.
                    if sides is not None:
                        left = [s for s in choices(c, reference[c] // 4) if s != reference[c]]
                        right = [s for s in choices(d, reference[d] // 4) if s != reference[d]]
                        found = consider_pair(c, d, left, right, sides)
                        if found:
                            return found
                    left = choices(c, reference[d] // 4)
                    if left:
                        found = consider_pair(c, d, left, choices(d, reference[c] // 4), sides)
                        if found:
                            return found
            confirmed_radius = 2
        return result('exhausted')
    except Interrupted as exc:
        return result(str(exc))


def main():
    parser = argparse.ArgumentParser(description='Check an offline board for one- and two-cell score improvements.')
    parser.add_argument('board', type=Path, help='JSON input containing the oriented board array under board')
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--radius', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    from validator import load_bundle, validate_board
    bundle = load_bundle(args.data)
    board = json.loads(args.board.read_text(encoding='utf-8-sig'))['board']
    answer = search(from_bundle(bundle), board, radius=args.radius)
    checked = validate_board(answer['board'], bundle)
    if not checked['valid'] or checked['score'] != answer['score']:
        raise RuntimeError('independent official validator rejected result')
    print(json.dumps(answer))


if __name__ == '__main__':
    main()
