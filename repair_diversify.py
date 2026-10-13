"""Bounded legal plateau escapes for the structured repair portfolio.

Only working copies may lose score. No candidate is a pruning fact or a saved
incumbent; the caller independently checks and retains its best-ever board.
"""
from dataclasses import replace
import random
from exact_model import check_solution, constraint_edges, frame_ok, grid_edges


class Diversifier:
    def __init__(self, problem):
        if set(constraint_edges(problem)) != set(grid_edges(problem.size)):
            raise ValueError('Diversification requires the full grid')
        self.problem = replace(problem, fixed=dict(problem.fixed), edges=())
        self.allowed = tuple(frozenset(s for s in domain
                             if frame_ok(problem.size, c, problem.faces[s]))
                             for c, domain in enumerate(problem.domains))
        self.mutable = tuple(c for c in range(problem.size**2) if c not in problem.fixed)

    def perturb(self, board, *, seed=0, moves=4, forbidden_boards=(), should_stop=None):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('seed must be a uint32 integer')
        if type(moves) is not int or not 1 <= moves <= 32:
            raise ValueError('moves must be 1..32')
        if should_stop is not None and not callable(should_stop):
            raise ValueError('should_stop must be callable')
        if not check_solution(self.problem, board):
            raise ValueError('Illegal diversification parent')
        original = tuple(board)
        # The portfolio passes a bounded recent-history deque. Bound other callers.
        import itertools
        forbidden = {tuple(b) for b in itertools.islice(forbidden_boards, 64)}
        forbidden.add(original)
        visited = {original}
        working = list(board)
        rng = random.Random(seed)
        performed = 0
        groups = {}
        for c in self.mutable:
            groups.setdefault(sum(v == 0 for v in self.problem.faces[board[c]]), []).append(c)
        def rotations(cell, piece):
            return [s for s in range(4*piece, 4*piece+4) if s in self.allowed[cell]]
        for _ in range(moves * 32):
            if should_stop and should_stop():
                return {'board': list(original), 'changed': False, 'moves': 0, 'termination': 'stopped'}
            if not self.mutable or performed >= moves:
                break
            cell = rng.choice(self.mutable)
            trial = list(working)
            kind = sum(v == 0 for v in self.problem.faces[working[cell]])
            pool = groups[kind]
            if len(pool) > 1 and rng.random() < .8:
                other = rng.choice(pool)
                if other == cell:
                    continue
                left = rotations(cell, working[other]//4)
                right = rotations(other, working[cell]//4)
                if not left or not right:
                    continue
                trial[cell] = rng.choice(left)
                trial[other] = rng.choice(right)
            else:
                options = [s for s in rotations(cell, working[cell]//4) if s != working[cell]]
                if not options:
                    continue
                trial[cell] = rng.choice(options)
            key = tuple(trial)
            if key in visited or key in forbidden:
                continue
            visited.add(key)
            working = trial
            performed += 1
        if not check_solution(self.problem, working):
            raise RuntimeError('Diversification violated a piece, domain, frame or clue')
        return {'board': working, 'changed': tuple(working) != original,
                'moves': performed, 'termination': 'perturbed' if performed else 'no_legal_move'}
