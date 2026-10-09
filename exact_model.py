"""Exact edge-matching task definitions shared by independently implemented engines.

States are 4*zero-based physical piece + clockwise rotation; faces are NESW.
An explicit edge subset defines a regional task, never a whole-puzzle proof.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import random

@dataclass(frozen=True)
class ExactProblem:
    size: int
    faces: tuple[tuple[int, int, int, int], ...]
    fixed: dict[int, int]
    domains: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int, int, int], ...] | None = None

    def __post_init__(self):
        if type(self.size) is not int or not 1 <= self.size <= 16:
            raise ValueError('Puzzle size must be 1..16')
        n = self.size * self.size
        if len(self.faces) != 4*n or any(len(f)!=4 or any(type(c) is not int or c<0 or c>255 for c in f) for f in self.faces):
            raise ValueError('Expected four NESW states per physical piece')
        if len(self.domains)!=n or any(any(type(s) is not int or not 0<=s<4*n for s in d) for d in self.domains):
            raise ValueError('Invalid cell domains')
        if any(type(c) is not int or not 0<=c<n or type(s) is not int or not 0<=s<4*n for c,s in self.fixed.items()):
            raise ValueError('Invalid fixed placement')
        if self.edges is not None:
            legal = set(grid_edges(self.size))
            if any(len(e)!=4 or any(type(v) is not int for v in e) for e in self.edges) or len(set(self.edges)) != len(self.edges) or any(e not in legal for e in self.edges):
                raise ValueError('Edges must be distinct canonical orthogonal adjacencies')

def grid_edges(size):
    return tuple((c,c+1,1,3) for c in range(size*size) if c%size<size-1) + tuple((c,c+size,2,0) for c in range(size*(size-1)))

def constraint_edges(problem):
    return grid_edges(problem.size) if problem.edges is None else problem.edges

def frame_ok(size, cell, face):
    outside = (cell<size, cell%size==size-1, cell>=size*(size-1), cell%size==0)
    return all((color==0)==border for color,border in zip(face,outside))

def initial_domains(problem):
    fixed_pieces = {state//4 for state in problem.fixed.values()}
    return tuple(tuple(dict.fromkeys(s for s in domain if frame_ok(problem.size,c,problem.faces[s])
                  and (s==problem.fixed[c] if c in problem.fixed else s//4 not in fixed_pieces)))
                 for c,domain in enumerate(problem.domains))

def make_problem(size, faces, fixed=None, domains=None, edges=None):
    faces = tuple(tuple(f) for f in faces)
    fixed = dict(fixed or {})
    domains = tuple(tuple(d) for d in domains) if domains is not None else tuple(tuple(range(len(faces))) for _ in range(size*size))
    problem = ExactProblem(size, faces, fixed, domains, None if edges is None else tuple(tuple(e) for e in edges))
    return ExactProblem(size, faces, fixed, initial_domains(problem), problem.edges)

def check_solution(problem, board):
    try:
        if len(board)!=problem.size**2 or any(type(s) is not int or not 0<=s<len(problem.faces) for s in board): return False
        if len({s//4 for s in board})!=len(board): return False
        if any(s not in problem.domains[c] or not frame_ok(problem.size,c,problem.faces[s]) for c,s in enumerate(board)): return False
        if any(board[c]!=s for c,s in problem.fixed.items()): return False
        return all(problem.faces[board[a]][sa]==problem.faces[board[b]][sb] for a,b,sa,sb in constraint_edges(problem))
    except (TypeError, IndexError, KeyError):
        return False

def problem_hash(problem):
    value = {'size':problem.size,'faces':problem.faces,'fixed':sorted(problem.fixed.items()),'domains':problem.domains,'edges':constraint_edges(problem)}
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()

def from_bundle(bundle, *, board=None, free_cells=None):
    if free_cells is None:
        return make_problem(16,bundle.oriented_edges,bundle.fixed_clues)
    if board is None:
        raise ValueError('Regional tasks require a validated complete seed board')
    from validator import validate_board
    if not validate_board(board,bundle)['valid']:
        raise ValueError('Illegal regional seed board')
    free = set(free_cells)
    if not free or any(type(c) is not int or not 0<=c<256 for c in free):
        raise ValueError('Invalid free-cell region')
    fixed = {c:s for c,s in enumerate(board) if c not in free}
    fixed.update(bundle.fixed_clues)
    edges = tuple(e for e in grid_edges(16) if e[0] in free or e[1] in free)
    return make_problem(16,bundle.oriented_edges,fixed,edges=edges)

def generated_problem(size, colors, seed, clue_count=0):
    """Known-solvable synthetic correctness/benchmark case; not an EII solve."""
    if type(size) is not int or not 1<=size<=16: raise ValueError('Puzzle size must be 1..16')
    if type(colors) is not int or not 1<=colors<=255: raise ValueError('Invalid color count')
    rng=random.Random(seed)
    target=[[0]*4 for _ in range(size*size)]
    for a,b,sa,sb in grid_edges(size):
        color=rng.randint(1,colors);target[a][sa]=target[b][sb]=color
    order=list(range(size*size));rng.shuffle(order)
    faces=[];answer=[None]*(size*size)
    for piece,cell in enumerate(order):
        shift=rng.randrange(4)
        base=tuple(target[cell][(side-shift)%4] for side in range(4))
        states=[tuple(base[(side-rot)%4] for side in range(4)) for rot in range(4)]
        faces.extend(states)
        answer[cell]=4*piece+states.index(tuple(target[cell]))
    if not 0<=clue_count<=size*size: raise ValueError('Invalid clue count')
    fixed={c:answer[c] for c in rng.sample(list(range(size*size)),clue_count)}
    return make_problem(size,faces,fixed),answer
