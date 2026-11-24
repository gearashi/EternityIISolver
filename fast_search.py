"""Validated offline adapter for the compiled fast-search helper.

The helper (scripts/fast_dfs.cpp, packaged as native/eternity-fast) is a
table-driven row-major backtracker. Two engines use it:

  strict  every shared edge must match; threads split the top of one search
          tree, so a run is a resumable exhaustive search for all matches.
  score   a budget of mismatching edges is allowed late in the scan and a
          color-depletion heuristic is applied; randomized restarts look for
          complete boards. Each board found tightens the budget so the next
          one must score higher. Heuristic only: a score run never shows
          that a better board does not exist.

The helper is never trusted. Every board it writes is checked here against the
problem before it is reported, and the worker validates it again.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from app_paths import resource_root
from exact_model import frame_ok, grid_edges, initial_domains

HELPER_VERSION = 'eternity-fast-1'
MAX_WORKERS = 64
MAX_COLOR = 31
MODES = ('strict', 'score')
UNBOUNDED = 1.0e9
# SHA-256 of the official 256 pieces as written to the helper (see _piece_text).
OFFICIAL_PIECES_SHA256 = '1f0ec5db754c3ac94b95656f303745b85347c14adff33a0fda8f762ca79024ae'


def _concave_schedule(sides, fraction, first, last, scale, step=2):
    """Points (cell, minimum depletion-color sides used): fast early, flat by cell `last`."""
    top = 1 - math.exp(-(last - first) / scale)
    return ((first, 0),) + tuple((cell, round(sides * fraction * (1 - math.exp(-(cell - first) / scale)) / top, 2))
                                 for cell in range(first + step, last + 1, step))


# Tuned on the official five-clue puzzle; docs/FAST_SEARCH.md describes how.
# breaks[k] is the first cell index (row-major) where the (k+1)-th mismatching
# edge may be taken; this is the loosest budget, the ladder's first step.
# The 256 pieces have 146 sides in colors 2, 18 and 20; at least 80% of them
# must be used by cell 160, most of them early.
OFFICIAL_SCORE_PROFILE = {
    'name': 'official-ladder-1',
    'breaks': (190, 196, 204, 206, 210, 215, 219, 220, 227, 229, 231, 232, 234, 235, 236, 237, 241, 243, 244, 245, 248, 249),
    'depletion_colors': (2, 18, 20),
    'depletion_schedule': _concave_schedule(146, 0.8, 16, 160, 90),
    'restart_nodes': 200_000_000,
}


def helper_path():
    name = 'eternity-fast.exe' if os.name == 'nt' else 'eternity-fast'
    packaged = resource_root() / 'native' / name
    if packaged.is_file():
        return packaged
    # A source checkout after python scripts/build_fast_helper.py.
    return Path(__file__).resolve().parent / 'build' / 'native-helper' / name


def _piece_text(problem):
    cells = problem.size ** 2
    rows = []
    for piece in range(cells):
        base = problem.faces[4 * piece]
        for rotation in range(4):
            if problem.faces[4 * piece + rotation] != tuple(base[(side - rotation) % 4] for side in range(4)):
                raise ValueError('Fast search needs states that are clockwise rotations of each piece')
        if max(base) > MAX_COLOR:
            raise ValueError('Fast search supports colors 0..31')
        rows.append(f'{base[0]} {base[2]} {base[3]} {base[1]}\n')  # up down left right
    return ''.join(rows)


def _whole_puzzle(problem):
    if problem.edges is not None and set(problem.edges) != set(grid_edges(problem.size)):
        return False
    plain = type(problem)(problem.size, problem.faces, problem.fixed,
                          tuple(tuple(range(len(problem.faces))) for _ in range(problem.size ** 2)))
    return tuple(map(set, problem.domains)) == tuple(map(set, initial_domains(plain)))


def score_board(problem, board):
    """Return the number of matching shared edges, or None for an illegal board."""
    cells = problem.size ** 2
    try:
        if len(board) != cells or any(type(state) is not int or not 0 <= state < 4 * cells for state in board):
            return None
        if len({state // 4 for state in board}) != cells:
            return None
        if any(not frame_ok(problem.size, cell, problem.faces[state]) for cell, state in enumerate(board)):
            return None
        if any(board[cell] != state for cell, state in problem.fixed.items()):
            return None
        return sum(problem.faces[board[a]][sa] == problem.faces[board[b]][sb] for a, b, sa, sb in grid_edges(problem.size))
    except (TypeError, IndexError):
        return None


def default_score_profile(problem):
    """The tuned profile for the official pieces; a plain late-break rule otherwise."""
    cells = problem.size ** 2
    if problem.size == 16 and hashlib.sha256(_piece_text(problem).encode()).hexdigest() == OFFICIAL_PIECES_SHA256:
        return dict(OFFICIAL_SCORE_PROFILE)
    return {'name': 'generic-late-breaks', 'breaks': tuple(range(cells * 7 // 10, cells, 2)),
            'depletion_colors': (), 'depletion_schedule': (), 'restart_nodes': 50_000_000}


def _profile_arguments(profile, cells, budget):
    """Helper arguments for a profile with only the first `budget` mismatches allowed.

    The profile lists its loosest schedule. A smaller budget keeps evenly
    spaced entries of that list, so the allowed mismatches stay spread over the
    same late part of the scan.
    """
    breaks = tuple(profile.get('breaks', ()))
    colors = tuple(profile.get('depletion_colors', ()))
    schedule = tuple(tuple(point) for point in profile.get('depletion_schedule', ()))
    restart = profile.get('restart_nodes', 0)
    if (not breaks or any(type(d) is not int or not 0 <= d <= cells for d in breaks) or list(breaks) != sorted(breaks)
            or len(breaks) > 255 or type(restart) is not int or restart < 0
            or type(budget) is not int or not 0 <= budget <= len(breaks)
            or any(type(c) is not int or not 1 <= c <= MAX_COLOR for c in colors) or bool(colors) != bool(schedule)
            or any(len(point) != 2 or type(point[0]) is not int or not 0 <= point[0] <= cells
                   or type(point[1]) not in (int, float) or not math.isfinite(point[1]) or point[1] < 0 for point in schedule)
            or [point[0] for point in schedule] != sorted({point[0] for point in schedule})
            or [point[1] for point in schedule] != sorted(point[1] for point in schedule)):
        raise ValueError('Invalid fast-search score profile')
    result = ['--restart-nodes', str(restart)]
    if budget:
        result += ['--breaks', ','.join(str(breaks[k * len(breaks) // budget]) for k in range(budget))]
    if colors:
        result += ['--hcolors', ','.join(map(str, colors)),
                   '--hsched', ','.join(f'{cell}:{count:g}' for cell, count in schedule)]
    return result


def _read_json(path, limit=262144):
    """Small helper files are replaced while we poll; a missing or partial file is retried later."""
    try:
        with Path(path).open('rb') as stream:
            payload = stream.read(limit + 1)
        return json.loads(payload) if len(payload) <= limit else None
    except (OSError, ValueError):
        return None


def _helper_board(problem, path):
    """Return (score, board) for a helper board file that passes independent checks, else None."""
    record = _read_json(path)
    if not isinstance(record, dict) or not isinstance(record.get('board'), list):
        return None
    score = score_board(problem, record['board'])
    if score is None or record.get('claimed_score') != score:
        raise RuntimeError('Fast-search helper wrote a board that fails independent validation')
    return score, list(record['board'])


def has_saved_search(state_dir):
    return any(Path(state_dir).glob('ckpt_t*.txt'))


def _clear_checkpoints(state_dir):
    for path in Path(state_dir).glob('ckpt_t*.txt'):
        path.unlink(missing_ok=True)


def _run_helper(command, state_dir, seconds, should_stop, tick, poll_seconds):
    """Run one helper process to its exit; tick() may return True to end it early. Returns (exit code, final status)."""
    stop_path, log_path, status_path = state_dir / 'STOP', state_dir / 'helper-log.txt', state_dir / 'status.json'
    stop_path.unlink(missing_ok=True)
    status_path.unlink(missing_ok=True)
    if seconds < UNBOUNDED:
        command = command + ['--seconds', f'{max(seconds, 0.001):.3f}']
    with log_path.open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            stopping = False
            while process.poll() is None:
                wants_stop = tick(_read_json(status_path))
                if not stopping and (wants_stop or should_stop()):
                    stop_path.write_text('stop\n', encoding='ascii')
                    stopping = True
                time.sleep(poll_seconds)
        finally:
            if process.poll() is None:
                # Closing stdin trips the helper's guard: it saves its checkpoints and exits.
                try:
                    process.stdin.close()
                    process.wait(timeout=20)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait()
            try:
                process.stdin.close()
            except OSError:
                pass
    stop_path.unlink(missing_ok=True)
    code = process.returncode
    if code not in (0, 3, 4):
        detail = ''
        try:
            detail = log_path.read_text(encoding='utf-8', errors='replace').strip().splitlines()[-1][:300]
        except (OSError, IndexError):
            pass
        if 'checkpoint' in detail:
            raise ValueError(f'Saved fast search cannot be continued with these settings ({detail})')
        raise RuntimeError(f'Fast-search helper failed with exit code {code}: {detail}')
    status = _read_json(status_path)
    if not isinstance(status, dict) or status.get('final') is not True:
        raise RuntimeError('Fast-search helper ended without its final report')
    return code, status


def solve(problem, seconds, seed=1, workers=1, should_stop=None, on_progress=None, *, state_dir, mode='strict',
          reference=None, on_improvement=None, on_board=None, profile=None, helper=None, prefix=4, poll_seconds=0.25, max_nodes=None):
    """Run the compiled search in state_dir, continuing its checkpoints when present.

    Outcomes. strict: solved, infeasible (search tree exhausted), timeout or
    stopped. score: solved, improved or unchanged (the current budget's search
    is exhausted, possible on small puzzles only), timeout or stopped. A score
    run calls on_improvement for each validated board that beats the reference.
    on_board receives every complete score candidate, including ties and lower scores.
    max_nodes ends the run (as a timeout) once every worker has tried that many
    placements in total; it exists for reproducible tests and benchmarks.
    """
    if mode not in MODES:
        raise ValueError('Fast search mode must be strict or score')
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f'Fast search permits 1..{MAX_WORKERS} workers')
    if type(seed) is not int or not 0 <= seed < 2 ** 64:
        raise ValueError('seed must be a uint64 integer')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if max_nodes is not None and (type(max_nodes) is not int or max_nodes < 1):
        raise ValueError('max_nodes must be a positive integer')
    if not _whole_puzzle(problem):
        raise ValueError('Fast search supports whole puzzles only (all grid edges, unrestricted domains)')
    cells = problem.size ** 2
    edges = len(grid_edges(problem.size))
    pieces = _piece_text(problem)
    should_stop = should_stop or (lambda: False)
    reference_score = -1
    if reference is not None:
        reference_score = score_board(problem, reference)
        if reference_score is None:
            raise ValueError('Fast search reference board is illegal')
    counters = {'branches': 0, 'max_depth': 0, 'complete_boards': 0, 'restarts': 0, 'nodes_per_second': 0}
    result = {'board': None, 'mode': mode, 'workers': workers, 'engine_best_score': None}
    if mode == 'score':
        profile = dict(default_score_profile(problem) if profile is None else profile)
        result['profile'] = str(profile.get('name', 'custom'))[:80]
        _profile_arguments(profile, cells, len(profile.get('breaks', ())))  # validate before any work
    elif type(prefix) is not int or not 1 <= prefix <= cells:
        raise ValueError('Invalid partition prefix depth')
    if seconds == 0 or should_stop():
        return dict(result, **counters, outcome='stopped' if should_stop() else 'timeout')
    helper = Path(helper) if helper is not None else helper_path()
    if not helper.is_file():
        raise RuntimeError('The compiled fast-search helper is missing. Release archives include it; '
                           'from a source checkout run: python scripts/build_fast_helper.py')
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    piece_path, best_path, ladder_path = state_dir / 'pieces.txt', state_dir / 'best.json', state_dir / 'ladder.json'
    piece_path.write_text(pieces, encoding='ascii', newline='\n')
    resumed = has_saved_search(state_dir)
    result['resumed'] = resumed
    base = [str(helper), '--pieces', str(piece_path), '--size', str(problem.size), '--out', str(state_dir),
            '--threads', str(workers), '--seed', str(seed), '--report', '0.5', '--checkpoint', '30',
            '--stop-file', str(state_dir / 'STOP'), '--stdin-guard']
    for cell, state in sorted(problem.fixed.items()):
        base += ['--clue', f'{cell}:{state}']
    if max_nodes is not None:
        base += ['--max-nodes', str(max_nodes)]
    started = time.monotonic()
    remaining = lambda: seconds if seconds >= UNBOUNDED else seconds - (time.monotonic() - started)
    earlier = {'branches': 0, 'complete_boards': 0, 'restarts': 0}  # finished score stages

    def progress(status, extra, final=False):
        if isinstance(status, dict):
            for source, target in (('nodes', 'branches'), ('complete_boards', 'complete_boards'), ('restarts', 'restarts')):
                value = status.get(source)
                if type(value) is int and value >= 0:
                    counters[target] = earlier[target] + value
            for key in ('max_depth', 'nodes_per_second'):
                value = status.get(key)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    counters[key] = max(counters[key], value) if key == 'max_depth' else value
        if on_progress is not None:
            on_progress(dict(counters, **extra, phase='finished' if final else 'searching'))

    if mode == 'strict':
        best_path.unlink(missing_ok=True)
        command = base + ['--partition', '--prefix', str(prefix)] + (['--resume'] if resumed else [])
        code, status = _run_helper(command, state_dir, remaining(), should_stop,
                                   lambda live: progress(live, {}), poll_seconds)
        progress(status, {}, final=True)
        result.update(counters, helper_state=status.get('state'))
        if code == 0:
            found = _helper_board(problem, best_path)
            if found is None or found[0] != edges:
                raise RuntimeError('Fast-search helper reported a solution without a fully matching board')
            return dict(result, outcome='solved', board=found[1], score=edges)
        if code == 3:
            if status.get('state') != 'exhausted' or status.get('perfect_boards') != 0:
                raise RuntimeError('Fast-search helper exit status disagrees with its final report')
            scope = {'proof_scope': 'remaining_checkpoint_frontier', 'coverage_verified': False} if resumed else {}
            return dict(result, outcome='infeasible', **scope)
        return dict(result, outcome='stopped' if should_stop() else 'timeout')

    # Score mode: a ladder of mismatch budgets. Each complete board found sets
    # the next budget one below what that board used, so later boards score higher.
    loosest = len(profile['breaks'])
    engine_best, engine_board = -1, None
    saved = _read_json(ladder_path) if resumed else None
    prior = _helper_board(problem, best_path) if best_path.exists() else None
    if (resumed and isinstance(saved, dict) and saved.get('profile') == result['profile'] and saved.get('seed') == seed
            and type(saved.get('budget')) is int and 0 <= saved['budget'] <= loosest):
        budget = saved['budget']
        for key in earlier:
            if type(saved.get(key)) is int and saved[key] >= 0:
                earlier[key] = saved[key]
        if prior is not None:
            engine_best, engine_board = prior
    elif resumed:
        raise ValueError('Saved fast search cannot be continued with these settings (its checkpoint record does not match)')
    else:
        best_path.unlink(missing_ok=True)
        budget = loosest
    improved = False

    if on_board is not None and resumed:
        # Older releases only published improvements above the reference. Recover
        # their retained boards on resume before continuing the same ladder.
        for path in sorted((state_dir / 'boards').glob('*.json')):
            record = _read_json(path)
            if not isinstance(record, dict) or type(record.get('score')) is not int or score_board(problem, record.get('board')) != record['score']:
                raise ValueError(f'Invalid retained fast-search board: {path.name}')
            on_board(record)
        if prior is not None:
            on_board({'board': prior[1], 'score': prior[0]})

    def publish_pending():
        # --keep gives each completion its own file, so two boards arriving
        # between status polls cannot overwrite one another in best.json.
        if on_board is not None:
            for path in state_dir.glob('board_*.json'):
                candidate = _helper_board(problem, path)
                if candidate is None:
                    raise ValueError(f'Unreadable fast-search board: {path.name}')
                on_board({'board': candidate[1], 'score': candidate[0]})
                path.unlink()

    def take_board():
        nonlocal engine_best, engine_board, reference_score, improved
        found = _helper_board(problem, best_path)
        if found is None or found[0] <= engine_best:
            return False
        engine_best, engine_board = found
        digest = hashlib.sha256(json.dumps(engine_board, separators=(',', ':')).encode()).hexdigest()[:16]
        keep = state_dir / 'boards'
        keep.mkdir(exist_ok=True)
        (keep / f'{engine_best}-{digest}.json').write_text(
            json.dumps({'score': engine_best, 'board': engine_board, 'engine': HELPER_VERSION, 'seed': seed}) + '\n', encoding='utf-8')
        if engine_best > reference_score:
            reference_score = engine_best
            improved = True
            if on_improvement is not None:
                on_improvement({'board': engine_board, 'score': engine_best})
        return True

    while True:
        target = edges - budget
        extra = {'target_score': target}
        if engine_best >= 0:
            extra['engine_best_score'] = engine_best
        ladder_path.write_text(json.dumps(dict(earlier, budget=budget, profile=result['profile'], seed=seed)) + '\n', encoding='utf-8')
        command = base + _profile_arguments(profile, cells, budget)
        if on_board is not None:
            command += ['--keep', str(target)]
        if engine_best >= 0:
            command += ['--floor', str(engine_best)]
        if has_saved_search(state_dir):
            command.append('--resume')

        def tick(live):
            publish_pending()
            found = take_board()
            if found:
                extra['engine_best_score'] = engine_best
            progress(live, extra)
            return found  # a board met this target: end the stage and tighten

        code, status = _run_helper(command, state_dir, remaining(), should_stop, tick, poll_seconds)
        publish_pending()
        take_board()
        if engine_best >= 0:
            extra['engine_best_score'] = engine_best
        result.update(engine_best_score=engine_best if engine_best >= 0 else None, break_budget=budget,
                      target_score=target, helper_state=status.get('state'))
        if engine_best == edges:
            progress(status, extra, final=True)
            return dict(result, **counters, outcome='solved', board=engine_board, score=edges)
        if engine_best >= target and not should_stop() and remaining() > 0:
            # Stage met: its checkpoints belong to the old budget, so the next stage starts clean.
            progress(status, extra)
            for key in earlier:
                earlier[key] = counters[key]
            _clear_checkpoints(state_dir)
            budget = min(budget - 1, edges - engine_best - 1)
            continue
        progress(status, extra, final=True)
        result.update(counters)
        if code == 3 and status.get('state') == 'exhausted':
            return dict(result, outcome='improved' if improved else 'unchanged',
                        proof_scope='heuristic_break_schedule', coverage_verified=False)
        return dict(result, outcome='stopped' if should_stop() else 'timeout')
