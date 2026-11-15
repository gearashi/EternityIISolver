"""Optional temperature-guided exchanges of complete GPU replica states.

This is a search heuristic. The move kernel uses biased proposals, so this
module makes no claim about equilibrium sampling or detailed balance.
"""
from __future__ import annotations
import math
import random
import numpy as np


def exchange_plan(scores, temperatures, *, seed, round_index=0):
    """Disjoint adjacent-temperature pairs; higher scores favor colder replicas."""
    values = list(scores)
    temps = list(temperatures)
    if not 1 <= len(values) <= 32768 or len(temps) != len(values):
        raise ValueError('Exchange needs 1..32768 matching scores and temperatures')
    if any(type(s) is not int or not 0 <= s <= 480 for s in values):
        raise ValueError('Scores must be integers in 0..480')
    if any(type(t) not in (int, float) or not math.isfinite(t) or t < 1e-12 for t in temps):
        raise ValueError('Exchange temperatures must be finite and at least 1e-12')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32')
    if type(round_index) is not int or not 0 <= round_index < 2**53:
        raise ValueError('Invalid exchange round')
    rng = random.Random((seed << 53) | round_index)
    order = sorted(range(len(values)), key=lambda i: (temps[i], i))
    permutation = list(range(len(values)))
    pairs = []
    accepted = 0
    for k in range(round_index % 2, len(order) - 1, 2):
        a, b = order[k:k+2]
        # Avoid division overflow for arbitrarily small positive temperatures.
        if values[a] == values[b] or temps[a] == temps[b]:
            log_weight = 0.0
        else:
            log_weight = (values[b] - values[a]) * (1.0 / temps[a] - 1.0 / temps[b])
        swap = log_weight >= 0 or math.log(max(rng.random(), 1e-300)) < log_weight
        pairs.append((a, b, swap))
        if swap:
            permutation[a], permutation[b] = b, a
            accepted += 1
    return {'permutation': permutation, 'pairs': pairs, 'attempts': len(pairs),
            'accepted': accepted, 'round_index': round_index}


def exchange_engine(engine, *, seed, round_index=0):
    """Validate, stage and exchange working states; retain every replica's best.

    Board/score/inverse-position arrays move together. RNG streams, temperatures
    and strategy arms remain attached to their replica slots. No new search
    proposal is counted for routing an already existing board. The explicit
    seed/round makes this independent of the move kernel's RNG streams.
    """
    from validator import validate_board
    if type(engine.n) is not int or not 1 <= engine.n <= 32768:
        raise ValueError('Exchange supports at most 32768 replicas')
    boards = engine.boards.get()
    scores = engine.scores.get()
    best = engine.bestboards.get()
    bestscores = engine.bestscores.get()
    positions = engine.positions.get()
    temps = engine.temps.get()
    expected = {'boards': (boards, (256, engine.n), np.int16),
                'best': (best, (256, engine.n), np.int16),
                'scores': (scores, (engine.n,), np.int32),
                'bestscores': (bestscores, (engine.n,), np.int32),
                'positions': (positions, (256, engine.n), np.int16)}
    for name, (array, shape, dtype) in expected.items():
        if array.shape != shape or array.dtype != dtype:
            raise ValueError('Invalid exchange array: ' + name)
    plan = exchange_plan([int(s) for s in scores], [float(t) for t in temps],
                         seed=seed, round_index=round_index)
    for values, reported in ((boards, scores), (best, bestscores)):
        for board, value in zip(values.T, reported):
            checked = validate_board(board, engine.bundle)
            if not checked['valid'] or checked['score'] != int(value):
                raise ValueError('Exchange rejected an invalid board or score')
    if not np.array_equal(positions, np.argsort(boards.T // 4, axis=1).astype(np.int16).T):
        raise ValueError('Exchange rejected inconsistent inverse positions')
    perm = plan['permutation']
    incoming = boards[:, perm].copy()
    incoming_scores = scores[perm].copy()
    incoming_positions = positions[:, perm].copy()
    improved = incoming_scores > bestscores
    kept = best.copy()
    kept_scores = bestscores.copy()
    kept[:, improved] = incoming[:, improved]
    kept_scores[improved] = incoming_scores[improved]
    # Stage all allocations first: an allocation failure leaves engine intact.
    staged = {name: engine.xp.asarray(value) for name, value in (
        ('boards', incoming), ('scores', incoming_scores),
        ('positions', incoming_positions), ('bestboards', kept), ('bestscores', kept_scores))}
    for name, value in staged.items():
        setattr(engine, name, value)
    return {**plan, 'best_promotions': int(improved.sum()),
            'global_best': int(kept_scores.max()), 'new_move_proposals': 0}
