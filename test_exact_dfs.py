"""Small, independent exact-search correctness checks; no GPU or network."""
import itertools
import hashlib
import json
import math
import random
import unittest
from unittest.mock import patch

import exact_dfs
from exact_dfs import solve
from exact_model import ExactProblem, check_solution, generated_problem, make_problem


def edges(size):
    return [(i, i + 1, 1, 3) for i in range(size * size) if i % size + 1 < size] + [
        (i, i + size, 2, 0) for i in range(size * (size - 1))]


def oracle_domains(problem):
    """Separate scalar boundary/fixed filter for a deliberately tiny oracle."""
    result = []
    for cell, domain in enumerate(problem.domains):
        r, c = divmod(cell, problem.size)
        outside = [r == 0, c == problem.size - 1, r == problem.size - 1, c == 0]
        result.append([state for state in domain
                       if all((problem.faces[state][side] == 0) == outside[side] for side in range(4))
                       and (cell not in problem.fixed or problem.fixed[cell] == state)])
    return result


def exhaustive(problem):
    """Enumerate every tiny candidate tuple, independently of DFS pruning."""
    required = edges(problem.size) if problem.edges is None else problem.edges
    for board in itertools.product(*oracle_domains(problem)):
        if len(set(state // 4 for state in board)) != len(board):
            continue
        if all(problem.faces[board[a]][sa] == problem.faces[board[b]][sb]
               for a, b, sa, sb in required):
            return list(board)
    return None


def reseal(checkpoint):
    """Test deliberately altered, self-consistent data without a secret MAC."""
    body = {k: v for k, v in checkpoint.items() if k != 'checksum'}
    raw = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
    return dict(body, checksum=hashlib.sha256(raw).hexdigest())


def cyclic_unsat():
    # Two disjoint two-edge color cycles cannot form the board's four-edge
    # cycle while using all pieces. Initial arc consistency still has support.
    faces = []
    for right, down in ((1, 2), (2, 1), (3, 4), (4, 3)):
        base = (0, right, down, 0)
        faces.extend(tuple(base[(side - rotation) % 4] for side in range(4)) for rotation in range(4))
    return make_problem(2, faces)


class ExactDfsTests(unittest.TestCase):
    def test_generated_solutions_and_unchanged_inputs(self):
        for size in (1, 2, 3):
            with self.subTest(size=size):
                problem, _ = generated_problem(size, 4, 77, clue_count=1)
                before = (problem.domains, dict(problem.fixed), problem.faces)
                result = solve(problem, seconds=2, seed=41)
                self.assertEqual(result['outcome'], 'solved')
                self.assertTrue(check_solution(problem, result['board']))
                self.assertEqual(before, (problem.domains, problem.fixed, problem.faces))
                self.assertLessEqual(result['build_seconds'], result['elapsed_seconds'])
                self.assertEqual(result['backend'], 'dfs-python')
                self.assertEqual(result['workers'], 1)

    def test_small_domain_cases_agree_with_exhaustive_enumeration(self):
        counts = {'solved': 0, 'infeasible': 0}
        rng = random.Random(622)
        for case in range(32):
            source, known = generated_problem(2, 3, case)
            domains = [tuple(s for s in d if rng.random() < .55 or (case % 2 == 0 and s == known[cell]))
                       for cell, d in enumerate(source.domains)]
            problem = ExactProblem(2, source.faces, {}, tuple(domains))
            expected = exhaustive(problem)
            result = solve(problem, seconds=2, seed=case)
            with self.subTest(case=case):
                outcome = 'infeasible' if expected is None else 'solved'
                self.assertEqual(result['outcome'], outcome)
                counts[outcome] += 1
                if expected is None:
                    self.assertIsNone(result['board'])
                else:
                    self.assertTrue(check_solution(problem, result['board']))
        self.assertGreater(counts['solved'], 0)
        self.assertGreater(counts['infeasible'], 0)

    def test_explicit_edges_do_not_impose_omitted_mismatches(self):
        source, answer = generated_problem(2, 4, 24)
        faces = list(source.faces)
        piece = answer[0] // 4
        base = list(faces[4 * piece])
        side = next(i for i, color in enumerate(base) if color != 0)
        base[side] = 99  # Odd color supply is allowed when its edge is omitted.
        faces[4 * piece:4 * piece + 4] = [tuple(base[(s - rot) % 4] for s in range(4)) for rot in range(4)]
        fixed = dict(enumerate(answer))
        required = [e for e in edges(2) if faces[answer[e[0]]][e[2]] == faces[answer[e[1]]][e[3]]]
        self.assertLess(len(required), len(edges(2)))
        full = make_problem(2, faces, fixed)
        regional = make_problem(2, faces, fixed, edges=required)
        self.assertIsNone(exhaustive(full))
        self.assertEqual(solve(full, seconds=2)['outcome'], 'infeasible')
        result = solve(regional, seconds=2)
        self.assertEqual(result['outcome'], 'solved')
        self.assertEqual(result['board'], answer)
        self.assertTrue(check_solution(regional, result['board']))

    def test_hall_contradiction_without_missing_or_unique_location_pieces(self):
        source, _ = generated_problem(4, 1, 31)
        border_cells = [c for c in range(16) if sum((c < 4, c % 4 == 3, c >= 12, c % 4 == 0)) == 1]
        border_pieces = sorted({s // 4 for s in source.domains[border_cells[0]]})
        domains = list(source.domains)
        for cell in border_cells[:3]:
            domains[cell] = tuple(s for s in domains[cell] if s // 4 in border_pieces[:2])
        # Every piece occurs in multiple cells, yet three cells share only two
        # possible pieces. Edges are omitted to isolate all-different pruning.
        problem = make_problem(4, source.faces, domains=domains, edges=())
        occurrences = [sum(any(s // 4 == p for s in d) for d in problem.domains) for p in range(16)]
        self.assertTrue(all(count > 1 for count in occurrences))
        result = solve(problem, seconds=2)
        self.assertEqual(result['outcome'], 'infeasible')
        self.assertEqual(result['branches'], 0)
        self.assertGreater(result['matching_failures'], 0)

    def test_unfiltered_domains_and_duplicate_fixed_piece(self):
        source, _ = generated_problem(2, 2, 19)
        raw = ExactProblem(2, source.faces, {}, (tuple(range(16)),) * 4)
        result = solve(raw, seconds=2)
        self.assertEqual(result['outcome'], 'solved')
        self.assertTrue(check_solution(raw, result['board']))
        legal = oracle_domains(raw)
        piece = legal[0][0] // 4
        fixed = {0: next(s for s in legal[0] if s // 4 == piece),
                 1: next(s for s in legal[1] if s // 4 == piece)}
        impossible = ExactProblem(2, source.faces, fixed, raw.domains)
        self.assertEqual(solve(impossible, seconds=2)['outcome'], 'infeasible')

    def test_timeout_and_cancellation_are_not_infeasibility(self):
        problem, _ = generated_problem(3, 3, 13)
        timed = solve(problem, seconds=0)
        self.assertEqual(timed['outcome'], 'timeout')
        self.assertIsNone(timed['board'])
        self.assertEqual(solve(problem, seconds=2, should_stop=lambda: True)['outcome'], 'stopped')
        calls = 0
        def stop_during_build():
            nonlocal calls
            calls += 1
            return calls >= 4
        stopped = solve(problem, seconds=2, should_stop=stop_during_build)
        self.assertEqual(stopped['outcome'], 'stopped')
        self.assertEqual(calls, 4)
        self.assertIsNone(stopped['board'])

    def test_final_progress_and_reproducible_unbounded_order(self):
        problem, _ = generated_problem(3, 2, 71)
        progress = []
        first = solve(problem, seconds=2, seed=85, on_progress=progress.append)
        second = solve(problem, seconds=2, seed=85)
        self.assertEqual(first['outcome'], 'solved')
        self.assertEqual(progress[-1], first)
        self.assertEqual(first['board'], second['board'])
        self.assertEqual(first['branches'], second['branches'])

    def test_invalid_budget_and_worker_requests(self):
        problem, _ = generated_problem(1, 1, 1)
        for seconds in (-1, math.inf, math.nan, True, '1'):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                solve(problem, seconds=seconds)
        for workers in (0, 2, True):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                solve(problem, workers=workers)

    def test_repeated_tiny_time_budgets_preserve_sat_and_unsat_frontiers(self):
        sat, _ = generated_problem(3, 1, 29)
        unsat = cyclic_unsat()
        self.assertIsNone(exhaustive(unsat))
        cases = [(sat, None), (unsat, None),
                 (sat, [[0] * 9, list(reversed(range(9)))]),
                 (unsat, [[0] * 4, [15] * 4])]
        for problem, hints in cases:
            fresh = solve(problem, seconds=2, seed=123, hints=hints)
            self.assertGreater(fresh['branches'], 1)
            checkpoint = None
            now = [100.0]
            for iteration in range(64):
                prior_branches = 0 if checkpoint is None else checkpoint['counters']['branches']
                advanced = [False]
                def expire_after_next_branch(snapshot):
                    if snapshot['counters']['branches'] > prior_branches and not advanced[0]:
                        advanced[0] = True
                        now[0] += 1
                # Deterministic clock expiry avoids machine-speed assumptions.
                # Each 1ms budget expires just after one additional branch.
                with patch('exact_dfs.time.perf_counter', side_effect=lambda: now[0]):
                    result = solve(problem, seconds=.001, seed=123, checkpoint=checkpoint, hints=hints,
                                   on_checkpoint=expire_after_next_branch, checkpoint_interval=0)
                if result['outcome'] not in ('stopped', 'timeout'):
                    break
                self.assertEqual(result['outcome'], 'timeout')
                checkpoint = json.loads(json.dumps(result['checkpoint']))
                self.assertLess(len(json.dumps(checkpoint)), exact_dfs.MAX_CHECKPOINT_BYTES)
            else:
                self.fail('Repeated saved frontiers failed to finish the tiny case')
            self.assertGreater(iteration, 1)
            self.assertEqual(result['outcome'], fresh['outcome'])
            self.assertEqual(result['board'], fresh['board'])
            self.assertEqual(result['branches'], fresh['branches'])
            self.assertEqual(result['conflicts'], fresh['conflicts'])
            self.assertTrue(result['resumed'])
            self.assertFalse(result['coverage_verified'])
            self.assertEqual(result['proof_scope'], 'remaining_checkpoint_frontier')
            self.assertEqual(result['solution_verified'], result['outcome'] == 'solved')
            self.assertIsNone(result['checkpoint'])

    def test_interruption_during_ordering_preserves_random_state(self):
        problem, _ = generated_problem(3, 1, 55)
        fresh = solve(problem, seconds=2, seed=61)
        original = exact_dfs._Search.check
        interrupted = [False]
        def interrupt_after_shuffle(search):
            if search.stack and search.stack[-1]['phase'] == 'pending' and search.rng.getstate() != search.committed_rng:
                interrupted[0] = True
                raise exact_dfs._Interrupted('stopped')
            return original(search)
        with patch.object(exact_dfs._Search, 'check', interrupt_after_shuffle):
            stopped = solve(problem, seconds=2, seed=61)
        self.assertTrue(interrupted[0])
        self.assertEqual(stopped['outcome'], 'stopped')
        resumed = solve(problem, seconds=2, seed=61, checkpoint=stopped['checkpoint'])
        self.assertEqual(resumed['board'], fresh['board'])
        self.assertEqual(resumed['branches'], fresh['branches'])

    def test_interruption_during_propagation_repeats_whole_pending_node(self):
        problem, _ = generated_problem(3, 8, 144, clue_count=1)
        fresh = solve(problem, seconds=2, seed=8)
        stop = [False]
        def observe(snapshot):
            stop[0] |= snapshot['counters']['propagations'] > 0
        stopped = solve(problem, seconds=2, seed=8, should_stop=lambda: stop[0],
                        on_checkpoint=observe, checkpoint_interval=0)
        self.assertEqual(stopped['outcome'], 'stopped')
        resumed = solve(problem, seconds=2, seed=8, checkpoint=stopped['checkpoint'])
        self.assertEqual(resumed['outcome'], fresh['outcome'])
        self.assertEqual(resumed['board'], fresh['board'])
        self.assertEqual(resumed['branches'], fresh['branches'])

    def test_checkpoint_hash_shape_seed_and_semantic_validation(self):
        problem, _ = generated_problem(3, 1, 81)
        stop = [False]
        def observe(snapshot):
            stop[0] = snapshot['counters']['branches'] >= 1
        saved = solve(problem, seconds=2, seed=9, should_stop=lambda: stop[0],
                      on_checkpoint=observe, checkpoint_interval=0)['checkpoint']
        self.assertEqual(saved['stack'][0]['phase'], 'branch')
        with self.assertRaises(ValueError):
            solve(problem, seed=10, checkpoint=saved)
        other, _ = generated_problem(3, 2, 82)
        with self.assertRaises(ValueError):
            solve(other, seed=9, checkpoint=saved)
        altered = json.loads(json.dumps(saved))
        altered['counters']['branches'] += 1
        with self.assertRaisesRegex(ValueError, 'checksum'):
            solve(problem, seed=9, checkpoint=altered)
        for mutate in (
            lambda c: c.update(extra=True),
            lambda c: c.update(version=exact_dfs.CHECKPOINT_VERSION + 1),
            lambda c: c.update(stack=c['stack'] * 20),
            lambda c: c['stack'][0].update(next=-1),
            lambda c: c['stack'][0]['candidates'].__setitem__(0, True),
            lambda c: c['rng_state'][1].__setitem__(-1, 625),
            lambda c: c['counters'].update(branches=True),
        ):
            altered = json.loads(json.dumps(saved))
            mutate(altered)
            with self.subTest(checkpoint=altered['stack'][:1]), self.assertRaises(ValueError):
                solve(problem, seed=9, checkpoint=reseal(altered))
        altered = json.loads(json.dumps(saved))
        choices = altered['stack'][0]['candidates']
        choices[0] = next(s for s in range(len(problem.faces)) if s not in choices)
        with self.assertRaisesRegex(ValueError, 'partition'):
            solve(problem, seed=9, checkpoint=reseal(altered))

    def test_cursor_claim_is_not_a_whole_problem_proof(self):
        problem, _ = generated_problem(2, 1, 19)
        stop = [False]
        def observe(snapshot):
            stop[0] = snapshot['counters']['branches'] > 0
        saved = solve(problem, seconds=2, seed=2, should_stop=lambda: stop[0],
                      on_checkpoint=observe, checkpoint_interval=0)['checkpoint']
        # A self-consistent cursor can claim all siblings were completed. It
        # is not authenticated evidence of that past work, even with checksum.
        saved['stack'] = saved['stack'][:1]
        saved['stack'][0]['next'] = len(saved['stack'][0]['candidates'])
        result = solve(problem, seconds=2, seed=2, checkpoint=reseal(saved))
        self.assertEqual(result['outcome'], 'infeasible')
        self.assertIsNotNone(exhaustive(problem))
        self.assertFalse(result['coverage_verified'])
        self.assertEqual(result['proof_scope'], 'remaining_checkpoint_frontier')

    def test_cancelled_restore_returns_original_frontier(self):
        problem, _ = generated_problem(3, 1, 98)
        stop = [False]
        def observe(snapshot):
            stop[0] = snapshot['counters']['branches'] > 1
        saved = solve(problem, seconds=2, seed=5, should_stop=lambda: stop[0],
                      on_checkpoint=observe, checkpoint_interval=0)['checkpoint']
        again = solve(problem, seconds=2, seed=5, checkpoint=saved, should_stop=lambda: True)
        self.assertEqual(again['outcome'], 'stopped')
        self.assertEqual(again['checkpoint'], saved)
        resumed = solve(problem, seconds=2, seed=5, checkpoint=again['checkpoint'])
        self.assertEqual(resumed['outcome'], 'solved')

    def test_invalid_checkpoint_interval(self):
        problem, _ = generated_problem(1, 1, 1)
        for interval in (-1, math.inf, math.nan, True, '1'):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                solve(problem, checkpoint_interval=interval)

    def test_misleading_hints_agree_with_exhaustive_small_puzzles(self):
        cases = [cyclic_unsat()]
        rng = random.Random(105)
        for seed in range(12):
            source, _ = generated_problem(2, 3, seed)
            domains = tuple(tuple(s for s in d if rng.random() < .75) for d in source.domains)
            cases.append(ExactProblem(2, source.faces, {}, domains))
        for index, problem in enumerate(cases):
            expected = 'solved' if exhaustive(problem) is not None else 'infeasible'
            hints = [[0] * 4, [15] * 4, [rng.randrange(16) for _ in range(4)]]
            with self.subTest(case=index):
                result = solve(problem, seconds=2, seed=19, hints=hints)
                self.assertEqual(result['outcome'], expected)
                self.assertEqual(result['hint_count'], 3)
                if expected == 'solved':
                    self.assertTrue(check_solution(problem, result['board']))

    def test_hint_votes_change_tied_order_without_fixing_placements(self):
        problem, _ = generated_problem(2, 1, 14)
        baseline = solve(problem, seconds=2, seed=33)
        legal = oracle_domains(problem)
        target = [next(s for s in legal[cell] if s // 4 == baseline['board'][3 - cell] // 4) for cell in range(4)]
        self.assertTrue(check_solution(problem, target))
        self.assertNotEqual(target, baseline['board'])
        guided = solve(problem, seconds=2, seed=33, hints=[target, target])
        self.assertEqual(guided['board'], target)
        misleading = solve(problem, seconds=2, seed=33, hints=[[0] * 4] * 64)
        self.assertEqual(misleading['outcome'], 'solved')
        self.assertTrue(check_solution(problem, misleading['board']))

    def test_empty_hints_preserve_default_order_and_identity(self):
        problem, _ = generated_problem(3, 2, 91)
        default = solve(problem, seconds=2, seed=53)
        for hints in (None, [], ()):
            result = solve(problem, seconds=2, seed=53, hints=hints)
            self.assertEqual(result['board'], default['board'])
            self.assertEqual(result['branches'], default['branches'])
            self.assertEqual(result['conflicts'], default['conflicts'])
            self.assertEqual(result['hint_hash'], default['hint_hash'])
            self.assertEqual(result['hint_count'], 0)

    def test_hints_checkpoint_binding_and_old_version_rejection(self):
        problem, answer = generated_problem(2, 2, 17)
        hints = [answer, [0] * 4]
        saved = solve(problem, seconds=0, seed=7, hints=hints)['checkpoint']
        self.assertEqual(saved['version'], 2)
        self.assertEqual(saved['ordering_version'], exact_dfs.ORDERING_VERSION)
        self.assertEqual(saved['problem_hash'], exact_dfs.problem_hash(problem))
        result = solve(problem, seconds=2, seed=7, hints=tuple(tuple(b) for b in hints), checkpoint=saved)
        self.assertEqual(result['outcome'], 'solved')
        for changed in (None, [answer], list(reversed(hints)), [answer, [1] * 4]):
            with self.subTest(hints=changed), self.assertRaisesRegex(ValueError, 'hint hash'):
                solve(problem, seconds=2, seed=7, hints=changed, checkpoint=saved)
        altered = dict(saved, ordering_version='different-ordering')
        with self.assertRaisesRegex(ValueError, 'ordering version'):
            solve(problem, seconds=2, seed=7, hints=hints, checkpoint=reseal(altered))
        legacy = {k: v for k, v in saved.items() if k not in ('hint_hash', 'ordering_version')}
        legacy['version'] = 1
        with self.assertRaisesRegex(ValueError, 'Unsupported DFS checkpoint version'):
            solve(problem, seconds=2, seed=7, hints=hints, checkpoint=reseal(legacy))

    def test_hint_shape_and_state_bounds(self):
        problem, answer = generated_problem(2, 2, 17)
        malformed = ('not boards', {0: answer}, iter([answer]), [answer] * 65,
                     [[0]], [[0] * 5], [[True, 1, 2, 3]], [[0.0, 1, 2, 3]],
                     [[-1, 1, 2, 3]], [[16, 1, 2, 3]], [None])
        for hints in malformed:
            with self.subTest(hints=str(hints)[:40]), self.assertRaises(ValueError):
                solve(problem, seconds=2, hints=hints)


if __name__ == '__main__':
    unittest.main()
