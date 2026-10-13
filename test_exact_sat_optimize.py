"""Finite SAT-score and learned-core controller oracles; no full-puzzle run."""
from dataclasses import replace
import importlib.util
import itertools
import threading
import time
import unittest
from unittest.mock import patch

from exact_model import ExactProblem, generated_problem
from exact_optimize import _CoreHittingSet, core_guided_repair, evaluate_board
from exact_sat_optimize import ScoreSession, optimize
from test_exact_sat import tiny_problem


def boards(problem):
    for board in itertools.product(*problem.domains):
        checked = evaluate_board(problem, board)
        if checked['valid']:
            yield checked['score'], list(board)


@unittest.skipUnless(importlib.util.find_spec('pysat'), 'python-sat unavailable')
class IncrementalScoreTests(unittest.TestCase):
    def test_every_threshold_matches_exhaustive_oracle_in_both_query_directions(self):
        for impossible in (False, True):
            for fixed in ({}, {0: 0}):
                problem = tiny_problem(impossible, fixed=fixed)
                optimum = max(boards(problem))[0]
                with ScoreSession(problem, seconds=3) as session:
                    identity, clauses, total_branches = None, None, 0
                    for target in (0, 4, 2, 3, 1, 4, 0):
                        answer = session.solve(target)
                        self.assertEqual(answer['outcome'], 'solved' if target <= optimum else 'infeasible')
                        if answer['board'] is not None:
                            checked = evaluate_board(problem, answer['board'])
                            self.assertTrue(checked['valid'])
                            self.assertEqual(checked['score'], answer['score'])
                            self.assertGreaterEqual(answer['score'], target)
                        if identity is None:
                            identity, clauses = id(session.native), answer['clauses']
                        self.assertEqual(id(session.native), identity)
                        self.assertEqual(answer['clauses'], clauses)
                        self.assertEqual(answer['model_builds'], 1)
                        total_branches += answer['branches']
                        self.assertEqual(total_branches, answer['cumulative_branches'])

    def test_all_tiny_legal_boards_have_exact_score_channels(self):
        problem = tiny_problem(True)
        for score, board in boards(problem):
            fixed = replace(problem, fixed=dict(enumerate(board)))
            with ScoreSession(fixed, seconds=2) as session:
                self.assertEqual(session.solve(score)['score'], score)
                if score < 4:
                    self.assertEqual(session.solve(score + 1)['outcome'], 'infeasible')

    def test_random_small_domain_oracles_and_restricted_range(self):
        import random
        rng = random.Random(127)
        for seed in range(8):
            problem, _ = generated_problem(2, 3, seed)
            domains = tuple(tuple(s for s in d if rng.random() < .7) for d in problem.domains)
            restricted = replace(problem, domains=domains)
            optimum = max((score for score, board in boards(restricted)), default=-1)
            with ScoreSession(restricted, seconds=2, min_score=2) as session:
                for target in (4, 2, 3):
                    self.assertEqual(session.solve(target)['outcome'], 'solved' if optimum >= target else 'infeasible')
                with self.assertRaises(ValueError):
                    session.solve(1)

    def test_optimizer_reaches_exhaustive_optimum_and_keeps_callbacks_strict(self):
        for impossible in (False, True):
            problem = tiny_problem(impossible)
            candidates = sorted(boards(problem))
            reference = candidates[0][1]
            improvements = []
            answer = optimize(problem, reference, seconds=3, on_improvement=improvements.append)
            self.assertEqual(answer['score'], candidates[-1][0])
            self.assertEqual(answer['outcome'], 'improved' if impossible else 'solved')
            self.assertEqual(answer['model_builds'], 1)
            self.assertEqual([v['score'] for v in improvements], sorted(set(v['score'] for v in improvements)))
            self.assertTrue(all(evaluate_board(problem, v['board'])['score'] == v['score'] for v in improvements))
            self.assertTrue(improvements)
            if impossible:
                self.assertEqual(answer['termination'], 'optimal_for_supplied_problem')
                self.assertGreater(answer['target_queries'], 1)

    def test_reference_phases_are_hints_and_never_freezes(self):
        problem = tiny_problem()
        reference = min(boards(problem))[1]
        for hints in (False, True):
            answer = optimize(problem, reference, seconds=2, reference_phases=hints)
            self.assertEqual(answer['outcome'], 'solved')
            self.assertNotEqual(answer['board'], reference)
            self.assertEqual(answer['reference_phases'], hints)
        with self.assertRaises(ValueError):
            ScoreSession(problem, phase_board=[0, 0, 0, 0])
        with self.assertRaises(ValueError):
            optimize(problem, reference, reference_phases=1)

    def test_target_infeasible_and_budget_stop_preserve_reference(self):
        problem = tiny_problem(True)
        score, reference = max(boards(problem))
        self.assertEqual(optimize(problem, reference, seconds=2)['outcome'], 'infeasible_for_target')
        for kwargs, expected in (({'seconds': 0}, 'timeout'), ({'seconds': 2, 'should_stop': lambda: True}, 'stopped')):
            with patch('pysat.solvers.Glucose4', side_effect=AssertionError('Native search forbidden')):
                answer = optimize(problem, reference, **kwargs)
            self.assertEqual(answer['outcome'], expected)
            self.assertEqual(answer['board'], reference)
            self.assertEqual(answer['model_builds'], 0)

    def test_unknown_and_short_timeout_do_not_poison_weaker_later_target(self):
        with ScoreSession(tiny_problem(True), seconds=2) as session:
            session.solve(0)
            native = session.native
            with patch.object(native, 'solve_limited', return_value=None), \
                 patch.object(native, 'get_model', side_effect=AssertionError('Stale model')):
                self.assertEqual(session.solve(4)['outcome'], 'timeout')
            self.assertEqual(session.solve(4, seconds=0)['outcome'], 'timeout')
            self.assertEqual(session.solve(4)['outcome'], 'infeasible')
            self.assertEqual(session.solve(3)['outcome'], 'solved')

    def test_close_and_reentry_callbacks_do_not_deadlock_or_change_target(self):
        session = ScoreSession(tiny_problem(), seconds=2)
        rejected = []
        def progress(update):
            with self.assertRaises(RuntimeError):
                session.solve(4)
            rejected.append(session.target_score)
        session.on_progress = progress
        self.assertEqual(session.solve(1)['outcome'], 'solved')
        self.assertTrue(rejected)
        self.assertEqual(set(rejected), {1})
        session.on_progress = lambda update: session.close()
        self.assertEqual(session.solve(2)['outcome'], 'stopped')
        self.assertTrue(session.closed)
        self.assertIsNone(session.native)

    def test_active_native_interrupt_and_callback_errors_cleanup(self):
        stop = threading.Event()
        with ScoreSession(tiny_problem(), seconds=2, should_stop=stop.is_set) as session:
            session.solve(0)
            interrupted = threading.Event()
            original = session.native.interrupt
            def interrupt():
                original()
                interrupted.set()
            def waiting(**kwargs):
                stop.set()
                self.assertTrue(interrupted.wait(1))
                return False
            with patch.object(session.native, 'interrupt', side_effect=interrupt), \
                 patch.object(session.native, 'solve_limited', side_effect=waiting):
                answer = session.solve(4)
            self.assertEqual(answer['outcome'], 'stopped')
            self.assertIsNone(answer['board'])
        def boom(update):
            raise ValueError('callback failed')
        with self.assertRaises(ValueError):
            optimize(tiny_problem(), min(boards(tiny_problem()))[1], seconds=2, on_improvement=boom)
        self.assertFalse(any(t.name == 'eternity-sat-interrupt' for t in threading.enumerate()))

    def test_build_time_is_not_reissued_and_problem_inputs_are_snapshotted(self):
        problem = tiny_problem()
        with ScoreSession(problem, seconds=2) as session:
            first = session.solve(4)
            problem.fixed[0] = 7
            again = session.solve(4)
            self.assertEqual(first['outcome'], again['outcome'])
            self.assertEqual(again['outcome'], 'solved')
        original = ScoreSession._add
        def slow_add(session, clause):
            time.sleep(.02)
            return original(session, clause)
        with patch.object(ScoreSession, '_add', slow_add):
            answer = optimize(tiny_problem(), min(boards(tiny_problem()))[1], seconds=.01)
        self.assertEqual(answer['outcome'], 'timeout')
        self.assertEqual(answer['target_queries'], 1)

    def test_progress_mutation_cannot_change_encoded_score_problem(self):
        problem = tiny_problem(fixed={0: 0})
        reference = min(boards(problem))[1]
        snapshot = replace(problem, fixed=dict(problem.fixed))
        def mutate(update):
            problem.fixed.clear()
            problem.fixed[0] = 7
            reference[0] = 7
        answer = optimize(problem, reference, seconds=2, on_progress=mutate)
        self.assertEqual(answer['outcome'], 'solved')
        self.assertEqual(answer['board'][0], 0)
        self.assertTrue(evaluate_board(snapshot, answer['board'])['valid'])

    def test_bad_parameters_partial_edges_and_zero_edge_board(self):
        problem = tiny_problem()
        reference = min(boards(problem))[1]
        for kwargs in ({'target_score': 0}, {'target_score': True}, {'target_score': 5}, {'seconds': -1},
                       {'on_improvement': 1}, {'workers': 0}, {'seed': -1}):
            with self.assertRaises(ValueError):
                optimize(problem, reference, **kwargs)
        with self.assertRaises(ValueError):
            ScoreSession(replace(problem, edges=()))
        one, board = generated_problem(1, 1, 0)
        with ScoreSession(one, seconds=2) as session:
            self.assertEqual(session.solve(0)['score'], 0)
        self.assertEqual(optimize(one, board, seconds=2)['outcome'], 'solved')


@unittest.skipUnless(importlib.util.find_spec('ortools'), 'ortools unavailable')
class HittingSetTests(unittest.TestCase):
    def test_persistent_cores_choose_exact_minimum_and_can_refreeze(self):
        controller = _CoreHittingSet(range(4))
        controller.add((0, 1))
        first = controller.choose(seconds=1)
        self.assertEqual(len(first), 1)
        discarded = next(iter(first))
        other = 1 - discarded
        controller.add((other, 2))
        controller.add((other, 3))
        second = controller.choose(previous=first, seconds=1)
        self.assertEqual(second, {other})
        self.assertNotIn(discarded, second)
        self.assertTrue(controller.optimal)
        self.assertEqual(controller.solves, 2)

    def test_bounded_fallback_is_feasible_but_not_claimed_minimal(self):
        controller = _CoreHittingSet(range(4))
        for core in ((0, 1), (1, 2), (2, 3), (0, 3)):
            controller.add(core)
        for kwargs in ({'seconds': 0}, {'seconds': 1, 'should_stop': lambda: True}):
            cover = controller.choose(**kwargs)
            self.assertTrue(all(cover & core for core in controller.cores))
            self.assertFalse(controller.optimal)
        with patch.object(controller.cp_model.CpSolver, 'solve', return_value=controller.cp_model.UNKNOWN):
            cover = controller.choose(seconds=1)
        self.assertTrue(all(cover & core for core in controller.cores))
        self.assertFalse(controller.optimal)

    def test_small_hypergraph_cardinalities_match_exhaustive_oracle(self):
        all_cores = [{0, 1}, {1, 2}, {0, 2, 3}, {3, 4}, {1, 4}]
        controller = _CoreHittingSet(range(5))
        for core in all_cores:
            controller.add(core)
            chosen = controller.choose(seconds=1)
            optimum = min(len(s) for bits in itertools.product((False, True), repeat=5)
                          for s in [{i for i, b in enumerate(bits) if b}]
                          if all(s & c for c in controller.cores))
            self.assertEqual(len(chosen), optimum)
            self.assertTrue(controller.optimal)

    def test_cancelled_preprocessing_never_scans_a_long_core_history(self):
        controller = _CoreHittingSet(range(6))
        controller.add((0, 1))
        class UnreadableHistory:
            def __iter__(self):
                raise AssertionError('Core history scanned after stop or zero budget')
        saved = controller.cores
        controller.cores = UnreadableHistory()
        self.assertEqual(controller.choose(seconds=0), set(range(6)))
        self.assertEqual(controller.choose(seconds=1, should_stop=lambda: True), set(range(6)))
        controller.cores = saved * 10000
        ticks = [0]
        def stop():
            ticks[0] += 1
            return ticks[0] >= 3
        self.assertEqual(controller.choose(seconds=1, should_stop=stop), set(range(6)))
        self.assertEqual(ticks[0], 3)
        self.assertEqual(controller.solves, 0)
        self.assertFalse(controller.optimal)

    def test_controller_native_cancellation_and_exception_cleanup(self):
        from types import SimpleNamespace
        controller = _CoreHittingSet(range(3))
        controller.add((0, 1))
        entered, interrupted = threading.Event(), threading.Event()
        fake = SimpleNamespace(parameters=SimpleNamespace(), stop_search=interrupted.set)
        def search(model):
            entered.set()
            self.assertTrue(interrupted.wait(1))
            return controller.cp_model.UNKNOWN
        fake.solve = search
        with patch.object(controller.cp_model, 'CpSolver', return_value=fake):
            cover = controller.choose(seconds=2, should_stop=entered.is_set)
        self.assertTrue(cover & {0, 1})
        self.assertFalse(controller.optimal)
        self.assertFalse(any(t.name == 'eternity-core-hitting-set' for t in threading.enumerate()))
        entered.clear()
        interrupted.clear()
        def failure():
            if entered.is_set():
                raise ValueError('stop callback error')
            return False
        with patch.object(controller.cp_model, 'CpSolver', return_value=fake), self.assertRaises(RuntimeError):
            controller.choose(seconds=2, should_stop=failure)
        self.assertFalse(any(t.name == 'eternity-core-hitting-set' for t in threading.enumerate()))

    def test_transient_stop_in_controller_remains_terminal_for_core_repair(self):
        problem = tiny_problem(fixed={0: 0})
        reference = min(boards(problem))[1]
        trigger = False
        def stop_once():
            nonlocal trigger
            answer, trigger = trigger, False
            return answer
        original = _CoreHittingSet.choose
        def choose(controller, *args, **kwargs):
            nonlocal trigger
            trigger = True
            return original(controller, *args, **kwargs)
        with patch.object(_CoreHittingSet, 'choose', choose):
            answer = core_guided_repair(problem, reference, seconds=2, free_cells=[], should_stop=stop_once)
        self.assertEqual(answer['outcome'], 'stopped')
        self.assertEqual(answer['board'], reference)
        self.assertEqual(answer['rounds'], 1)

    def test_both_release_policies_respect_clues_and_finish_tiny_repair(self):
        problem = tiny_problem(fixed={0: 0})
        reference = min(boards(problem))[1]
        for policy in ('greedy', 'hitting-set'):
            answer = core_guided_repair(problem, reference, seconds=3, free_cells=[], release_policy=policy)
            self.assertEqual(answer['outcome'], 'solved')
            self.assertEqual(answer['board'][0], 0)
            self.assertEqual(answer['release_policy'], policy)
            self.assertTrue(all(0 not in core for core in answer['cores']))
            if policy == 'hitting-set':
                self.assertGreater(answer['hitting_set_solves'], 0)
                self.assertTrue(all(set(answer['free_cells']) & set(core) for core in answer['cores']))
        with self.assertRaises(ValueError):
            core_guided_repair(problem, reference, release_policy='unknown')


if __name__ == '__main__':
    unittest.main()
