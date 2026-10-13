"""Tiny incremental SAT lifecycle and assumption-scope checks; no full search."""
import importlib.util
import itertools
import threading
import time
import unittest
from unittest.mock import patch

import exact_sat
from exact_model import ExactProblem, check_solution
from test_exact_sat import tiny_problem


@unittest.skipUnless(importlib.util.find_spec('pysat'), 'python-sat unavailable')
class AssumptionSessionTests(unittest.TestCase):
    def test_repeated_queries_match_fresh_solves_and_exhaustive_oracle(self):
        from pysat.solvers import Glucose4
        problem = tiny_problem()
        bad = [7, 1, 13, 11]
        queries = [{}, dict(enumerate(bad)), {0: 0}, {0: 7, 1: 1}, {}, {0: 1}, {}]
        decisions = conflicts = 0
        with patch('pysat.solvers.Glucose4', wraps=Glucose4) as factory:
            with exact_sat.AssumptionSession(problem, seconds=2, seed=17) as session:
                identity = None
                answers = []
                for assumptions in queries:
                    result = session.solve(assumptions)
                    answers.append(result)
                    if identity is None:
                        identity = id(session.native)
                    self.assertEqual(id(session.native), identity)
                    self.assertEqual(result['model_builds'], 1)
                    exists = any(check_solution(problem, b) and all(b[c] == s for c, s in assumptions.items())
                                 for b in itertools.product(*problem.domains))
                    self.assertEqual(result['outcome'], 'solved' if exists else 'infeasible')
                    self.assertEqual(result['core_cells'] is None, exists)
                    if exists:
                        self.assertTrue(check_solution(problem, result['board']))
                    else:
                        core = result['core_placements']
                        self.assertTrue(set(core.items()) <= set(assumptions.items()))
                        self.assertFalse(any(check_solution(problem, b) and all(b[c] == s for c, s in core.items())
                                             for b in itertools.product(*problem.domains)))
                    decisions += result['branches']
                    conflicts += result['conflicts']
                    self.assertEqual(decisions, result['cumulative_branches'])
                    self.assertEqual(conflicts, result['cumulative_conflicts'])
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(factory.call_args.kwargs, {'incr': True})
        self.assertTrue(session.closed)
        self.assertIsNone(session.native)
        for assumptions, result in zip(queries, answers):
            fresh = exact_sat.solve_with_assumptions(problem, assumptions, seconds=2, seed=17)
            self.assertEqual(result['outcome'], fresh['outcome'])

    def test_repeated_selector_sets_do_not_add_clauses_or_builds(self):
        with exact_sat.AssumptionSession(tiny_problem(), seconds=2) as session:
            first = session.solve({0: 0, 1: 4})
            second = session.solve({0: 0, 1: 4})
            released = session.solve({})
            self.assertEqual(first['clauses'], second['clauses'])
            self.assertEqual(first['variables'], second['variables'])
            self.assertEqual(released['clauses'], first['clauses'])
            self.assertEqual(released['outcome'], 'solved')
            self.assertEqual(released['model_builds'], 1)
            self.assertEqual(released['assumption_queries'], 3)

    def test_global_deadline_includes_idle_time_between_queries(self):
        with exact_sat.AssumptionSession(tiny_problem(), seconds=.03) as session:
            self.assertEqual(session.solve({})['outcome'], 'solved')
            time.sleep(.04)
            result = session.solve({0: 1})
        self.assertEqual(result['outcome'], 'timeout')
        self.assertIsNone(result['core_cells'])

    def test_query_timeout_and_unknown_do_not_poison_next_query(self):
        with exact_sat.AssumptionSession(tiny_problem(), seconds=2) as session:
            session.solve({})
            native = session.native
            with patch.object(native, 'solve_limited', return_value=None), \
                 patch.object(native, 'get_core', side_effect=AssertionError('Stale core queried')):
                unknown = session.solve({0: 7})
            self.assertEqual(unknown['outcome'], 'timeout')
            self.assertIsNone(unknown['core_cells'])
            self.assertIsNone(unknown['board'])
            limited = session.solve({0: 1}, seconds=0)
            self.assertEqual(limited['outcome'], 'timeout')
            self.assertIsNone(limited['core_cells'])
            result = session.solve({})
            self.assertEqual(result['outcome'], 'solved')
            self.assertEqual(result['model_builds'], 1)

    def test_native_short_query_interrupt_clears_for_next_query(self):
        with exact_sat.AssumptionSession(tiny_problem(), seconds=2) as session:
            session.solve({})
            event = threading.Event()
            native = session.native
            original_interrupt = native.interrupt
            def interrupt():
                original_interrupt()
                event.set()
            def waiting(**kwargs):
                self.assertTrue(event.wait(1))
                return False  # Must be suppressed because the budget expired.
            with patch.object(native, 'interrupt', side_effect=interrupt), patch.object(native, 'solve_limited', side_effect=waiting):
                timed = session.solve({0: 7}, seconds=.02)
            self.assertEqual(timed['outcome'], 'timeout')
            self.assertIsNone(timed['core_cells'])
            self.assertEqual(session.solve({})['outcome'], 'solved')
        self.assertFalse(any(t.name == 'eternity-sat-interrupt' for t in threading.enumerate()))

    def test_close_during_native_query_interrupts_before_delete(self):
        session = exact_sat.AssumptionSession(tiny_problem(), seconds=2)
        session.solve({})
        entered, interrupted = threading.Event(), threading.Event()
        answer = []
        native = session.native
        original_interrupt = native.interrupt
        def interrupt():
            original_interrupt()
            interrupted.set()
        def waiting(**kwargs):
            entered.set()
            self.assertTrue(interrupted.wait(1))
            return None
        with patch.object(native, 'interrupt', side_effect=interrupt), patch.object(native, 'solve_limited', side_effect=waiting):
            worker = threading.Thread(target=lambda: answer.append(session.solve({})))
            worker.start()
            self.assertTrue(entered.wait(1))
            session.close()
            worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(answer[0]['outcome'], 'stopped')
        self.assertIsNone(answer[0]['core_cells'])
        session.close()  # Idempotent cleanup.
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            session.solve({})

    def test_base_unsat_reuses_empty_core_and_stop_does_not_reuse_core(self):
        stop = threading.Event()
        with exact_sat.AssumptionSession(tiny_problem(True), seconds=2, should_stop=stop.is_set) as session:
            first = session.solve({})
            second = session.solve({0: 0})
            self.assertEqual(first['outcome'], 'infeasible')
            self.assertEqual(second['core_cells'], [])
            stop.set()
            stopped = session.solve({0: 0})
            self.assertEqual(stopped['outcome'], 'stopped')
            self.assertIsNone(stopped['core_cells'])

    def test_invalid_queries_and_callback_failure_close_context(self):
        with exact_sat.AssumptionSession(tiny_problem(), seconds=2) as session:
            for assumptions in ({True: 0}, {0: True}, {4: 0}, {0: 16}, []):
                with self.assertRaises(ValueError):
                    session.solve(assumptions)
            with self.assertRaises(ValueError):
                session.solve({}, seconds=float('nan'))
        def broken():
            raise OSError('fixture callback error')
        with self.assertRaisesRegex(RuntimeError, 'cancellation callback'):
            with exact_sat.AssumptionSession(tiny_problem(), seconds=2, should_stop=broken) as failed:
                failed.solve({})
        self.assertTrue(failed.closed)

    def test_partial_encoding_timeout_is_discarded_before_retry(self):
        first = [True]
        def slow_once(update):
            if first[0] and update['phase'] == 'encoding':
                first[0] = False
                time.sleep(.02)
        with exact_sat.AssumptionSession(tiny_problem(), seconds=2, on_progress=slow_once) as session:
            expired = session.solve({}, seconds=.005)
            self.assertEqual(expired['outcome'], 'timeout')
            self.assertIsNone(expired['core_cells'])
            self.assertIsNone(session.native)
            self.assertFalse(session._built)
            retried = session.solve({})
            self.assertEqual(retried['outcome'], 'solved')
            self.assertEqual(retried['model_builds'], 2)
            self.assertTrue(check_solution(tiny_problem(), retried['board']))

    def test_progress_exception_closes_allocated_native_model(self):
        def broken(update):
            raise OSError('fixture progress failure')
        session = exact_sat.AssumptionSession(tiny_problem(), seconds=2, on_progress=broken)
        with self.assertRaisesRegex(OSError, 'fixture progress failure'):
            with session:
                session.solve({})
        self.assertTrue(session.closed)
        self.assertIsNone(session.native)

    def test_session_snapshots_mutable_problem_containers(self):
        base = tiny_problem(fixed={0: 0})
        mutable = ExactProblem(base.size, [list(face) for face in base.faces], dict(base.fixed),
                               [list(domain) for domain in base.domains], list(((0, 1, 1, 3), (0, 2, 2, 0))))
        with exact_sat.AssumptionSession(mutable, seconds=2) as session:
            frozen = session.problem
            first = session.solve({})
            self.assertEqual(first['outcome'], 'solved')
            self.assertEqual(first['board'][0], 0)
            mutable.fixed.clear()
            mutable.fixed[0] = 7
            mutable.faces[0][1] = 99
            mutable.domains[0].clear()
            mutable.edges.clear()
            again = session.solve({})
            self.assertEqual(again['outcome'], 'solved')
            self.assertEqual(again['board'][0], 0)
            self.assertTrue(check_solution(frozen, again['board']))
            self.assertEqual(session.solve({0: 7})['outcome'], 'infeasible')
            self.assertEqual(session.problem.fixed, {0: 0})
            self.assertEqual(session.problem.faces, base.faces)
            self.assertTrue(session.problem.domains[0])
            self.assertEqual(len(session.problem.edges), 2)
            self.assertEqual(session.model_builds, 1)

    def test_should_stop_can_close_from_native_monitor_without_deadlock(self):
        entered, interrupted, finished = threading.Event(), threading.Event(), threading.Event()
        returned, answers, failures = threading.Event(), [], []
        session = None
        def cancel_from_monitor():
            if entered.is_set() and threading.current_thread().name == 'eternity-sat-interrupt':
                session.close()
                self.assertIsNotNone(session.native, 'Deletion must wait for native exit')
                returned.set()
            return False
        session = exact_sat.AssumptionSession(tiny_problem(), seconds=2, should_stop=cancel_from_monitor)
        session.solve({})
        native = session.native
        original_interrupt, original_delete = native.interrupt, native.delete
        def interrupt():
            original_interrupt()
            interrupted.set()
        def waiting(**kwargs):
            entered.set()
            self.assertTrue(interrupted.wait(.5), 'Close callback did not interrupt the native query')
            finished.set()
            return None
        def delete():
            self.assertTrue(finished.is_set(), 'Native solver deleted before solve returned')
            original_delete()
        def query():
            try:
                answers.append(session.solve({}))
            except BaseException as exc:
                failures.append(exc)
        with patch.object(native, 'interrupt', side_effect=interrupt), \
             patch.object(native, 'solve_limited', side_effect=waiting), \
             patch.object(native, 'delete', side_effect=delete):
            worker = threading.Thread(target=query, daemon=True)
            worker.start()
            worker.join(1)
            self.assertFalse(worker.is_alive(), 'Callback close deadlocked the native monitor')
        self.assertEqual(failures, [])
        self.assertTrue(returned.is_set())
        self.assertEqual(answers[0]['outcome'], 'stopped')
        self.assertIsNone(answers[0]['core_cells'])
        self.assertTrue(session.closed)
        self.assertIsNone(session.native)
        self.assertFalse(any(t.name == 'eternity-sat-interrupt' for t in threading.enumerate()))

    def test_progress_callback_close_is_deferred_until_callback_returns(self):
        for phase in ('encoding', 'searching'):
            called = []
            session = None
            def close_in_progress(update):
                if update['phase'] == phase:
                    session.close()
                    self.assertIsNotNone(session.native)
                    self.assertFalse(session.closed)
                    called.append(True)
            session = exact_sat.AssumptionSession(tiny_problem(), seconds=2, on_progress=close_in_progress)
            result = session.solve({})
            self.assertEqual(called, [True])
            self.assertEqual(result['outcome'], 'stopped')
            self.assertIsNone(result['core_cells'])
            self.assertTrue(session.closed)
            self.assertIsNone(session.native)


if __name__ == '__main__':
    unittest.main()
