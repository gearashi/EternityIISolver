"""Finite policy contracts: choices guide search, never prove exhaustion."""
import copy
import itertools
import json
import unittest
from dataclasses import replace

from exact_model import generated_problem
from repair_policy import PairSelector, AdaptiveRegions


def pair_fixture():
    problem, board = generated_problem(4, 7, 19, clue_count=0)
    cells = [5, 6, 9, 10]
    long_cycle = list(board)
    for i, cell in enumerate(cells):
        long_cycle[cell] = board[cells[(i+1) % len(cells)]]
    split_cycles = list(board)
    for a, b in ((5, 6), (9, 10)):
        split_cycles[a], split_cycles[b] = board[b], board[a]
    return problem, board, long_cycle, split_cycles


class PairPolicyTests(unittest.TestCase):
    def test_cycle_structure_prefers_more_than_two_parent_choices(self):
        problem, board, long_cycle, split_cycles = pair_fixture()
        selector = PairSelector(problem)
        baseline = selector.choose([board, long_cycle])
        self.assertEqual(baseline['active_cycles'], 1)
        result = selector.choose([board, long_cycle, split_cycles], seed=17, samples=3)
        self.assertEqual(result['examined_pairs'], 3)
        self.assertGreaterEqual(result['active_cycles'], 2)
        self.assertGreaterEqual(result['nontrivial_cycles'], 2)
        self.assertGreaterEqual(result['mismatch_cycles'], 2)
        self.assertEqual((result['left'], result['right']), (0, 2))
        # Independent fixture permutation: two swaps have four binary outcomes,
        # while the single 4-cycle has only the two original parents.
        self.assertEqual(2**result['active_cycles'], 4)
        self.assertEqual(2**baseline['active_cycles'], 2)

    def test_seeded_sampling_bounded_and_inputs_unchanged(self):
        problem, *bank = pair_fixture()
        selector = PairSelector(problem)
        original = copy.deepcopy(bank)
        a = selector.choose(bank, seed=41, samples=2)
        b = selector.choose(bank, seed=41, samples=2)
        self.assertEqual(a, b)
        self.assertEqual(a['examined_pairs'], 2)
        self.assertEqual(bank, original)
        self.assertIsNone(selector.choose([]))
        self.assertIsNone(selector.choose(bank[:1]))

    def test_rotation_components_and_duplicate_boards(self):
        problem, board, _, _ = pair_fixture()
        changed = list(board)
        for c in (5, 6):
            changed[c] = 4*(board[c]//4) + (board[c]+1)%4
        selector = PairSelector(problem)
        result = selector.choose([board, changed])
        self.assertEqual(result['active_cycles'], 2)
        self.assertEqual(result['nontrivial_cycles'], 0)
        self.assertEqual(selector.choose([board, board])['active_cycles'], 0)

    def test_illegal_parent_rejected_and_snapshot_detached(self):
        problem, board, cycle, _ = pair_fixture()
        fixed_problem = replace(problem, fixed={5: board[5]})
        fixed_selector = PairSelector(fixed_problem)
        with self.assertRaises(ValueError):
            fixed_selector.choose([board, cycle])
        # Snapshot retains the fixed clue even if the supplied mutable mapping
        # changes after construction.
        fixed_problem.fixed.clear()
        with self.assertRaises(ValueError):
            fixed_selector.choose([board, cycle])
        bad = list(board);bad[1] = bad[0]
        with self.assertRaises(ValueError):
            PairSelector(problem).choose([board, bad])
        bad = list(board);bad[0] = 4*(bad[0]//4)+(bad[0]+1)%4
        with self.assertRaises(ValueError):
            PairSelector(problem).choose([board, bad])
        domains = list(problem.domains);domains[5] = (board[5],)
        with self.assertRaises(ValueError):
            PairSelector(replace(problem,domains=tuple(domains))).choose([board, cycle])

    def test_bounds_cancellation_and_bad_configuration(self):
        problem, board, cycle, _ = pair_fixture()
        selector = PairSelector(problem)
        self.assertIsNone(selector.choose([board,cycle],should_stop=lambda:True))
        for kwargs in ({'seed':True},{'seed':-1},{'samples':0},{'samples':33},{'should_stop':1}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                selector.choose([board,cycle],**kwargs)
        with self.assertRaises(ValueError):
            selector.choose(itertools.repeat(board))
        with self.assertRaises(ValueError):
            selector.choose([board,itertools.repeat(0)])
        with self.assertRaises(ValueError):
            PairSelector(replace(problem,edges=()))


class AdaptiveRegionTests(unittest.TestCase):
    def test_initial_exploration_covers_size_and_time_profiles(self):
        policy = AdaptiveRegions(max_seconds=1.)
        plans = []
        for _ in range(4):
            plan = policy.plan(elapsed_seconds=100.,remaining_seconds=10.)
            plans.append(plan)
            policy.observe(plan['arm'],elapsed_seconds=plan['seconds'],score_gain=0)
        self.assertEqual([p['arm'] for p in plans], [0,1,2,3])
        self.assertEqual([p['region_size'] for p in plans], [8,12,16,20])
        self.assertEqual([p['seconds'] for p in plans], [.125,.25,.5,1.])

    def test_smallest_profile_cap_still_progresses(self):
        policy=AdaptiveRegions(max_seconds=.02)
        for arm in range(4):
            plan=policy.plan(elapsed_seconds=100.,remaining_seconds=10.)
            self.assertEqual(plan['arm'],arm)
            self.assertEqual(plan['seconds'],.02)
            policy.observe(arm,elapsed_seconds=.02,score_gain=0)

    def test_reward_per_time_adapts_without_starvation(self):
        policy = AdaptiveRegions(max_seconds=1.)
        choices = []
        for step in range(80):
            plan = policy.plan(elapsed_seconds=1000.,remaining_seconds=10.)
            arm = plan['arm'];choices.append(arm)
            # Controlled equal-cost outcomes: only arm 2 gives a verified gain.
            policy.observe(arm,elapsed_seconds=.1,score_gain=1 if arm==2 else 0)
        self.assertGreater(choices.count(2), 50)
        self.assertTrue(all(choices.count(i)>=4 for i in range(4)))
        # Every 20-plan window after initial sampling still explores every arm.
        self.assertTrue(all(set(choices[start:start+20])==set(range(4))
                            for start in range(4,61)))

    def test_zero_successes_continue_balanced_exploration(self):
        policy = AdaptiveRegions()
        chosen = []
        for _ in range(24):
            plan = policy.plan(elapsed_seconds=1000.,remaining_seconds=10.)
            chosen.append(plan['arm'])
            policy.observe(plan['arm'],elapsed_seconds=.1,score_gain=0)
        self.assertEqual(chosen, [0,1,2,3]*6)

    def test_budget_debt_caps_stop_and_bad_inputs(self):
        policy = AdaptiveRegions(max_seconds=1.,share=.2)
        self.assertIsNone(policy.plan(elapsed_seconds=.05,remaining_seconds=10.))
        self.assertIsNone(policy.plan(elapsed_seconds=100.,remaining_seconds=.01))
        self.assertIsNone(policy.plan(elapsed_seconds=100.,remaining_seconds=10.,should_stop=lambda:True))
        first=policy.plan(elapsed_seconds=1.,remaining_seconds=.05)
        self.assertAlmostEqual(first['seconds'], .05)
        # An overrun consumes the measured time, so the scheduler waits for its
        # total-search allowance to catch up rather than silently forgiving it.
        policy.observe(first['arm'],elapsed_seconds=.5,score_gain=0)
        self.assertIsNone(policy.plan(elapsed_seconds=2.,remaining_seconds=10.))
        plan=policy.plan(elapsed_seconds=3.,remaining_seconds=10.)
        self.assertAlmostEqual(plan['seconds'],.1)
        for kwargs in ({'max_seconds':0},{'max_seconds':5.1},{'share':True},{'share':.9}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):AdaptiveRegions(**kwargs)
        for kwargs in ({'elapsed_seconds':-1,'remaining_seconds':1},
                       {'elapsed_seconds':1,'remaining_seconds':float('inf')},
                       {'elapsed_seconds':1,'remaining_seconds':1,'should_stop':7}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):policy.plan(**kwargs)
        for arm,seconds,gain in ((True,1,0),(4,1,0),(0,-1,0),(0,1,481),(0,1,True)):
            with self.subTest(arm=arm,seconds=seconds,gain=gain),self.assertRaises(ValueError):
                policy.observe(arm,elapsed_seconds=seconds,score_gain=gain)

    def test_checkpoint_roundtrip_exact_next_choice_and_independence(self):
        original = AdaptiveRegions(max_seconds=1.)
        for i in range(12):
            plan=original.plan(elapsed_seconds=100.,remaining_seconds=10.)
            original.observe(plan['arm'],elapsed_seconds=.1,score_gain=1 if plan['arm']==1 else 0)
        state=json.loads(json.dumps(original.export_state(),allow_nan=False))
        restored=AdaptiveRegions(max_seconds=1.)
        restored.restore_state(state)
        self.assertEqual(restored.export_state(),original.export_state())
        self.assertEqual(restored.plan(elapsed_seconds=100.,remaining_seconds=10.),
                         original.plan(elapsed_seconds=100.,remaining_seconds=10.))
        state['attempts'][0]=999
        self.assertNotEqual(restored.export_state(),state)

    def test_corrupt_checkpoint_is_rejected_atomically(self):
        policy=AdaptiveRegions()
        good=policy.export_state()
        corrupt=[]
        for key,value in [('schema','bad'),('max_seconds',1.),('share',float('nan')),
                          ('attempts',[0,0,0,True]),('seconds',[0,0,0,float('inf')]),
                          ('gains',[1,0,0,0]),('attempts',[0])]:
            value_state=copy.deepcopy(good);value_state[key]=value;corrupt.append(value_state)
        extra=copy.deepcopy(good);extra['surprise']=0;corrupt.append(extra)
        missing=copy.deepcopy(good);del missing['schema'];corrupt.append(missing)
        for value in corrupt:
            with self.subTest(value=value),self.assertRaises(ValueError):policy.restore_state(value)
            self.assertEqual(policy.export_state(),good)


if __name__=='__main__':unittest.main()
