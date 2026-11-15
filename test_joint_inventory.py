"""Independent finite oracles for exact joint inventory necessary conditions."""
from dataclasses import replace
from itertools import combinations,product
import random
import unittest
from exact_model import generated_problem,make_problem,check_solution,from_bundle
from exact_rules import AdditionalRules,DEFAULT_OPTIONS
from joint_inventory import JointInventory,cardinality_vectors


def bitdomains(problem): return [sum(1<<s for s in set(d)) for d in problem.domains]
def rotations(face): return [tuple(face[(side-r)%4] for side in range(4)) for r in range(4)]


def joint_counterexample():
    # Four movable interior vectors: (1,0),(0,1),(2,0),(0,2).
    # Two must be black; the required pair sum is (1,3), impossible jointly.
    # Each one-color target, including the filler color3, is separately possible.
    faces=[];fixed={}
    special={2:(0,3,1,3),1:(0,3,2,3),4:(3,2,3,0),11:(3,0,3,2),
             5:(1,3,3,3),6:(2,3,3,3),9:(1,1,3,3),10:(2,2,3,3)}
    for c in range(16):
        face=special.get(c,tuple(0 if boundary else 3 for boundary in
                       (c<4,c%4==3,c>=12,c%4==0)))
        faces.extend(rotations(face))
        if c not in (5,6,9,10):fixed[c]=4*c
    return make_problem(4,faces,fixed)


class JointInventoryTests(unittest.TestCase):
    def test_vector_dp_matches_exhaustive_subset_oracle(self):
        rng=random.Random(51025)
        for width in (1,2,3):
            for _ in range(90):
                n=rng.randrange(0,9);count=rng.randrange(n+1)
                weights=[tuple(rng.randrange(5) for _ in range(width)) for _ in range(n)]
                target=tuple(rng.randrange(4*max(1,count)+1) for _ in range(width))
                expected=any(tuple(sum(weights[i][k] for i in subset) for k in range(width))==target
                             for subset in combinations(range(n),count))
                out=cardinality_vectors(weights,count,target,max_work=100000,max_states=100000)
                self.assertTrue(out['complete'])
                self.assertEqual(out['consistent'],expected,(weights,count,target))
                self.assertEqual(out['work'],out['preprocessing_work']+out['dp_transitions']+out['state_copies'])

    def test_joint_contradiction_passes_each_scalar_inventory(self):
        p=joint_counterexample();checker=JointInventory(p);domains=bitdomains(p)
        options={k:False for k,v in DEFAULT_OPTIONS.items() if type(v) is bool}
        options['color_balance']=True
        scalar=AdditionalRules(p,options=options)
        self.assertTrue(scalar.apply(domains.copy())[0])
        for color in checker.palette:
            out=checker.check(domains,colors=[color])
            self.assertTrue(out['consistent']);self.assertTrue(out['complete'])
        out=checker.check(domains,colors=[1,2])
        self.assertEqual(out['target'],[1,3]);self.assertEqual(out['required_black'],2)
        self.assertFalse(out['consistent']);self.assertTrue(out['complete'])
        self.assertEqual(out['termination'],'joint_inventory_impossible')

    def test_preserves_every_exhaustive_tiny_solution(self):
        for seed in range(10):
            p,answer=generated_problem(2,3,seed)
            solutions=[list(b) for b in product(*p.domains) if check_solution(p,list(b))]
            self.assertTrue(solutions)
            checker=JointInventory(p)
            for board in solutions:
                out=checker.check([1<<s for s in board],width=3)
                self.assertTrue(out['consistent']);self.assertTrue(out['complete'])

    def test_preserves_narrowed_known_solution_domains(self):
        for size in (3,4,5):
            for seed in (101,211,307):
                p,answer=generated_problem(size,5,seed,clue_count=2)
                rng=random.Random(seed);domains=[]
                for c,d in enumerate(p.domains):
                    domains.append(sum(1<<s for s in d if s==answer[c] or rng.randrange(4)==0))
                checker=JointInventory(p)
                for width in (1,2,3):
                    out=checker.check(domains,width=width,max_work=200000,max_states=100000)
                    self.assertTrue(out['consistent'])

    def test_work_cap_and_stop_never_reject(self):
        p=joint_counterexample();checker=JointInventory(p);domains=bitdomains(p)
        for cap in (0,1,20,80):
            out=checker.check(domains,colors=[1,2],max_work=cap)
            self.assertTrue(out['consistent']);self.assertFalse(out['complete'])
            self.assertLessEqual(out['work'],cap)
        out=checker.check(domains,colors=[1,2],should_stop=lambda:True)
        self.assertEqual(out['termination'],'stopped');self.assertTrue(out['consistent'])
        self.assertEqual(domains,bitdomains(p))

    def test_state_cap_is_unknown_not_a_negative_proof(self):
        out=cardinality_vectors([(1,0),(0,1),(2,0),(0,2)],2,(1,3),max_states=1)
        self.assertTrue(out['consistent']);self.assertFalse(out['complete'])
        self.assertEqual(out['termination'],'state_limit')

    def test_inapplicable_problem_skips(self):
        p=joint_counterexample()
        for changed in (replace(p,edges=()),replace(p,faces=((0,3,4,0),)+p.faces[1:])):
            out=JointInventory(changed).check(bitdomains(changed))
            self.assertTrue(out['consistent']);self.assertFalse(out['complete'])
        p,answer=generated_problem(2,2,7)
        raw=replace(p,domains=tuple(tuple(range(16)) for _ in range(4)))
        out=JointInventory(raw).check(bitdomains(raw))
        self.assertEqual(out['termination'],'unframed_domains');self.assertTrue(out['consistent'])

    def test_setup_is_detached_from_input_and_checks_do_not_mutate(self):
        p=joint_counterexample();checker=JointInventory(p);before=bitdomains(p)
        p.fixed.clear()
        first=checker.check(before,colors=[1,2]);second=checker.check(before,colors=[1,2])
        self.assertEqual(first,second)
        self.assertEqual(before,bitdomains(p))

    def test_invalid_values_rejected(self):
        p=joint_counterexample();checker=JointInventory(p);domains=bitdomains(p)
        for options in ({'width':True},{'width':4},{'colors':[0]},{'colors':[1,1]},
                        {'max_work':-1},{'max_states':0},{'should_stop':42}):
            with self.assertRaises(ValueError):checker.check(domains,**options)
        for invalid in ([True]*16,domains[:-1],[-1]*16):
            with self.assertRaises(ValueError):checker.check(invalid)
        wrong=domains.copy();wrong[0]|=1<<4
        with self.assertRaises(ValueError):checker.check(wrong)
        for vectors,count,target in (([(5,)],1,(1,)), ([(True,)],1,(1,)), ([(1,)],True,(1,)), ([(1,)],1,())):
            with self.assertRaises(ValueError):cardinality_vectors(vectors,count,target)

    def test_original_puzzle_is_bounded_and_uses_all_clues(self):
        from validator import load_bundle
        p=from_bundle(load_bundle());checker=JointInventory(p)
        self.assertEqual(len(p.fixed),5)
        result=checker.check(bitdomains(p),width=2,max_work=10000,max_states=4000)
        self.assertTrue(result['consistent']);self.assertLessEqual(result['work'],10000)
        self.assertLessEqual(checker.setup_work,300000)

if __name__=='__main__':unittest.main()
