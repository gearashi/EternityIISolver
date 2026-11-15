"""Evaluate learned pair choice and geometric difficulty archive on unseen banks."""
from itertools import combinations
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import binary_pairwise
from repair_parent_learning import features,vector,fit,difficulty_archive
from repair_parents import _features as diversity_features,_diverse
from repair_policy import PairSelector
from smart_repair import cycle_recombine
from exact_model import generated_problem,grid_edges,from_bundle
from exact_optimize import evaluate_board
from validator import load_bundle,validate_board


def bank_for(seed):
    p,sol=generated_problem(5,5,seed)
    cells=(6,7,8,11,12,13,16,17,18)
    rng=random.Random(seed)
    bank=[]
    for i in range(9):
        b=sol.copy()
        sub=rng.sample(cells,rng.randint(3,7))
        origins=sub.copy()
        if i%3:
            rng.shuffle(origins)
        for c,q in zip(sub,origins):
            b[c]=(sol[q]//4)*4+rng.randrange(4)
        bank.append(b)
    return p,bank


def gather(p,bank,seed,bundle=None):
    selector=PairSelector(p)
    checked=[selector._parent(b) for b in bank]
    vals=[features(p,b) for b in bank]
    pairs={}
    true_max=max(v[3] for v in checked)
    edges=grid_edges(p.size)
    dp=binary_pairwise.maximize
    inner=[]
    def observe(unary,terms,*args,**kwargs):
        result=dp(unary,terms,*args,**kwargs)
        inner.append(dict(binary_variables=len(unary),pair_terms=len(terms),assignments=result['assignments_evaluated'],
                          tree_components=result['tree_components']))
        return result
    for i,j in combinations(range(len(bank)),2):
        inner.clear()
        with patch.object(binary_pairwise,'maximize',observe):
            r=cycle_recombine(p,bank[i],bank[j],seconds=1,seed=seed)
        checked_score=evaluate_board(p,r['board'])
        assert checked_score['valid'] and checked_score['score']==r['score']
        if bundle is not None:
            official=validate_board(r['board'],bundle)
            assert official['valid'] and official['score']==r['score']
        x=vector(p,selector,checked[i],checked[j],vals[i],vals[j])
        heuristic=selector._features(checked[i],checked[j])[1]
        pairs[i,j]=dict(x=x,child_score=r['score'],verified_gain=max(0,r['score']-true_max),
                        exact=bool(r['optimal']),branches=r.get('branches',0),inner=list(inner),rank=heuristic)
    ranked=sorted(range(len(bank)),key=lambda i:(checked[i][3],-i),reverse=True)
    candidates=[(checked[i][3],-i,tuple(bank[i]),diversity_features(p,bank[i],edges)) for i in ranked]
    old=[bank.index(list(b)) for b in _diverse(candidates,4,p.size**2,lambda:False)]
    new=difficulty_archive(vals,bank,4)
    return pairs,true_max,old,new


def main(path):
    if path.exists():raise ValueError('output exists')
    xs=[];ys=[];training=[]
    for seed in range(190,202):
        p,bank=bank_for(seed)
        pairs,baseline,old,new=gather(p,bank,seed)
        for r in pairs.values():
            if r['exact']:
                xs.append(r['x']);ys.append(r['verified_gain']/len(grid_edges(p.size)))
        training.append(dict(seed=seed,bank_best=baseline,pairs=[dict(pair=ij,**r) for ij,r in pairs.items()]))
    w=fit(xs,ys)
    held=[]
    def test(name,p,bank,seed,bundle=None):
        pairs,baseline,old,new=gather(p,bank,seed,bundle)
        results={}
        for label,keep in (('all',list(range(len(bank)))),('diverse',old),('difficulty',new)):
            allowed={ij:r for ij,r in pairs.items() if all(i in keep for i in ij)}
            policy=max(allowed,key=lambda ij:(allowed[ij]['rank'],tuple(-x for x in ij)))
            learned=max(allowed,key=lambda ij:(sum(a*b for a,b in zip(w,allowed[ij]['x'])),tuple(-x for x in ij)))
            results[label]=dict(kept=keep,policy_pair=policy,policy_gain=allowed[policy]['verified_gain'],
                                learned_pair=learned,learned_gain=allowed[learned]['verified_gain'],
                                best_observed_gain=max(r['verified_gain'] for r in allowed.values()))
        held.append(dict(name=name,bank_best=baseline,results=results,pairs=[dict(pair=ij,**r) for ij,r in pairs.items()]))
    for seed in range(370,390):
        p,bank=bank_for(seed)
        test('generated-'+str(seed),p,bank,seed)
    bundle=load_bundle(ROOT/'data')
    p=from_bundle(bundle)
    bank=[list(bundle.record_board)]+[json.loads(f.read_text(encoding='utf-8'))['board'] for f in sorted((ROOT/'data/seeds').glob('*.json'))]
    test('official-nine-parents',p,bank,17,bundle)
    summary={label:{mode:sum(row['results'][label][mode] for row in held[:-1])
                    for mode in ('policy_gain','learned_gain','best_observed_gain')} for label in ('all','diverse','difficulty')}
    with path.open('x',encoding='utf-8') as stream:
        json.dump(dict(weights=w,training=training,heldout=held,summary=summary),stream)
        stream.write('\n')
    print(json.dumps(summary))

if __name__=='__main__':
    main(Path(sys.argv[1]))
