"""Inventory, flow and region deductions with sound decision-assumption explanations."""
from exact_rules import AdditionalRules
from explained_dfs import _Search, _bits


class _ExplainedRules(_Search):
    def __init__(self,*args,advanced_rules=True,**kwargs):
        if type(advanced_rules) is not bool:
            raise ValueError('advanced_rules must be boolean')
        super().__init__(*args,**kwargs)
        self.advanced_rules=advanced_rules
        self.rules=None
        self.counters.update(rule_work=0, rule_domain_copies=0)

    def setup(self):
        domains,reasons=super().setup()
        if self.advanced_rules:
            self.rules=AdditionalRules(self.problem,check=lambda:self.tick('rule_work'),
                                       options={'relational':True})
        return domains,reasons

    def rule_context(self,domains):
        return True

    def record_rule_effect(self,context,consistent,changed,filtered,domains):
        pass

    def propagate(self,domains,reasons):
        while True:
            conflict=super().propagate(domains,reasons)
            if conflict is not None or self.rules is None:
                return conflict
            context=self.rule_context(domains)
            if context is None:return None
            self.tick('rule_domain_copies',len(domains))
            filtered=domains.copy()
            consistent,changed=self.rules.apply(filtered)
            self.record_rule_effect(context,consistent,changed,filtered,domains)
            if consistent and not changed:
                return None
            # These assumptions jointly imply every current deletion. Any
            # necessary domain consequence is sound under the same premises.
            why=self.union(r for removed in reasons for r in removed.values())
            if not consistent:
                return self.conflict(why)
            for cell in changed:
                for state in _bits(domains[cell]&~filtered[cell]):
                    self.remove(cell,state,why,domains,reasons)

    def report(self,phase='searching'):
        result=super().report(phase)
        result['advanced_rule_counters']={} if self.rules is None else dict(self.rules.counters)
        return result


def solve(problem,seconds=30.,seed=0,workers=1,should_stop=None,on_progress=None,*,
          max_work=2000000,max_nogoods=256,include_nogoods=False):
    return _ExplainedRules(problem,seconds,seed,workers,should_stop,on_progress,True,max_work,
                           max_nogoods,64,include_nogoods,True).run()
