"""Persistent local CDCL oracle. All changing regional choices are assumptions."""
from collections import defaultdict
import threading


class RegionalSAT:
    def __init__(self,problem,legal,cells,internal,incident,tick,counts):
        from pysat.solvers import Glucose4
        self.native=Glucose4()
        self.problem=problem;self.tick=tick;self.counts=counts
        self.state_vars={};self.piece_vars={};self.edge_vars={};self.last=0
        try:self._encode(legal,cells,internal,incident)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.native is not None:self.native.delete();self.native=None

    def variable(self):
        self.tick();self.last+=1;return self.last

    def clause(self,lits):
        self.tick(1+len(lits));self.native.add_clause(lits)

    def at_most_one(self,lits):
        # A linear sequential prefix witness. Each true input excludes every
        # subsequent input. No auxiliary literal is used as a puzzle choice.
        if len(lits)<2:return
        if len(lits)==2:self.clause([-v for v in lits]);return
        last=None
        for lit in lits[:-1]:
            prefix=self.variable();self.clause([-lit,prefix])
            if last is not None:
                self.clause([-last,prefix]);self.clause([-lit,-last])
            last=prefix
        self.clause([-lits[-1],-last])

    def _encode(self,legal,cells,internal,incident):
        pieces=defaultdict(list)
        for cell in cells:
            values={}
            for state in legal[cell]:
                lit=self.variable();values[state]=lit;pieces[state//4].append(lit)
            self.state_vars[cell]=values
            self.clause(list(values.values()));self.at_most_one(list(values.values()))
        for piece,lits in pieces.items():
            self.at_most_one(lits)
            allowed=self.variable();self.piece_vars[piece]=allowed
            for lit in lits:self.clause([-lit,allowed])
        def colors(cell,side,channels):
            for state,lit in self.state_vars[cell].items():
                color=self.problem.faces[state][side]
                if color not in channels:channels[color]=self.variable()
                self.clause([-lit,channels[color]])
        for a,b,sa,sb in internal:
            channels={}
            colors(a,sa,channels);colors(b,sb,channels)
            self.at_most_one(list(channels.values()))
        for edge,cell,side in incident:
            channels={}
            colors(cell,side,channels)
            self.at_most_one(list(channels.values()))
            for color,lit in channels.items():self.edge_vars[edge,color]=lit

    def query(self,pool,boundary,check,progress):
        assumptions=[];labels={}
        pool=set(pool)
        for piece,lit in self.piece_vars.items():
            self.tick()
            if piece not in pool:assumptions.append(-lit);labels[-lit]=~piece
        for choices in boundary.values():
            for _,color,edge in choices:
                self.tick()
                lit=self.edge_vars.get((edge,color))
                if lit is None:return None,{edge}
                assumptions.append(lit);labels[lit]=edge
        finished=threading.Event();failure=[]
        def monitor():
            while not finished.wait(.02):
                try:check();progress()
                except BaseException as error:
                    failure.append(error);self.native.interrupt();return
        before=self.native.accum_stats()
        monitor_thread=threading.Thread(target=monitor,daemon=True)
        monitor_thread.start()
        try:answer=self.native.solve_limited(assumptions=assumptions,expect_interrupt=True)
        finally:finished.set();monitor_thread.join()
        after=self.native.accum_stats()
        for stat in ('decisions','conflicts','propagations','restarts'):
            self.counts['local_sat_'+stat]+=after[stat]-before[stat]
        if failure:raise failure[0]
        if answer is None:raise RuntimeError('Unexpected local SAT interruption')
        if not answer:
            core=self.native.get_core() or []
            self.tick(1+len(core));return None,{labels[lit] for lit in core}
        model=self.native.get_model()
        self.tick(len(model));positive={lit for lit in model if lit>0}
        board={}
        for cell,values in self.state_vars.items():
            self.tick(len(values))
            board[cell]=next(state for state,lit in values.items() if lit in positive)
        return board,set()
