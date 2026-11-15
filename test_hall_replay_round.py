import itertools
from pathlib import Path
import random
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from hall_replay import HallWitnesses, capture_shortages, root_rejects, shortage
from test_explained_matching import hall_fixture, search
from exact_model import check_solution


class ReplayTests(unittest.TestCase):
    def test_candidate_hints_against_all_physical_assignments(self):
        rng = random.Random(65461)
        for n in range(2,7):
            for _ in range(40):
                ds = [rng.randrange(1<<(4*n)) for _ in range(n)]
                group = rng.sample(range(n),rng.randrange(1,n+1))
                impossible = shortage(ds,group,[0])
                if impossible:
                    self.assertFalse(any(all(ds[c] & (15 << (4*p)) for c,p in zip(group,perm))
                                         for perm in itertools.permutations(range(n),len(group))))
        self.assertFalse(shortage([1,16],[0,0],[0]))
        self.assertTrue(shortage([1,8],[0,1],[0]))
        self.assertFalse(shortage([1,16],[0,1],[0]))

    def test_produce_reopen_different_scope_and_corrupt(self):
        negative,_,_ = hall_fixture(True)
        positive,_,_ = hall_fixture(False)
        hints = HallWitnesses(25)
        producer = search(negative)
        capture_shortages(producer,hints)
        self.assertEqual(producer.run()['outcome'],'infeasible')
        self.assertTrue(hints.groups)
        with TemporaryDirectory() as temp:
            path = Path(temp)/'hints.db'
            hints.save(path)
            reopened = HallWitnesses(25)
            self.assertTrue(reopened.load(path))
            self.assertTrue(root_rejects(negative,reopened)[0])
            self.assertFalse(root_rejects(positive,reopened)[0])
            self.assertTrue(check_solution(positive,search(positive).run()['board']))
            path.write_bytes(b'invalid file')
            self.assertFalse(HallWitnesses(25).load(path))
            self.assertFalse(HallWitnesses(25).load(Path(temp)/'missing.db'))

    def test_duplicate_rows_eviction_and_invalid_blob(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'hints.db'
            for g in ([0], [1], [2], [2]):
                h=HallWitnesses(3,2)
                h.remember(g)
                h.save(path)
            db=sqlite3.connect(path)
            with db:
                self.assertEqual(db.execute('SELECT count(*) FROM hall_v1').fetchone()[0],2)
                db.execute('INSERT INTO hall_v1 VALUES(3,?)',(bytes([250]),))
            db.close()
            loaded=HallWitnesses(3,4)
            self.assertTrue(loaded.load(path))
            self.assertTrue(all(all(i<3 for i in g) for g in loaded.groups))
            self.assertFalse(loaded.rejects([1,16,256]))
