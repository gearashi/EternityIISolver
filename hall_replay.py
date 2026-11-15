"""Share recheckable all-different shortage hints between local searches.

Only sets of cell indices are stored. A persisted row can reject an instance
only after that instance's current domains establish the Hall shortage again.
This makes facts safe across changes to domains, targets and earlier decisions
provided the caller still requires distinct physical pieces at these cells.
"""
import sqlite3
from contextlib import closing


def shortage(domains, cells, work):
    """Proof uses current oriented domains and piece identity state//4."""
    cells = tuple(cells)
    if len(cells) < 1 or len(set(cells)) != len(cells):
        return False
    n = len(domains)
    if any(type(c) is not int or c < 0 or c >= n for c in cells):
        return False
    union = 0
    for c in cells:
        d = domains[c]
        if type(d) is not int or d < 0 or d.bit_length() > 4 * n:
            raise ValueError('Domain must be a bounded oriented-state bitset')
        union |= d
        work[0] += 1
    pieces = union | union >> 1 | union >> 2 | union >> 3
    low_bits = ((1 << (4 * n)) - 1) // 15
    return (pieces & low_bits).bit_count() < len(cells)


class HallWitnesses:
    def __init__(self, count, limit=128):
        if type(count) is not int or not 1 <= count <= 256:
            raise ValueError('Invalid cell count')
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ValueError('Invalid hint capacity')
        self.count, self.limit = count, limit
        self.groups = {}
        self.reads = self.proof_work = self.hits = 0

    def record(self, domains, cells):
        if len(domains) != self.count:
            raise ValueError('Mismatched domain count')
        group = tuple(sorted(cells))
        work = [0]
        proved = shortage(domains, group, work)
        self.proof_work += work[0]
        if proved:
            self.remember(group)
        return proved

    def rejects(self, domains):
        if len(domains) != self.count:
            raise ValueError('Mismatched domain count')
        work = [0]
        for group in self.groups:
            self.reads += 1
            if shortage(domains, group, work):
                self.proof_work += work[0]
                self.hits += 1
                return True
        self.proof_work += work[0]
        return False

    def save(self, path):
        with closing(sqlite3.connect(path, timeout=10)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS hall_v1(n INTEGER, cells BLOB, PRIMARY KEY(n,cells))')
            db.executemany('INSERT OR IGNORE INTO hall_v1 VALUES(?,?)',
                           ((self.count,bytes(g)) for g in self.groups))
            # Keep the highest-priority short candidate proofs for each size.
            db.execute('DELETE FROM hall_v1 WHERE n=? AND rowid NOT IN '
                       '(SELECT rowid FROM hall_v1 WHERE n=? ORDER BY length(cells),cells LIMIT ?)',
                       (self.count,self.count,self.limit))

    def load(self, path):
        from pathlib import Path
        uri = Path(path).resolve().as_uri() + '?mode=ro'
        try:
            with closing(sqlite3.connect(uri, uri=True, timeout=10)) as db:
                rows = list(db.execute('SELECT cells FROM hall_v1 WHERE n=? AND length(cells) BETWEEN 1 AND ? '
                              'ORDER BY length(cells),cells LIMIT ?', (self.count,self.count,self.limit)))
        except sqlite3.DatabaseError:
            return False
        for (blob,) in rows:
            if type(blob) is bytes and blob and blob == bytes(sorted(set(blob))) and max(blob) < self.count:
                self.remember(tuple(blob))
        return True

    def remember(self, cells):
        group = tuple(sorted(set(cells)))
        if not group or any(type(c) is not int or not 0 <= c < self.count for c in group):
            return
        self.groups[group] = None
        if len(self.groups) > self.limit:
            del self.groups[next(iter(self.groups))]


def capture_shortages(search, witnesses):
    """Take cell hints from the explained engine's existing Hall proof."""
    if search.n != witnesses.count:
        raise ValueError('Mismatched instance and witness scopes')
    explain = search.hall_reason
    def hall_reason(cells, pieces, reasons):
        reason = explain(cells, pieces, reasons)
        if len(cells) > len(pieces):
            witnesses.remember(cells)
        return reason
    search.hall_reason = hall_reason


def root_rejects(problem, witnesses):
    """Optional check before constructing a native SAT/CP model."""
    # Raw domains are supersets of the frame/clue-filtered domains. A shortage
    # here is already sufficient. Include every raw state examined in setup.
    if not witnesses.groups:
        return False, 0
    masks = []
    work = 0
    for domain in problem.domains:
        mask = 0
        for state in domain:
            mask |= 1 << state
        masks.append(mask)
        work += len(domain) + 1
    return witnesses.rejects(masks), work


def install_matching_witnesses(search, witnesses):
    """Attach to one fresh DFS instance. Every hit replaces a matching check."""
    if search.n != witnesses.count:
        raise ValueError('Mismatched instance and witness scopes')
    original = search.matching
    def match(domains, previous):
        if witnesses.rejects(domains):
            search.matching_checks += 1
            search.matching_failures += 1
            return None
        return original(domains, previous)
    search.matching = match
