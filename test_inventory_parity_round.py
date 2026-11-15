"""Exhaustive controls for parity support and equivalent-rotation projection."""
from dataclasses import replace
from itertools import product
import random
import unittest

from exact_model import generated_problem, check_solution
from inventory_parity import CheckerboardParity, forced_variables
from residual_rotations import quotient_rotations


class InventoryChecks(unittest.TestCase):
    def test_affine_fixed_coordinates_match_complete_enumeration(self):
        rng = random.Random(85423)
        for n in range(1, 8):
            for _ in range(80):
                rows = [rng.randrange(2 << n) for _ in range(rng.randrange(2 * n + 2))]
                assignments = [x for x in range(1 << n) if all(
                    ((r & x).bit_count() % 2) == (r >> n) for r in rows)]
                expected = None if not assignments else {
                    i: assignments[0] >> i & 1 for i in range(n)
                    if len({x >> i & 1 for x in assignments}) == 1}
                self.assertEqual(forced_variables(rows, n, lambda: None), expected)

    def test_board_certificates_do_not_remove_tiny_solutions(self):
        for seed in range(32):
            p, known = generated_problem(2, 3, seed, clue_count=seed % 2)
            for limit in (0, 8, 10000):
                bits = [sum(1 << s for s in row) for row in p.domains]
                filtered, work = CheckerboardParity(p).apply(bits, limit=limit)
                self.assertLessEqual(work, limit)
                if filtered is None:
                    self.assertLess(limit, 10000)
                    continue
                self.assertTrue(filtered)
                for board in product(*p.domains):
                    if check_solution(p, board):
                        self.assertTrue(all(d & 1 << s for d, s in zip(filtered, board)))
                self.assertTrue(check_solution(p, known))

    def test_incompatible_color_total_and_unsupported_edge_subset(self):
        p, known = generated_problem(2, 3, 7)
        faces = list(p.faces)
        for i in range(4):
            side = next(j for j in range(4) if p.faces[0][j])
            faces[i] = tuple(81 if j == (side + i) % 4 else v for j, v in enumerate(faces[i]))
        odd = replace(p, faces=tuple(faces))
        bits = [sum(1 << s for s in row) for row in p.domains]
        self.assertIs(CheckerboardParity(odd).apply(bits)[0], False)
        self.assertIsNone(CheckerboardParity(replace(odd, edges=())).apply(bits)[0])
        self.assertIsNone(CheckerboardParity(p).apply(bits, should_stop=lambda: True)[0])

    def test_equivalent_rotations_preserve_all_distinct_solution_signatures(self):
        for seed in range(16):
            p, known = generated_problem(2, 2, seed, clue_count=seed % 2)
            from exact_model import constraint_edges
            for edges in (constraint_edges(p), constraint_edges(p)[::2], ()):
                task = replace(p, edges=edges)
                reduced, _, _ = quotient_rotations(task)
                def signatures(q):
                    return {(tuple(s // 4 for s in b),
                             tuple(q.faces[b[a]][sa] for a, _, sa, _ in edges),
                             tuple(b[c] for c in sorted(p.fixed))) for b in product(*q.domains)
                            if check_solution(q, b)}
                self.assertEqual(signatures(task), signatures(reduced))

    def test_ambiguous_orientation_at_fixed_cell(self):
        p, _ = generated_problem(1, 1, 1)
        task = replace(p, fixed={0: 3}, domains=((0, 1, 2, 3),))
        self.assertEqual(quotient_rotations(task)[0].domains, ((3,),))
        unrestricted = replace(task, fixed={})
        self.assertEqual(quotient_rotations(unrestricted)[:2], (replace(unrestricted, domains=((0,),)), 3))

if __name__ == '__main__':
    unittest.main()
