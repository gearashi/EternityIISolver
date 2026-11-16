Version **0.5.7** adds neighbor capacity proofs to single-worker DFS on the five-clue puzzle. The rule skips a placement when the required neighboring squares have too few distinct physical pieces available. For example, if three neighboring cells can use only two tile IDs, multiple rotations of those tiles cannot fill the third cell.

| Regional search | Forced-neighbor proofs | Neighbor capacity proofs |
| --- | ---: | ---: |
| 24 regions on the 466 reference board | 6,867 | 6,857 |
| Same southeast 9×9 region, second seed | 6,808 | 6,796 |
| Independent 9×9 window on a 465 board | 13,337 | 13,325 |
| Three other regional controls | 4, 0, 1 | 4, 0, 1 |

The counts include DFS branches, lookahead probes and any replayed placements. Across the 29 paired searches, the rule avoided 34 attempts. Independent verification found 24 distinct combinations of parent domains and excluded choices. Both profiles returned the same first board in each pair. These regional runs used explicitly selected edges. The verified boards scored 465 or 466 out of 480 overall.

To try it, select **Advanced search → DFS → Maximum filtering + neighbor capacity proofs (experimental) → 1 worker**. When switching profiles, select **Restart saved search** once to archive the previous frontier. [Search settings and replay instructions](https://github.com/gearashi/EternityIISolver/blob/v0.5.7/docs/SEARCH_SETTINGS.md)
