Version **0.5.6** adds an optional DFS rule for proving that a piece cannot occupy a particular location.

When DFS tries a cell with four possible states or a piece with three or four possible locations, the solver can test whether a choice forces a contradiction. If an oriented piece has only one legal neighbor on a required edge, every completion using the first placement must also use that neighbor. DFS follows those implications and skips a location only if every allowed rotation leads to a conflict.

| Regional task | Single-placement proofs | Forced-neighbor proofs |
| --- | ---: | ---: |
| 24 regions on the 466 reference board | 6,868 | 6,867 |
| Southeast 9×9 within those 24 | 6,858 | 6,857 |
| Same 9×9 and same parent, seed 20261123 | 6,809 | 6,808 |
| Separate regions from three 465 boards | 4, 0, 1 | 4, 0, 1 |

Both profiles returned the same first board on every regional comparison. The hardest region returned 466/480 under both seeds. Placement counts include ordinary DFS choices and tentative lookahead trials.

On the southeast 9×9 region, DFS considered placing tile #87 in row 11, column 13. Only one rotation was available. Two chains of forced neighbors required colors 4 and 6 on the same shared edge. The solver skipped that placement, then continued in the original order. It built 36 graphs and checked 119 choices on that run. The second seed encountered the same position and eliminated the same placement.

In a separate run requiring all 480 edges, both profiles reached the same checkpoint after 111 ordinary placements and 17 lookahead trials. The new rule checked 7 choices and skipped none in this opening segment.

To use this profile, select **Advanced search → DFS → Maximum filtering + forced-neighbor proofs (experimental) → 1 worker**. When switching from another DFS profile, select **Restart saved search** once to archive its frontier. [Settings and replay instructions](https://github.com/gearashi/EternityIISolver/blob/v0.5.6/docs/SEARCH_SETTINGS.md)
