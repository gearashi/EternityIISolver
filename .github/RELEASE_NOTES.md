Version **0.5.5** adds an optional DFS profile that can skip a specific piece and rotation when a cell has two remaining choices. It uses edge colors and physical piece identities to prove that choice fails. DFS keeps its original order and counts a placement whenever it actually tries the other choice.

| Five-clue regional task | v0.5.4 pivot proof | Single-placement proofs |
| --- | ---: | ---: |
| 24 regions on the 466 reference board | 6,892 attempts | 6,868 attempts |
| Hardest region within those 24 | 6,882 | 6,858 |
| Same region and parent with a second seed | 6,829 | 6,809 |

Each comparison returned the same first board. Two regional searches on separate 465 parents tied at four and zero attempts. The hardest region returned 466/480 under both seeds. The counts include lookahead placements.

A fresh southeast 9×9 test used a third 465-point parent and seed 20261129. Both profiles returned the same board after 1 attempt. This case had no single-placement skips.

With all 480 edges required, both profiles reached the same checkpoint after 111 branch placements and 17 lookahead trials. The new rule skipped zero placements in that opening segment.

To use this profile, select **Advanced search → DFS → Maximum filtering + single-placement proofs (experimental) → 1 worker**. When switching from another DFS profile, select **Restart saved search** once to archive its frontier. [Settings and replay instructions](https://github.com/gearashi/EternityIISolver/blob/v0.5.5/docs/SEARCH_SETTINGS.md)
