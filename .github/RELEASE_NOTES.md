Version **0.5.4** adds an optional DFS pruning rule for cells with two remaining placements. If edge colors and physical piece identities prove that each choice leads to a contradiction, DFS can skip both placements. Other branches keep their original search order.

| Regional comparison | Maximum filtering | With the new rule |
| --- | ---: | ---: |
| 24 regions on the five-clue 466 reference board | 6,910 attempts | 6,892 attempts |
| Hardest region alone, with a different random seed | 6,845 attempts | 6,829 attempts |

These counts include ordinary placements and lookahead trials. Both settings returned identical boards within each comparison. Two additional regions from different 465 boards tied at four and zero attempts. The hard-region witnesses scored **466/480** when checked against all 480 edges. The number of moves needed for a full solution is still unknown.

To use the rule, select **Advanced search → DFS → Maximum filtering + pivot proof (experimental) → 1 worker**. When switching from another DFS profile, select **Restart saved search** once to archive its frontier. [Settings and replay instructions](https://github.com/gearashi/EternityIISolver/blob/v0.5.4/docs/SEARCH_SETTINGS.md)

The command line now reads its version from the package metadata. Downloads include Windows x64, Linux x64, Intel Mac and Apple Silicon Mac builds, a Python wheel and a source archive.
