# Choosing a search for the five-clue puzzle

The release includes a disposition for each of the 67 proposals. [Improvement decisions](IMPROVEMENTS.md)

## Exact search for 480

For the fewest placement attempts observed on these five-clue regional tests, select **Advanced search → DFS → Maximum filtering + forced-neighbor proofs (experimental) → 1 worker**.

```text
EternityIISolver start --method exact --exact-engine dfs --pruning full-neighbor --workers 1
```

The earlier **Maximum filtering + single-placement proofs (experimental)** profile is available with this command.

```text
EternityIISolver start --method exact --exact-engine dfs --pruning full-dispatch --workers 1
```

The earlier **Maximum filtering + pivot proof (experimental)** profile is available with this command.

```text
EternityIISolver start --method exact --exact-engine dfs --pruning full-pivot --workers 1
```

When switching from a saved DFS profile, select **Restart saved search** once so the app archives the old frontier. Starting again with the same profile and seed resumes its own compatible checkpoint.

The DFS benchmark used the 256 official pieces, the five oriented clues, the Jef 466 reference and seed 20261007. A case freed a square 5×5, 7×7 or 9×9 corner. It required the originally matched edges in the declared scope so that a completion was known to exist. Each result passed a separate check of all physical pieces, fixed states, frame and selected edges.

### Proving a piece cannot occupy a location

When DFS tries a cell with four possible states or a piece with three or four possible locations, the solver can test whether a choice forces a contradiction. If an oriented piece has only one legal neighbor on a required edge, every completion using the first placement must also use that neighbor. DFS follows those implications and skips a location only if every allowed rotation leads to a conflict.

| Regional task | Single-placement proofs | Forced-neighbor proofs |
| --- | ---: | ---: |
| 24 regions on the 466 reference board | 6,868 | 6,867 |
| Southeast 9×9 within those 24 | 6,858 | 6,857 |
| Same 9×9 and same parent, seed 20261123 | 6,809 | 6,808 |
| Separate regions from three 465 boards | 4, 0, 1 | 4, 0, 1 |

On the southeast 9×9 region, DFS considered placing tile #87 in row 11, column 13. Only one rotation was available. Two chains of forced neighbors required colors 4 and 6 on the same shared edge. The solver skipped that placement, then continued in the original order. It built 36 graphs and checked 119 choices on that run. The second seed encountered the same position and eliminated the same placement.

Both profiles returned the same first board on every regional comparison. The hardest region returned 466/480 under both seeds. Placement counts include ordinary DFS choices and tentative lookahead trials.

In a separate run requiring all 480 edges, both profiles reached the same checkpoint after 111 ordinary placements and 17 lookahead trials. The new rule checked 7 choices and skipped none in this opening segment. With instrumentation enabled, building the seven graphs took 6.96 seconds and examined 2,485,972 oriented-state and required-edge pairs.

### Skipping one impossible orientation

When DFS chooses a cell with two remaining options, each option specifies a physical piece and a rotation. Trying one can force other placements through shared edge colors and the requirement to use each piece once. If those placements require the original cell to take its other option, DFS has proved the starting choice impossible. The single-placement profile skips that choice when its turn comes in the original branch order. DFS charges the usual placement if it tries the other option. The profile retains the existing rule for cases where both choices are impossible.

| Regional search | Pivot proof | Single-placement proofs |
| --- | ---: | ---: |
| 24 regions on the five-clue 466 reference, seed 20261007 | 6,892 | 6,868 |
| Southeast 9×9 within those 24 | 6,882 | 6,858 |
| Same southeast 9×9 and same parent, seed 20261123 | 6,829 | 6,809 |
| Separate regions from two 465 parents | 4 and 0 | 4 and 0 |

The table counts ordinary branch placements and lookahead trials. Across the 24 regions, ordinary placements fell from 6,288 to 6,264, and both profiles used 604 lookahead trials. All 24 first boards and the first boards in the repeat-seed and two 465-parent tests matched. The two southeast 9×9 runs returned boards scoring 466/480 against the full puzzle. The number of attempts needed to match all 480 edges remains unknown.

A fresh southeast 9×9 test used a third 465-point parent and seed 20261129. Both profiles returned the same board after 1 attempt. This case had no single-placement skips.

With all 480 edges required, both profiles reached the same checkpoint after 111 branch placements and 17 lookahead trials. The new rule skipped zero placements in that opening segment.

### Pruning both placements of a cell

Once DFS has chosen a cell with exactly two possible placements, it can sometimes prove that either choice leads to a contradiction. For example, using one physical piece here can force a neighboring cell to take its other piece. That placement can force another through its edge color. If both starting choices lead back to their own opposites, DFS skips both. It orders the original choices before checking the proof so that other branches keep their existing order. This option applies to single-worker DFS with Maximum filtering and no ordering hints.

| Verified regional searches | Maximum filtering | With pivot proof |
| --- | ---: | ---: |
| 24 reference regions, seed 20261007 | 6,306 branches + 604 probes | 6,288 branches + 604 probes |
| Southeast 9×9 alone, same seed | 6,303 + 597 | 6,285 + 597 |
| Southeast 9×9, seed 20261123 | 6,253 + 592 | 6,237 + 592 |
| Other reference boards, two regions | 4 attempts and 0 attempts | 4 attempts and 0 attempts |

Each comparison returned the same first board in both runs. On the original southeast 9×9, nine proofs eliminated 18 unsuccessful placements. The solver made 2,073 proof calls and spent about 0.415 seconds in the proof code in the instrumented run. The two regions from different 465 boards finished before the rule could help. The 9×9 searches reconstructed a legal 466 board under the restricted edge set. The number of placements required to solve the unrestricted 480-edge puzzle remains unknown.

### v0.5.3 independent-completion checks

When no required edge joins two undecided cells, DFS can use a matching of physical pieces to fill the remaining cells independently. If a required edge blocks that shortcut, DFS remembers it. On the next call it checks whether both endpoints still have multiple placements. If either endpoint has become decided, DFS runs the ordinary matching and edge checks again. A new search starts with an empty watch.

Across 24 original-piece regional tasks, we counted both methods along the same 6,910-placement search path. A separate run of the release candidate reproduced every v0.5.2 board and every DFS counter. It used 6,306 ordinary branches and 604 lookahead placements. Each board was independently verified against the printed pieces, frame, five fixed clues and required regional edges. The following counts describe the 2,944 calls to the independent-completion check.

| Work inside those calls | v0.5.2 | With the edge watch |
| --- | ---: | ---: |
| Matching validations | 2,944 | 2 |
| Cell classifications in scans of all domains | 753,664 | 512 |
| Checks of remembered endpoints | 0 | 5,884 |
| Reads of the edge watch | 0 | 2,944 |
| Stores of a blocking edge | 0 | 2 |

A separate run used the full puzzle with all five clues and all 480 edge constraints. It stopped at a fixed budget of 16 choices (five branches and eleven lookahead trials). The independent-completion check ran five times. Rechecking the remembered endpoints seven times reduced matching validations from five to two over this prefix. The effect on time to solve the full puzzle remains unknown.

### v0.5.2 work per lookahead placement

A trial can finish the board only when every cell has exactly one legal placement left. DFS now remembers a cell that prevented a previous trial from being complete and checks it first on the next trial. If that cell is still undecided, a single check is sufficient. If it has become fixed, DFS scans the remaining cells. The index belongs to one search instance. A resumed search starts with a fresh index.

On the same 24 original-piece regions, both versions returned identical independently checked boards and made **6,910 search choices** (ordinary branches plus speculative placements). The hard southeast 9×9 took **6,900 choices** in each version. Within the 508 lookahead completion checks, v0.5.1 tested 60,872 individual cell domains and v0.5.2 tested 889. The latter total includes 505 tests of the remembered cell and 384 tests in fallback scans.

Across nine small generated cases, individual domain tests fell from 193 to 169 overall. One case needed one more test when the previously undecided cell had become fixed. The measurements establish a reduction in repeated domain inspection on the tested regions. The best verified board in the comparison was the imported 466.

### v0.5.1 lookahead comparison

DFS checks whether a speculative placement has forced exactly one state at every cell. If it has, the usual solver validator checks the completed board. A resumed search applies this shortcut only after it has reconstructed its saved frontier.

With Maximum filtering and one worker, v0.5.0 and v0.5.1 each solved all 24 restricted tasks from scratch using the same seed and a 17-minute limit per task. Each version made 24 final solver validation calls, and an independent checker replayed every result against its physical pieces, clues, frame and regional edge constraints.

| Across 24 regions | v0.5.0 | v0.5.1 |
| --- | ---: | ---: |
| Ordinary DFS branches | 6,306 | 6,306 |
| Lookahead placements | 605 | 604 |
| Total placements | 6,911 | 6,910 |

The southwest 9×9 interior took two placements in v0.5.0 and one in v0.5.1. The matching placement totals on the other 23 cases include 6,900 placements per version on the hardest southeast 9×9 interior. Checking successful lookahead trials for completion added 508 scans containing 60,872 individual domain checks. The best verified five-clue board in this comparison was the imported 466.

### v0.5.0 comparison of five DFS profiles with checkpoint resume

On the 23 easier regions that all five profiles completed within 1.5 seconds, Maximum filtering used 11 decisions in total, Adaptive used 30, and Baseline, Strong and Independent regions used 31 each. These totals count trial placements used for lookahead. The hardest case freed the bottom-right 9×9 and enforced originally matched internal edges. The first trial gave each profile two minutes. The continuations gave each unfinished profile up to 15 additional minutes from its saved frontier. A repeated placement or lookahead probe during resume counted again.

| Hard 9×9 | Decisions to verified completion | Total elapsed |
| --- | ---: | ---: |
| Baseline | 20,327 | 58 s |
| Strong | 90,051 | 13m 59s |
| Maximum filtering | 6,941 | 8m 33s |
| Adaptive | 14,164 | 5m 46s |
| Independent regions | 20,347 | 5m 03s |

Replay the same 133-edge region from source with `python scripts/replay_reference_region.py --profile full --seconds 1020`. A fresh uninterrupted replay avoids the checkpoint reconstruction charged to the continuation.

On the hard task, each profile reconstructed the bundled 466 board cell for cell. An independent checker rebuilt the physical orientations from the raw tile file and verified all 133 selected edges. All five profiles eventually solved all 24 tasks. Summing placements across that fixed set gives 6,952 for Maximum filtering, 14,194 for Adaptive, 20,358 for Baseline, 20,378 for Independent regions and 90,082 for Strong. The hardest case accounts for most of each total.

In the v0.5.0 profile comparison on this i7-4790, Baseline reached the hard regional witness first. Maximum filtering spent more work per branch and needed fewer placements. A full five-clue 480 run has a different search tree. The fastest full-board profile and its time to solution are unknown.

### Other exact-search engines

**SAT with piece channels** uses the Glucose native solver with explicit piece-membership variables and pairwise clauses for small groups. On 24 held-out generated tasks it used 5,881 native decisions, compared with 6,358 for standard Glucose. It won on eight cases, lost on twelve and tied four. **Explained DFS with inventory rules** feeds justified color and regional deletions into conflict learning. **Regional master + conflict cores** shares the color of an interior edge between its two incident cells and caches checked local witnesses. On 49 completed small cases this coupling reduced local placements from 387 to 235 and native master branches from 1,434 to 1,041. These counters are specific to their engines.

## Improving a saved board

For restricted next-score repairs, choose **Advanced search → Matching + cycle repair → Regional assignment and price bounds (experimental) → 1 worker**. **GPU costs + CPU matching** runs placement-cost evaluation on a compatible GPU and executes the same regional proof on the CPU. CUDA and OpenCL were both exercised on an RTX 2060 with 6 GiB; testing the OpenCL path on that device does not establish performance on other vendors.

On 24 completed original-piece regional targets, the combined assignment, compatibility and integer-verified price bounds lowered candidate placements from 239 to 75 compared with assignment and compatibility alone. Explicit work rose from 29,562 to 277,899 operations with 606 additional simplex iterations. The eight attempts to improve the intact 466 within their selected small regions were infeasible. Deliberately damaged copies supplied positive repair controls.

**Guided rotation repair + witness reuse** expands supported rotations while retaining physical tile identity. Reuse of independently verified patches reduced native work on 12 repeated original-piece repair queries from 7,548 to 2,516 branches. Guided and baseline repair both stayed at 466 on the undamaged official comparison. The conventional GPU board repair and the hybrid sampling stage remain available. The new GPU construction trial improved the scores of random starts but saved zero downstream DFS decisions on the two measured original-piece tasks. A moves-per-second figure measures a different quantity from placements to a completed search.

## What the local proofs establish

The bundled 466 is Jef’s imported five-clue board. Every strict improvement from it requires at least nine changed cell states, including two orthogonal neighbors. A tile rotation counts as a changed state. Using moves that change at most two cells, the reachable component at scores 465 and higher has ten boards. Allowing score 464 expands the component to 285 boards and still produces no exit to 467. The score-463 component has 3,785 boards. Lowering the floor to 462 gives 65,875 reachable boards, still with no exit to 467. Such a path must reach 461 or lower.

The radius solver now bounds each connected group of changed cells, including groups that lose matches. It discards combinations whose best possible total gain is zero or lower. On this board at radius seven, two exact methods agreed that no gain existed. The component accounting reduced the checked complete cell sets from **15,931,447 to 1,646,927**. At radius eight, it covered 60,271,297 possible combinations, evaluated 24,313,897 complete cell sets and found no gain.

A perfect board requires **at least 24 changed cell states** from this 466. Ten overlapping regions have certified minimum change counts. Their integer-weighted sum is 376, and each cell can contribute at most 16. Rounding 376/16 up gives 24. A separate checker reconstructs clockwise tile orientations from the printed sides and checks the 5,961 smaller subsets directly. [Certificate for the public reference](record466-distance.json)

```text
python scripts/verify_distance_certificate.py
EternityIISolver perfect-distance board.json
EternityIISolver repair-radius board.json --radius 8 --seconds 900
EternityIISolver repair-patches board.json first-patch.json second-patch.json
EternityIISolver repair-parent-pair board.json other-board.json third-board.json
```

The native app includes the C++ helper for radii four through eight. For a Python installation, build it from the source distribution using `python scripts/build_radius_helper.py` with a C++17 compiler, then supply the executable path with `--helper`. Every radius result reports how far the proof completed. A deadline or work cap leaves the rest of that neighborhood open.

All these commands work offline. The dashboard can fetch the public board archive after an explicit **Download and merge** request. **Export best** writes local files for manual review.
