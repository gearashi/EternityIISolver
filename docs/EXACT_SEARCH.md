# Offline CPU exact search

The dashboard's **CPU exact / hybrid** method looks for a complete solution to the original 16×16 puzzle with all five official clues in the application's canonical orientation. It enforces all 256 pieces exactly once, legal rotations, the gray frame, and all 480 internal edge matches. It does not promote partially matching intermediate assignments; the dashboard keeps displaying the independently validated saved best board until a full solution is found.

Choose **DFS**, **SAT**, or **CP-SAT**, then Start. DFS is the recommended fallback when native constraint encoding costs too much time or memory. It is a pruned Python baseline, not an optimized native DFS implementation. No engine is established as universally fastest, and none has solved the whole puzzle in the bounded comparisons described below.

An additional **Hybrid GPU hints + DFS** option is experimental. It samples on the GPU briefly, then runs the same exact CPU DFS pruning with a different branch preference. It is not the default or an established speed improvement.

CPU exact search is standalone and offline. It is not a BOINC replacement worker, does not consume production ticket assignments, and earns no BOINC credit. The separate GPU mode remains a heuristic board-repair search.

Starting in v0.2.1, the dashboard can independently download public library data when its **Library downloads** checkbox is enabled; fresh installs leave it off. Search workers still make no external requests and send no progress, results, node/work counts, or telemetry. This optional downloader does not change the search algorithms or measurements below. Versions v0.1.1 and v0.2.0 disabled runtime library downloads.

## Engines

| Engine | Model and pruning | Workers | Stop and restart |
| --- | --- | --- | --- |
| DFS | Piece/rotation domains; adjacency consistency; forced piece exclusion; unique piece locations; bipartite matching feasibility | 1 | Saves a compatible frontier and resumes it |
| SAT | Boolean placement choices; exactly one placement per cell and one use per piece; shared edge-color constraints; native Glucose solver through PySAT | 1 | Rebuilds the encoding and starts a fresh native search |
| CP-SAT | Allowed tables linking each cell's piece to its four colors; all-different piece IDs; equality of adjacent colors; native OR-Tools solver | 1–4 | Rebuilds the model and starts a fresh native search |
| Hybrid | Short GPU sampling stage; at most 64 validated frozen hints order branches in the same exact DFS | 1 CPU worker after GPU sampling | Reuses its hints and compatible separate DFS frontier |

All engines share the same legal domains and constraint-edge definition. Rotations that give identical colors can share a projected CP-SAT table entry; a returned state is recovered from the cell's allowed domain, retaining any fixed rotation. Every returned solution is independently checked against the original exact task before the runtime may promote it.

The DFS feasibility checks only remove assignments that cannot complete the specified task. They reduce work but do not make the full puzzle tractable by assumption. SAT and CP-SAT include model construction in their time budget; a large encoding can consume the budget before much native search occurs. More CP-SAT workers are not guaranteed to improve a particular instance.

## Experimental GPU sampling followed by CPU DFS

On a fresh hybrid search, the GPU generates and scores complete 256-piece arrangements for a short **3-second generation budget**. GPU initialization and kernel compilation take additional time, as do the final selected transfer and CPU screening. Low-scoring arrangements are accepted as possible hints; high matching scores are not required. The CPU validates a selected, diverse set of at most **64** arrangements, freezes that set, and releases GPU resources before continuing with one DFS worker. The worker's optional `--seconds` limit covers the whole run cooperatively; it can shorten the sampling stage, and a device operation already in progress may delay stopping.

The retained placements vote on which legal branches to try first, breaking ties after the existing value-support ordering. **Votes never remove a candidate or declare a branch impossible.** Frame/clue constraints, adjacency propagation, piece uniqueness, and matching feasibility continue to provide the exact pruning. Every full solution is independently validated. Freezing the hints keeps branch ordering compatible with the saved hybrid DFS frontier; subsequent starts reuse the hints instead of repeating GPU sampling.

This does not launch exhaustive CPU search for every generated board. The CPU screens the small retained set and runs one exact search. It does not exhaustively prune 100,000 full boards per second. The requested **100,000 samples/second** target must be evaluated against measured throughput on the selected hardware; it is not promised and does not imply a comparable improvement in solution time.

The dashboard reports `sampled_boards`, `sampling_boards_per_second`, and `hybrid_hint_count` separately from CPU branches/conflicts. A complete sample places all pieces; it need not match all edges. Generated samples may repeat, and the count does not mean that all samples were CPU-validated. The final batch rate includes generation, GPU scoring, selected transfers, and CPU screening; it excludes GPU compilation and initialization, which are recorded separately. The live rate during generation is provisional until final transfer and screening finish. After sampling ends, these statistics describe the completed sampling stage rather than ongoing GPU work. None of these counters represents BOINC work or credit.

Use **GPU sample batch** and the CUDA/OpenCL backend selector to configure this stage. The default batch is 4,096; larger batches are not guaranteed to improve throughput or hint quality. The core DFS/SAT/CP-SAT comparison below tests CPU-only engines and does not establish a hybrid speedup.

### Measured GPU sampling: 9 October 2026

Bounded tests on one **NVIDIA GeForce RTX 2060**, using batches of 4,096 and a 3-second generation budget, exceeded the requested 100,000-sample/second target. The final rate includes generation, scoring, the selected transfer, and CPU screening; GPU compilation and initialization are excluded. Each run followed a tiny correctness diagnostic, so these rates do not measure a cold application launch.

| Backend | Generated full samples | Measured stage time | Final samples / second | Retained hint scores / 480 |
| --- | ---: | ---: | ---: | ---: |
| [CUDA](evidence/hybrid-sampling-cuda-20261009.json) | 37,736,448 | 3.042 s | 12.40 million | 22–73 |
| [OpenCL](evidence/hybrid-sampling-opencl-20261009.json) | 35,184,640 | 3.049 s | 11.54 million | 29–73 |

Both runs passed the separate 64-board correctness diagnostic and retained 64 CPU-validated hints. Millions of generated samples were not individually transferred to the CPU or checked for uniqueness. The low retained scores are expected for random arrangements. These tests left the verified best at **466/480** and demonstrate sampling throughput, not a speedup in exact solution search. Results on other GPUs, drivers, batch sizes, or concurrent loads may differ. Hybrid performs this stage only when it needs fresh hints, then continues with CPU DFS; it does not keep generating at this rate throughout the exact search.

### Equal CPU budgets with and without hints

A separate [paired full-puzzle comparison](evidence/hybrid-ordering-20261009.json) used the 64 validated CUDA hints above, the same five clues and all 480 edge constraints, seed 17, and one worker in isolated sequential processes. Both variants received a **10-second wall-clock budget for CPU search**. Both revalidated the hint pool during untimed preparation; the earlier GPU sampling and setup were not charged to either CPU budget, so this was not an end-to-end timing comparison.

| DFS variant | Outcome | Branches | Deepest placement | Peak process memory |
| --- | --- | ---: | ---: | ---: |
| Ordinary branch ordering | Timeout | 897 | 110 | 31.4 MiB |
| GPU-hint tie-breaking | Timeout | 835 | 111 | 32.8 MiB |

The hints changed traversal. Neither variant solved the puzzle, and these operation counts and depths do not establish better pruning or faster completion. The verified best remained **466/480**. This single pair does not justify recommending hybrid over ordinary DFS.

To reproduce the bounded CPU comparison using the saved hint pool:

```sh
python scripts/benchmark_hybrid_ordering.py --hints docs/evidence/hybrid-sampling-cuda-20261009.json --seconds 10 --seed 17 --output hybrid-ordering.json
```

## Commands and saved state

Native application commands:

```text
EternityIISolver start --method exact --exact-engine dfs --workers 1
EternityIISolver run --method exact --exact-engine sat --workers 1 --seconds 60
EternityIISolver exact --engine cp-sat --workers 2 --seconds 60
EternityIISolver start --method exact --exact-engine hybrid --backend auto --replicas 4096
EternityIISolver exact --engine hybrid --backend auto --replicas 4096 --seconds 60
EternityIISolver stop
EternityIISolver status --json
EternityIISolver diagnose-exact
EternityIISolver diagnose-hybrid --backend cuda --seconds 3 --replicas 4096
```

Use `EternityIISolver.exe` on Windows or the platform's executable path. In a source checkout, replace `EternityIISolver` with `python launcher.py`. Python 3.11+ and `python -m pip install .` install the three CPU engines, including OR-Tools 9.15 and python-sat. Native archives bundle these dependencies. DFS, SAT, and CP-SAT do not allocate a GPU or require a GPU driver. A fresh hybrid sampling stage requires a supported GPU backend; source CUDA support uses the optional `.[cuda]` installation.

`diagnose-hybrid` first checks two tiny 32-board GPU batches on the CPU, then measures generation at the requested batch size and duration. It does not start ongoing DFS. Use `--backend opencl` for an OpenCL GPU and `--output PATH` to save its measurements. The diagnostic records generation/screening time separately from GPU compilation and initialization.

`start` opens the local dashboard and starts a worker. `run` and `exact` run in the foreground. `--seconds` bounds a run; without it, the exact worker continues until a solution, exhaustion of its search scope, a stop request, or an error. Closing the browser leaves the worker running. Stop requests a cooperative stop; it may take time to finish the current operation and write local state.

Library downloads have their own checkbox and continue independently while the local dashboard server runs, including after search Stop or closing the browser tab. Disable that checkbox to stop further downloads; the current GET may finish. The downloader checks the public index every 15 minutes and requests missing public arrangements at least 1 second apart, reusing cached content. Its fixed-endpoint GET requests carry no uploads or solver metrics, though the server can log the requests, IP address, and user-agent. See [library-download behavior](../README.md#saved-state-and-library-downloads) for the network boundary.

The state directory retains the saved best board, exports, cache, GPU repair checkpoints, ordinary DFS checkpoints, and separate hybrid hints/frontiers when switching methods. DFS resumes a compatible saved frontier. Hybrid reuses its frozen hints and compatible frontier. SAT and CP-SAT preserve run status but do not serialize their native search state; their next run starts again. Checkpoints are local recovery files. Exhaustion of a restored frontier is not an independent unsatisfiability certificate, particularly if a checkpoint has been edited or replaced.

If a DFS checkpoint is incompatible or you deliberately want to begin again, stop the active worker and use `EternityIISolver exact --engine dfs --fresh-exact`. This archives the old DFS checkpoint before starting from the full puzzle; it retains the saved best board and exports.

For hybrid, `EternityIISolver exact --engine hybrid --fresh-exact --backend auto --replicas 4096` archives that engine's previous hints and DFS checkpoint, then starts a new sampling stage. It retains the ordinary DFS and GPU-repair saved states as well as the best board and exports.

## Reading progress and outcomes

- **Branches** and **conflicts** are engine-specific CPU counters. They are not interchangeable with GPU swap attempts or BOINC DFS nodes. Native solver counters may only update when a run ends.
- **Deepest placement** is a DFS statistic, not the number of correctly matched edges. It does not alter the saved best score.
- **Preparing constraints** means the engine is constructing or simplifying its model. **Searching** means it has entered the search phase.
- **Solved** requires an independently validated board satisfying every constraint of the task. In normal application mode, that means 480/480 with all five clues.
- **Timeout** or **stopped** is inconclusive. It never means the puzzle is impossible.
- **Infeasible** means a completed exact search ruled out its specified task, subject to the solver and input assumptions. No portable UNSAT proof certificate is produced. A regional task or resumed frontier cannot establish whole-puzzle impossibility.

The dashboard's Export best board control still saves local candidate files for manual review. Neither the search workers nor the library downloader uploads them or changes the project's submission requirements. Current duplicate checks use the shared cache; historical “not found” labels describe the snapshot at check time and do not establish worldwide novelty.

## Bounded comparison and reproducibility

The comparison script uses identical task definitions for all three engines, one CPU worker per engine, and separate processes. It rotates engine order between cases and records source hashes, the Git revision and dirty state, Python/dependency versions, task hashes, elapsed time, peak process memory, solver counters, and independent witness validation. Encoding and search time both count against the per-case budget. A soft peak-memory limit requests a stop; the parent also applies a process timeout. These are resource controls, not a benchmark sandbox.

From the source checkout:

```sh
python scripts/benchmark_exact.py --seconds 5 --full-seconds 30 --memory-mb 2048 --output exact-comparison.json
```

To restrict the comparison, use `--engines dfs sat cp-sat` and `--cases generated-4 region-4 whole-five-clue`, or select other listed cases with `--help`. This command deliberately runs bounded CPU searches; normal dashboard startup does not run benchmarks.

The cases have different scopes:

| Case family | What a result establishes |
| --- | --- |
| `generated-4`, `generated-6`, `generated-8` | Performance on synthetic known-solvable 4×4, 6×6, and 8×8 tasks, independently checked |
| `region-4`, `region-6`, `region-8`, `region-12` | Feasibility of repairing selected regions of the actual 466 board with outside placements fixed |
| `whole-five-clue` | Search of the full original five-clue puzzle with all 480 equalities |

Regional tasks constrain every edge touching the selected region. They intentionally omit frozen-to-frozen edges, some of which already mismatch. A solved regional task therefore need not be a solved full board; the report separately records the full board's score. Regional infeasibility only rules out that region with those outside placements fixed.

### Measured comparison: 9 October 2026

The final run used Windows x64, Python 3.12.14, OR-Tools 9.15.6755, and python-sat 1.9.dev15. Each engine had one worker, a 5-second budget for each small task, a 30-second budget for the full puzzle, and a 2,048 MiB soft memory limit. The table shows solver elapsed seconds, including model construction but excluding process startup and common task creation. Every returned synthetic solution passed independent validation.

| Task | Outcome for all three | DFS (s) | SAT (s) | CP-SAT (s) |
| --- | --- | ---: | ---: | ---: |
| Synthetic 4×4 | Solved | 0.009 | 0.031 | 0.672 |
| Synthetic 6×6 | Solved | 0.015 | 0.047 | 0.671 |
| Synthetic 8×8 | Solved | 0.051 | 0.125 | 1.016 |
| Fixed-outside 4×4 region | Infeasible | 0.017 | 0.047 | 0.656 |
| Fixed-outside 6×6 region | Infeasible | 0.017 | 0.079 | 0.656 |
| Fixed-outside 8×8 region | Infeasible | 0.036 | 0.203 | 0.718 |
| Fixed-outside 12×12 region | Infeasible | 0.059 | 0.813 | 1.110 |
| Full five-clue puzzle | Timeout; inconclusive | 30.000 | 30.188 | 30.093 |

Peak process memory for the full-puzzle runs was **33.7 MiB for DFS**, **312.1 MiB for SAT**, and **367.2 MiB for CP-SAT**. Small overruns reflect cooperative cancellation and final result handling. DFS was fastest on the seven tested small tasks, but all three full-puzzle runs remained unresolved. DFS rejected all four regional tasks during initial propagation, before branching or a bipartite matching check; those cases provide no evidence that Hall matching caused the speed advantage.

The [raw comparison report](evidence/exact-comparison-20261009.json) has SHA-256 `552b2e503f0a1a2cb8b5094539fb74ef1feff5625c025e0aef2e69af1a0d445d`. It records the exact source hashes because the tested checkout contained uncommitted release changes.

These comparisons are exploratory: desktop and BOINC load was not controlled, and Python DFS is being compared with native SAT/CP-SAT implementations plus their encodings. Synthetic wins do not establish whole-puzzle performance. Branch and conflict counts are engine-specific and are not speedup ratios. Whole-puzzle timeouts do not establish impossibility, and these short runs do not establish a universally fastest engine.

### Controlled pruning comparison

The separate pruning experiment kept the same DFS code within its paired runs and disabled only its bipartite matching feasibility test. This test detects shortages such as thirteen cells sharing only twelve possible pieces, even when every individual cell still has candidates. Other pruning and the search seed remained unchanged. Each variant ran sequentially in an isolated process with one worker: 3 seconds per small case and 10 seconds for the full puzzle.

| Task | Matching enabled | Matching disabled |
| --- | --- | --- |
| Full five-clue puzzle | Timeout at 10 s; 901 branches; 31.1 MiB peak | Timeout at 10 s; 913 branches; 31.4 MiB peak |
| Harder synthetic 10×10 and 12×12 | Both timed out at 3 s | Both timed out at 3 s |
| Artificial piece-shortage 8×8 | Infeasible in 0.010 s, before branching | Timed out at 3 s after 30,227 branches |
| Fixed-outside 6×6 and 12×12 regions | Both infeasible during initial propagation | Both infeasible during initial propagation |

On the full puzzle, matching found 3 infeasible assignments in 477 checks, using 0.220 seconds, about 2.2% of the run's budget. Initial propagation reduced candidate placements from 149,081 to 131,311 in both variants. Neither completed, so the similar branch counts do not identify a performance winner. The harder synthetic cases produced 76 and 48 matching failures respectively, but still gave no completed timing comparison.

The artificial 8×8 case deliberately restricts thirteen edge cells to twelve pieces and omits adjacency constraints. It demonstrates that this rule can reject a large dead end immediately; it does not measure a speedup on Eternity II. Both actual regional cases failed before matching was called. Counts of failed checks are observed operations, not estimates of how many search subtrees were eliminated. The current DFS retains matching; this short ablation has not established a faster whole-puzzle alternative.

Reproduce the bounded experiment with:

```sh
python scripts/benchmark_pruning.py --seconds 3 --whole-seconds 10 --memory-mb 2048 --seed 17 --output pruning-comparison.json
```

The [raw pruning report](evidence/pruning-comparison-20261009.json) has SHA-256 `9d1c8ab7235c66bc643ad68c47f39738f7644e305e7a687c10198a49d8b11283`. It records the task and source hashes, rule timings, and limitations. These single runs share the uncontrolled desktop-load limitation of the main comparison.

The [finite-computation manifest](evidence/exact-computation-manifest.json) records the main comparison inputs, source hashes, resource bounds and residual limitations. It is a validated legacy version1 provenance record, not a mathematical proof.
