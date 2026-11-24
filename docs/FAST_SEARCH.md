# Fast compiled search

The app includes a compiled C++ search core and two engines that use it. Both run offline on the CPU, keep the five official clues fixed, and are started from **Advanced search** in the dashboard or from the command line.

| Engine | Goal | Workers | Stop and restart |
| --- | --- | --- | --- |
| **Fast compiled search for the highest score** (`fast-score`) | Complete boards with as many matching edges as possible | 1–64 threads | Continues from its checkpoints; keeps the best board of the search |
| **Fast compiled DFS for all 480** (`fast-dfs`) | A complete 480/480 solution, by exhaustive backtracking | 1–64 threads | Continues from its checkpoints with the same seed and worker count |

```sh
EternityIISolver exact --engine fast-score --workers 4
EternityIISolver exact --engine fast-dfs --workers 4 --seconds 3600
```

Neither engine has solved the puzzle. The speed below is the number of pieces tried per second, which is far higher than the Python DFS, but the search space remains far beyond exhaustive search. A stop, a time limit or a long run without a result is inconclusive.

## How the search works

The core fills the board in row order, left to right and top to bottom. For each cell it looks up the pieces whose left and top colors match the two neighbors already placed, in a table built once per run. Border cells, clue cells and the cells next to a clue have their own tables, so gray stays on the frame and every edge of a fixed clue is enforced when its neighbor is placed. A 256-entry array records which pieces are in use. No other pruning is applied. There is no memory allocation or system call in the inner loop.

**Fast compiled DFS** requires every edge to match. All threads share one candidate order (shuffled from the seed) and divide the first four cells' choices between them, so no branch is searched twice. If the search ever ended it would have tried every arrangement; on the full puzzle it does not end.

**Fast compiled score search** adds three things.

- *A mismatch budget.* A fixed number of mismatching edges is allowed, the k-th only at or after a set cell late in the scan. A cell never takes two mismatches. Every complete board found under a budget of B therefore scores at least 480 − B.
- *Color depletion.* Three inner colors (2, 18 and 20 in `data/pieces.txt` numbering) must be used up early: at least 80% of their 146 sides within the first 160 cells, most of them in the first rows. Pieces with more of those sides are tried first. This leaves the last rows a piece set with fewer colors to reconcile. The idea is known from earlier public Eternity II solvers; the schedule here was measured on this puzzle.
- *Restarts.* Each thread reshuffles its candidate order and starts again after 200 million placements, so threads explore unrelated parts of the space.

The engine starts with the loosest budget (22 mismatches, so any complete board scores 458 or more). Whenever it finds a complete board with score S, it tightens the budget so that the next board must score at least S + 1, and continues. Every independently validated arrangement is saved under `results/` and registered in the local library. The dashboard and **Export best board** use the best board of the current or most recent fast-score run, including scores below 466. Your overall best (the 466 reference, or your own better board) is replaced only by a higher score.

Depletion and the budget are heuristics. They remove branches that might contain better boards, so a score run never shows that a better board does not exist.

## Validation

The compiled helper only proposes boards. Each board file is re-scored in Python from the piece list (all 256 pieces once, gray on the frame, the five clues, matching edges counted), and a board whose claimed score differs is refused. The worker then validates it again with the same validator used for every other engine before it can replace the saved best. Only a validated 480/480 board is reported as solved.

## Measurements

A bounded measurement on 17 November 2026 used a cloud machine with four virtual CPUs (Intel Xeon, 2.1 GHz), the GCC build, four workers and seed 20261007. Each engine started from empty state. [Evidence](evidence/fast-search-cloud-20261117.json)

| Engine | Time | Pieces tried | Rate | Result |
| --- | ---: | ---: | ---: | --- |
| Fast compiled DFS | 60 s | 14.1 billion | 233 million / s | Deepest placement 200 of 256; no solution |
| Fast compiled score search | 1,500 s | 358.5 billion | 239 million / s | Complete boards of 459, 460 and 461; no 462 |

The score search found its 459 board in under a second, the 460 board after 344 seconds and the 461 board after 1,191 seconds. It then looked for a 462 board for the remaining five minutes without finding one. All three boards pass the release validator, and all are below the 466 reference, so the saved best did not change.

This is one run with one seed, and the time to each board varies a great deal between seeds. In this run each further point took several times longer than the one before. At that rate a board above the 466 reference is not expected within hours or days on four threads, and the measurement gives no evidence that this search can reach it.

For scale, the Python DFS with neighbor capacity proofs tried about 8 pieces per second on the same machine and reached placement 124 in two hours. Its placements each carry far more filtering, so the two rates do not compare progress toward a solution.

To repeat the measurement on your own computer:

```sh
python scripts/benchmark_fast_search.py --workers 4 --strict-seconds 60 --score-seconds 1500 --output fast-search.json
```

## Limits

- The engines apply none of the matching, capacity or inventory proofs of the Python DFS profiles. They trade pruning for speed.
- Checkpoints belong to one seed and one set of settings. `fast-dfs` also needs the same worker count, because the threads divide one tree. Use **Restart saved fast search** (or `--fresh-exact`) to archive the saved search and begin again.
- The helper runs on the CPU. There is no GPU version of this search; the GPU settings do not affect it.
- More workers than CPU cores slows the search down.
- The app's 2 GiB memory limit is not a concern here: the helper uses a few megabytes per thread.

## Building and running the helper directly

Release archives include the helper as `native/eternity-fast`. From a source checkout, build it with any C++17 compiler that setuptools can find:

```sh
python scripts/build_fast_helper.py
python -m unittest test_fast_search -v
```

The helper can also be run by itself. It reads a piece file (`up down left right` per line), writes only inside `--out`, and never uses the network:

```sh
build/native-helper/eternity-fast --pieces data/pieces.txt --size 16 \
  --clue 34:831 --clue 45:1019 --clue 135:554 --clue 210:723 --clue 221:992 \
  --threads 4 --partition --seconds 60 --out fast-run
```

Run it with no arguments for the full option list. Its exit code is 0 for a 480/480 board, 3 when its search is exhausted and 4 when it was stopped. Boards are written as state lists (`4 × piece + clockwise rotation`, row-major cells) and can be checked with `EternityIISolver validate board.json`.
