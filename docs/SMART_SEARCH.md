# Smart-search experiments

These additions preserve the original piece identities, gray frame and all five
oriented clues. They run offline. A manually imported archive supplies candidate
parents; it is never used to rule out a search branch or certify novelty.

The implemented methods are alternatives with different costs. There is no
established fastest solver for this instance, and no claim that this list exhausts
all possible algorithms. Correct pruning can be slower than a cheaper search.

## Implemented methods and scope

| Method | What it adds | Scope and limitations |
| --- | --- | --- |
| All-different SCC filtering | Removes piece/cell assignments absent from every perfect matching | Balanced cell/piece graph; rotations stay grouped by physical identity |
| Independent endgame | Finishes mutually nonadjacent undecided cells through a perfect matching | Requires adjacency propagation first; validates the completion |
| Piece-location / edge-color branching | Branches on a scarce piece location or edge color when it gives fewer alternatives | Every branch partition is exhaustive; ordering is a heuristic |
| Bounded singleton probes | Rejects a placement whose propagated consequences contradict the task | Interrupted probes never justify a deletion |
| Identical-domain Hall reservations | Reserves a piece subset for exactly enough competing cells | Cheaper subset rule; redundant with complete SCC filtering |
| Checkerboard inventory | Checks color multiplicities using bounded-cardinality exact integer arithmetic | Full framed grid and rotation-invariant inventory only |
| Directional color inventory | Checks total east/west and north/south color balance | Full framed grid only; this relaxation does not encode every parity/direction relation |
| 2x2 support joins | Deletes states absent from every distinct-piece completion of a square | Enumerates completely or skips; deterministic work caps |
| Color-component / flow feasibility | Detects shortages of reachable matching color stubs | A necessary relaxation, not sufficient for a solution; expensive cases skipped |
| SAT endpoint support | Links each edge color back to supporting placements on both endpoints | Improves propagation without changing solutions; measured separately |
| Edge-disjoint mismatch bounds | Proves minimum mismatch costs in bounded square relaxations and individual edges | Never sums overlapping edge costs; capped squares fall back to edge checks |
| Persistent partitioned DFS | Assigns disjoint cell-state subproblems to up to four CPU processes | Saves each frontier and completed ticket; restored exhaustion is not a global proof certificate |
| CP-SAT score optimization | Maximizes all matched edges while retaining hard structural rules | Can save 467–479 improvements; score-target infeasibility is scoped explicitly |
| Assumption-core repair | Expands frozen-board repair using actual SAT conflict cores | All edge equalities remain hard; timeout is never an UNSAT core |
| Nonadjacent-set weighted matching | Optimizes a whole legal piece permutation and its rotations at once | Exact for that frozen-complement neighborhood; can plateau |
| Conflict-core hitting sets | Selects a small set of released cells covering every verified SAT conflict core | Minimum applies only to collected cores; bounded fallback remains a checked cover |
| Adaptive optional pruning | Schedules extra filters using deterministic work and useful deletions | Mandatory consistency remains; scheduling state is checkpointed |
| Dynamic DFS splitting | Transfers a yielded remaining frontier into disjoint child tasks when workers idle | Cooperative slice boundaries only; caps preserve unsplit work |
| Incremental SAT score bounds | Reuses one score encoding and learned clauses while seeking best+1 | Temporary mismatch assumptions; each score witness independently validated |
| GPU batched placement costs | Computes integer cost matrices on CUDA/OpenCL; CPU solves exact assignments | Bounded batches, explicit device failure, independent full-board validation |
| Component binary optimization | Tree dynamic programming and small complete enumeration of cycle choices | Exact component objective; general CP-SAT fallback for unsupported or interrupted cases |
| Whole-cycle board recombination | Chooses between two cached boards by complete piece-permutation cycles | Preserves uniqueness by construction; exact objective for that parent pair |

The matching portfolio uses both checkerboard halves and mixed-parity independent
sets. Only alternating the two halves would permanently trap each physical piece
on its starting parity. Cycle recombination can change adjacent cells jointly.
The portfolio retains the best validated board on interruption and reports
completed neighborhoods separately from DFS branches.

## Controls and saved state

The dashboard exposes experimental score optimization, assumption-core repair,
matching/cycle repair, and partitioned DFS under the CPU search selector. Existing GPU repair and
GPU sampling followed by CPU DFS remain available. GPU costs + CPU matching adds a separate neighborhood batch control (1..64, default16).
The original parallel GPU search count is a different setting. GPU costs require an
actual device; device failures never silently switch this mode to CPU. Cycle choices,
assignments and final validation execute on the CPU.

DFS exposes `--pruning baseline`, `--pruning strong` , `--pruning adaptive` and `--pruning full`; the baseline remains
available for comparisons. The strong profile enables SCC filtering, independent
completion and alternative branch partitions. Maximum filtering (`full`) also enables
probes and the color, square and flow checks. Adaptive schedules those optional checks
using observed deterministic work and pruning utility, with periodic retry. Mandatory
arc consistency and perfect-matching feasibility always remain enabled. Either profile may be slower. Advanced Python callers may pass
`options` to `exact_dfs.solve` to select probes or `extra_rules` individually. See
`scripts/benchmark_smart_search.py` for isolated ablations. Expensive rules are
experimental rather than silently added to every node.

DFS checkpoint schema is now version 4 and binds the complete option set and
branch partitions. Version 2/3 checkpoints remain untouched but require the old
app to resume. Explicitly choosing a fresh search archives the old checkpoint;
changing profiles cannot silently reuse a different frontier. Parallel queue schema is
version2; version1 queues are preserved for the matching older app. Adaptive pre-node
state and split ownership are validated during replay. Native SAT/CP-SAT
and native score optimizers restart search using the saved best board. Structured matching now resumes its population, random state, recent-board history and bounded completed-neighborhood history from a compatible checkpoint. A partial batch replays from the previous complete boundary while retaining any independently validated improvement. Changing continuation settings requires an explicit restart, which archives the old file.

## Correctness and evidence

`test_dfs_improvements.py`, `test_exact_rules.py`, `test_exact_sat.py`,
`test_exact_optimize.py` and `test_smart_repair.py` compare small cases with
independent enumeration, exercise interruptions and validate every accepted
board. Full-puzzle candidates additionally pass `validator.py` before saving.
Tests and benchmarks use bounded runs and isolated state; they never run a
production BOINC task, submit a board or report work counts externally.

Benchmark reports record seeds, source hashes, task scope, time limits, outcomes,
wall time and independently checked results. A timeout proves neither success
nor impossibility. Regional infeasibility only describes the frozen region.
Fewer explored branches alone does not establish a speed improvement.

## Structured repair additions (0.4.0)

The **GPU costs + CPU matching** and **Matching + cycle repair** engines now
include all three additions below automatically. GPU costs still use CUDA or
OpenCL; native assignment, cycle decisions and regional CP-SAT use one CPU
worker. The ordinary GPU swap engine is unchanged.

- **Plateau escapes:** after eight stagnant visits to a working parent, perform
  a bounded set of legal swaps/rotations, then repair again. Working parents
  may lose matches. A recent-board tabu list discourages immediate reversal;
  a strict new local record overrides tabu. The independently validated
  best-ever board is copied separately and can only improve. Pending GPU
  batches are fully processed before a parent is perturbed.
- **Diverse library parents:** first validate up to eight elite candidates, then
  sample score bands with a seeded metadata-only pass. Reserve half the
  remaining preparation budget for fetching and selection. At most 256
  arrangements total are loaded by indexed row ID. Every accepted
  board is independently validated and rescored. Up to 32 parents retain a
  high-score elite quarter and otherwise spread piece placements, rotations
  and mismatched edges. Selection is read-only and has a five-second default
  budget; cancellation can return a smaller validated pool. No download is
  initiated. The reference board is retained even if cache selection fails.
- **Connected-region score repair:** choose a connected core near mismatched
  edges plus compatible remote donor cells. Freeze all other exact placements
  and seek the current best score + 1. The compact CP-SAT model enforces unique
  physical pieces, legal domains/frame and all fixed clue orientations, and
  scores each changing edge and boundary once plus the fixed score. Default
  regions vary across 8/12/16/20 core cells and 2/4/6 donors. Calls run once per
  eight portfolio batches, with a cooperative 0.25-second budget including
  selection and setup, and roughly 20% of elapsed time across the run. Native
  calls and cancellation checks can slightly overrun the requested budget. A failed neighborhood
  establishes nothing about regions with different frozen complements.

The dashboard shows working parents, plateau diversifications and region
searches/improvements. All saved improvements retain independent validation
and manual export. The search remains offline. API switches support bounded
baseline comparisons; they do not disable any mandatory physical constraints.

Finite tests compare regional objectives with exhaustive tiny-puzzle oracles,
exercise seed reproducibility/cancellation and verify that lower-score working
boards cannot overwrite the incumbent. Performance evidence is recorded in
[SEARCH_RESEARCH.md](SEARCH_RESEARCH.md); these heuristics do not imply a
verified route to 467 or 480.

## Bounded measurements: 13 October 2026

On this Windows PC (Intel Core i7-4790, Python 3.12.14), the short tests
established correctness in their tested range and exposed overhead. They did
not find a board above 466 or determine the fastest full-puzzle algorithm.

- The 8x8 synthetic DFS case took 0.061 s / 23 branches with the baseline,
  versus 0.335 s / 4 branches with square filtering. Fewer branches was slower.
- Every 3-second full-puzzle DFS ablation timed out. Repeated identical baseline
  trials explored 263, 290 and 192 branches, illustrating desktop-run variability.
- The repair comparison completed 30 independent-set neighborhoods and eight
  two-parent cycle neighborhoods optimally. Their best board remained 466.
- Reusing one incremental SAT model removed repeated construction in core
  repair. In an 18-second budget, baseline construction consumed 17.140 s and
  handled eight assumption sets (seven UNSAT cores). Reuse consumed 2.094 s
  for construction and handled 208 sets (207 cores). Both retained 466. This
  is evidence of less setup work, not a measured speedup to a solution.
- CP-SAT score optimization spent its 18-second run in native preprocessing and
  reported zero branches. The dashboard labels that phase explicitly.

Reports: [pruning](evidence/smart-pruning-20261013.json),
[repeats](evidence/smart-pruning-repeats-20261013.json),
[repair baseline](evidence/smart-repair-baseline-20261013.json), and
[incremental core repair](evidence/smart-core-incremental-20261013.json).
Each records exact measured source hashes. The baseline optimizer predates the
small-domain mismatch-bound integration; it is not a measurement of that new
precheck. Broad full-puzzle domains skip that precheck. Small finite tests show
it safely rejects a target that independent edge bounds alone miss.

The first-round parallel measurements below predate dynamic subdivision. Current
parallel search also subdivides a yielded frontier when there are idle workers. Its cancellation,
checkpoint and piece/clue correctness have tiny tests and a bounded real-board
CLI smoke check. These are not scaling benchmarks.

## Ideas retained for later discriminating experiments

- Explained conflict learning and backjumping already exist in the native SAT
  engine. Building another learning engine in Python would require reliable
  explanations from every custom propagator before enabling learned deletions.
- Larger inventory-aware region caches must include available physical pieces,
  boundaries, domains and clue states. Boundary-only cache keys are unsound.
- Color-2 path structure is unusually restricted in the real inventory, but
  decomposing around it may produce more cases than it eliminates. No pruning
  rule based solely on the current mismatched board has been enabled.
- The complete inventory audit found256 distinct rotational equivalence classes and
  no piece with repeated rotational states. There is no such symmetry to remove in
  this actual inventory. Equal color counts do not make pieces interchangeable.
- Impact heuristics and learned ordering require
  comparison against stable seeded baselines. Their advice may reorder search,
  but cannot justify deleting a branch without a sound constraint.

## Primary sources

Independent-set matching repair is established work, not a novelty claim:
[Schaus and Deville, JFPC 2008](https://webperso.info.ucl.ac.be/~yde/Papers/JFPC2008_Eternity_en.pdf).
All-different filtering follows the matching principle in
[Régin, AAAI 1994](https://cdn.aaai.org/AAAI/1994/AAAI94-055.pdf).
The SAT route is informed by
[Heule's Eternity II encoding](https://www.cs.cmu.edu/~mheule/publications/eternity.pdf).
These sources motivate algorithms; they do not demonstrate a speedup or a 480
solution for this implementation.

## Continuing improvement program

The route status and further experiments are tracked in [SEARCH_RESEARCH.md](SEARCH_RESEARCH.md).
The program distinguishes implemented algorithms, finite correctness evidence, measured
performance and untested candidates. An open-ended improvement loop cannot certify
that no possible optimization remains. It preserves all board/STOP files and never
turns a bounded negative result into a full-puzzle impossibility claim.


## Candidate decisions in 0.4.0

Repeated board acceptance reuses a detached immutable puzzle snapshot. Every board
still receives full structural and score checks. Structured matching also resumes
its working population and random state after Stop instead of rebuilding from the
best board alone. An incomplete batch is replayed from the last complete boundary;
any independently verified best is retained. Checkpoint settings and contents are
validated before use, and an explicit restart archives incompatible state.

Three other tested Python API options remain disabled: `neutral_ties` explores
seeded alternatives with the same primary matching score, `pair_selection` ranks
sampled parent pairs by cycle structure, and `adaptive_regions` learns from verified
regional score gains per elapsed second. They preserve all puzzle constraints, but
bounded real-board comparisons did not establish a score improvement. See the
[experiment decisions and raw evidence](SEARCH_RESEARCH.md#five-repair-candidates-20-october-2026-040).
