# Autonomous search improvement program

Target: implement the five requested improvements, then repeat discovery, correctness audit,
bounded comparison and integration while justified actionable candidates remain. This is an
algorithm research program, not a claim that every possible optimization can be exhausted.
No score of 480 has been found. Full-puzzle feasibility remains unresolved.

Base revision: f8fe853deddb49d7b9e2e2535be5b67907bdeec9. Checkpoint: 2026-10-13.
User permits bounded CPU/GPU tests. Production search remains stopped. All runs are offline,
with isolated state; the canonical STOP, best board, caches and checkpoints are preserved.
Only explicit manual library archive downloads may access the project. No uploads or counters.

## Routes

| Route | State | Owner and next evidence |
| --- | --- | --- |
| Verified-core minimum hitting set | Implemented; independent audit passed | Bounded greedy/hitting-set comparison below |
| Adaptive optional DFS pruning | Implemented; checkpoint replay audit passed | Baseline/strong/full/adaptive ablation below |
| Dynamic disjoint DFS splitting | Implemented; finite partition/ownership tests pass | Independent exhaustive frontier audit passed; full scaling unmeasured |
| Incremental SAT score bounds | Implemented; tiny exhaustive threshold tests pass | Full-puzzle setup/query comparison below |
| GPU batched matching costs | Implemented; scalar CPU oracle tests pass | CUDA/OpenCL original-board comparisons below |
| Component binary recombination | Implemented; exact small oracles pass | Same parent-pair comparison below |
| Actual piece rotation symmetry | Closed for this inventory:256 distinct rotation classes, no repeated rotations | [Finite audit](evidence/inventory-symmetry-20261013.json); no reduction applicable |
| Large region support caches | Candidate; not yet implemented | Requires inventory/domain-aware keys and benefit evidence |
| Conflict/impact ordering | Candidate; not yet implemented | Profile existing search before adding new ordering/state |
| Single-piece support bit test | Implemented and independently checked | Paired test supports reduced overhead |
| Bulk native matching arcs | Implemented; finite equivalence tests pass | Bounded same-matrix scalar/bulk comparison |
| Queue validation reuse | Implemented and independently checked | Setup-only matched comparison below |

## Evidence and decision rules

Correctness and speed are separate. Exhaustive tiny cases, independent validation, and
checkpoint/resume/cancellation tests justify tested soundness. Compare equal seeded problems,
wall budgets and full processing costs before recommending an optimization. Desktop timing
variation must be disclosed. Timeouts are inconclusive; fewer branches are not faster solving.
Record input/source hashes, options, device, limits and raw outcomes with every experiment.
Deferred/inconclusive routes retain their next needed evidence; they are not called exhausted.
A discovered defect returns its route to repair and invalidates affected performance claims.

Existing first-round measurements: [SMART_SEARCH.md](SMART_SEARCH.md).
Round-two measurements and decisions will be appended here after review and testing.

## Round-two hardware and repair evidence

[Actual-board GPU/cycle report](evidence/round2-gpu-cycle-20261013.json): original
five-clue EII boards, read-only cached parents, seed17, same16 independent neighborhoods,
two order-reversed repeats, exact scalar matrix oracle, independent full-board validation.
NVIDIA RTX2060, CUDA and OpenCL. CPU full batches took2.63/3.41 seconds; CUDA1.06/1.92.
In the OpenCL comparison CPU3.55/2.56 seconds; OpenCL1.11/1.05. These short desktop
measurements support this batched workload only; GPU initialization and desktop variation
matter. NVIDIA OpenCL execution does not certify AMD or Apple hardware.

Sixteen two-parent comparisons per method were all optimal for their selected cycles.
The general CP-SAT fallback took3.86 seconds total, exact component fast path1.87.
All board scores matched; best stayed466. No full puzzle progress or480 solution is claimed.
Sources stayed unchanged within this report. Canonical runtime and cache were not written.

The exact measured GPU/cycle sources are preserved in `evidence/round2-gpu-cycle-20261013.sources.zip`.
Its excluded CP-SAT import warmup used default worker settings; each measured cycle solve
used one worker. The follow-up harness explicitly caps warmup at one and rejects an empty parent set.

## Round-two pruning and SAT comparisons

[Pruning report](evidence/round2-pruning-20261013.json): baseline solved the known-solvable
synthetic8x8 in0.044s/23branches; strong0.142s/29; full0.582s/2; adaptive0.228s/2.
This supports adaptive scheduling over unconditional heavy filters for this case, but
baseline was cheaper. All full-EII3s runs timed out. Regional UNSAT describes only the
frozen12x12 repair task. The baseline remains the default; no universal winner is claimed.

[SAT/core report](evidence/round2-sat-core-20261013.json):18s budgets, same466reference,
seed17, one worker. Greedy release208rounds; hitting sets266; both score466.
Incremental SAT and no-phase-hint SAT reached native branching but retained466.
CP-SAT remained in native processing with no branches reported and retained466.
Setup plus cleanup can overrun cooperative search limits slightly; external process
limits provide a separate bound. Different decision counts do not rank these methods.

[Support-mask ablation](evidence/dfs-supported-bitmask-20261013.json): two same-seed
synthetic8x8 comparisons preserve23branches and checked solutions; legacy0.079/0.071s,
bit-length0.055/0.059s. Full-EII3s cases all timed out; the exact equivalent calculation
explored more branches within each budget. This measures overhead reduction only.

## Additional measured decisions

[Bulk assignment arcs](evidence/assignment-arcs-20261013.json):30 alternating samples
on3 synthetic16x16 matrices returned identical exact boards. Scalar arc loading plus
matching/validation total0.9755s, bulk0.6712s. Native bulk submission is bounded in4096arc
chunks and falls back to scalar only when the API is absent; execution errors propagate.

[Precomputed DFS masks](evidence/dfs-precomputed-complements-20261013.json): a follow-up
cache improved both tiny trials but showed mixed full-EII throughput under desktop
variance. Default remains the simpler inline bit-length calculation. This candidate is
inconclusive, not a proven useless optimization; resume with controlled longer profiles.
Measured source snapshots accompany reports. No sources are inferred from newer code.

[Queue setup reuse](evidence/queue-cache-setup-20261013.json): same original-puzzle4ticket
fixture, queue validation followed by3 dispatch/accept task+hash passes, three order-alternated
repeats. Uncached4.656/4.375/4.922s; cached0.469/0.422/0.453s. Fingerprints agree; peak38.4MiB.
This speeds queue preparation, not a measured path to480. Inputs are detached, cache keys
include complete decision paths, memory is bounded, and semantic checkpoint replay is never cached.

Current delivery gate: all requested methods and four additional accepted optimizations are
implemented. Whole-suite, packaged execution and stopped-app installation checks follow.
The user additionally authorized a new GitHub release after this improvement cycle and
platform builds pass. Release stopping rule is no remaining justified candidate ready for
implementation, never a claim that every mathematically possible improvement is exhausted.
Larger support regions, explained learning and ordering ideas need separate cost/correctness
evidence; they remain recorded rather than declared impossible or exhausted.

## Delivery validation checkpoint

2026-10-13: all383 source regression tests passed in47.418s after the nine accepted
improvements. New release request: version0.3.0; native packaging, real device frozen
diagnostics and remote platform builds are required before publication. A final profile
of repeated GPU batch preparation is being audited before freezing source for the build.

## Release-cycle stopping assessment

Remaining routes are not silently counted as implemented. Larger region caches require a
measured repeat rate and keys covering inventory, domains, boundary and clues; the current
full-filter baseline already spends substantially more per node than cheap propagation.
A larger operator is not ready to become a default without such evidence. Explained custom
learning would require complete reasons from every custom propagator; native SAT already
provides sound learning and is available. Impact/learned ordering needs representative deep
search trials to distinguish ordering effects from desktop noise. These are deferred research
experiments, not claims that no better algorithm exists. The precomputed-mask follow-up is
explicitly inconclusive and stays out of the default. No unmeasured throughput target is promised.

The final measured candidate is reusing a detached structural validation view for batches.
Every board must still be independently checked; only repeated reconstruction of the unchanged
full-domain model may be removed. Its proof, mutation tests and full-batch timings gate acceptance.
Once this candidate is accepted or rejected and regression/package/platform checks pass,
version0.3.0 will close this implementation/release cycle while preserving these open routes.

## Final batch-preparation candidate accepted

[Validation-view comparison](evidence/assignment-validation-20261013.json): original466
board, same16 CUDA neighborhoods, three order-reversed full-batch repeats. Rebuilding the
unchanged domain model took1.086/1.322/1.521s; reusing a detached immutable structural view
with exact typed compatibility took0.065/0.057/0.190s. All exact result boards agreed with
CPU reference and passed independent board validation. Every board still undergoes piece,
domain, frame, fixed-clue and score checks. Tests cover mutation around callbacks, immutable
clues/domains, list/tuple equivalence and rejection of typed aliases. Independent review and
10 separate negative probes passed. No checkpoint or GPU arithmetic changes are involved.

This closes the current set of implementation-ready candidates: the five requested methods
plus component recombination, bulk assignment arcs, support-mask optimization, queue setup
reuse and immutable batch validation are accepted. Inconclusive/deferred routes remain above.
Source now freezes for the final regression, local frozen build and GitHub platform builds.

## Final source regression

2026-10-13: all 387 source regression tests passed in 49.524 seconds after all ten
accepted improvements. The separate BOINC input-validator suite passed all 7 tests.
Version consistency passed for v0.3.0. These checks do not run production BOINC workunits.
Packaged execution and platform build checks remain publication gates.

Final diagnostic addition: CPU OpenCL is permitted only by an explicit bounded
diagnostic command, never by dashboard/search execution. Source and frozen Linux
PoCL checks now exercise the new assignment kernel against the scalar CPU oracle.
After this addition, all 390 source tests passed in 152.833 seconds and all 7 BOINC
validator tests passed in 0.075 seconds. Version consistency again passed for v0.3.0.
