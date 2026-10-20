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
| Structured plateau escapes | Implemented; legal-move/tabu/incumbent tests pass | Working parents can worsen; saved best is separate |
| Diverse library parents | Implemented; validated read-only seeded selection | Modest diversity change; selection overhead measured below |
| Adjacent-region score repair | Implemented; exhaustive tiny-region oracles pass | Compact one-worker CP-SAT targets best+1 with frozen complement |
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

## Cross-platform release correction

The v0.3.0 release gate correctly blocked publication when one integration-test
fixture compared a resolved queue path with an unresolved alias. macOS maps /var
to /private/var, and the Windows runner used its short username path. Both names
identified the same file; all other 389 tests passed on each affected runner. The
fixture now compares canonical paths, retaining its queue-identity and preservation
assertions. Solver code is unchanged. Version 0.3.1 carries this correction rather
than moving the already-pushed v0.3.0 tag. All platform gates must pass again.

## Three additional structured repair methods (local preview 0.3.2.dev1)

The previously identified diversification, cached-parent diversity and adjacent
score-target region operators are implemented and enabled together in the
matching/GPU-matching portfolio. Unit/integration coverage includes legal
working-board degradation without incumbent degradation, recent-board tabu,
read-only seeded cache selection, donor-pool uniqueness, frame/clue/domain
preservation, regional objective equality with exhaustive tiny-case oracles,
input-size bounds and native cancellation cleanup. The exact diagnostic command
also executes all three operators in a packaged process.

The initial full suite passed 424 tests; affected follow-up suites passed after
bounded iterator and preparation-budget checks. Seven BOINC result-validator
regressions passed; these tests do not execute any external project app.

The first six 15-second CUDA runs were dominated by initialization and host work.
They retained 466 and completed only 16–62 neighborhoods (1–4 batches), which
never reached the default regional/diversification thresholds. Their startup
behavior is recorded in [the cold report](evidence/repair-portfolio-startup-20261013.json)
and [its exact sources](evidence/repair-portfolio-startup-20261013.sources.zip).
They validate returned boards and resource/state handling, not the efficacy of
operators that did not activate. The follow-up comparison records warm-up
separately and keeps default thresholds and the 32-parent selection limit.

The warmed comparison used two seeds in reversed mode order, each with 45
seconds of search after separately reported GPU/native warm-up. Both combinations
ran the new diversification and regional operators. All six final boards validated
at 466. All reported application network-attempt lists were empty; the protected
runtime files and local cache hashes were unchanged.

| Mode | Seed | Cached parents | Matching neighborhoods | Diversifications | Region searches | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 17 | 32 | 334 | 0 | 0 | 466 |
| diverse-only | 17 | 32 | 222 | 0 | 0 | 466 |
| combined | 17 | 0 | 448 | 3 | 3 | 466 |
| combined | 41 | 32 | 334 | 23 | 2 | 466 |
| diverse-only | 41 | 32 | 382 | 0 | 0 | 466 |
| baseline | 41 | 32 | 414 | 0 | 0 | 466 |

[Warm report](evidence/repair-portfolio-warm-20261013.json) and
[exact measured sources](evidence/repair-portfolio-warm-20261013.sources.zip).
The combined seed-17 run exhausted the five-second parent-preparation budget
before decoding any cached boards, so it ran from the reference alone. Its higher
neighborhood count cannot be attributed to improved search quality or directly
compared with 32-parent cases. Selection's metadata-first scheduling was therefore
returned to implementation for a bounded elite-first fix. The other cases also
show substantial desktop timing variation. These runs establish activation and
validated outputs in their tested bounds, not a faster route to 467 or 480.

The preparation fix now validates up to eight elites before optional reservoir
sampling and reserves half of the remaining budget for validation/selection.
A scan-only timeout does not poison subsequent point reads; a global timeout
returns previously validated candidates. Twelve selector tests passed, including
fake-clock scan/global-deadline checks. The read-only actual-cache follow-up
returned 32 independently validated parents, including a 466 elite, for seeds 17/41
in 3.125s/3.562s. Protected runtime/database/WAL hashes stayed unchanged.
[Parent-budget evidence](evidence/repair-parent-budget-20261013.json).


The final CUDA confirmation used the repaired selector and loaded 32 parents in
4.812 seconds. It completed 302 matching neighborhoods, 22 diversifications and
two regional searches; the independently verified best stayed at 466. Search took
46.485 seconds against a cooperative 45-second budget; the process stayed within
its external 75-second limit. Source hashes, all 12 protected runtime files and
network-attempt checks passed. The report's aggregate `passed: false` is retained:
the library database/WAL changed while the user independently imported the latest
archive through the dashboard. The user confirmed this import, which added 15,957
boards (137,434 known arrangements afterward). No benchmark socket attempts were
recorded. The new boards were preserved; a subsequent read-only check selected
32 valid parents and confirmed that source, runtime and post-import cache hashes
stayed unchanged. Its 8.594-second total includes independent validation after
parent selection; it is not a parent-selection-only timing.

[Final GPU report](evidence/repair-portfolio-final-20261013.json),
[exact sources](evidence/repair-portfolio-final-20261013.sources.zip),
[post-import cache check](evidence/repair-cache-post-import-20261013.json) and
[computation manifest](evidence/repair-portfolio-manifest.json) preserve these
results and their limits. Desktop contention and short runs prevent a reliable
speed ranking. The tests establish bounded correctness and operator activation,
not faster time to 480 or global infeasibility. That experiment was a local Windows preview; its changes are included in 0.4.0,
and the 0.3.1 release remains available for rollback.


## Five repair candidates: 20 October 2026 (0.4.0)

The release keeps two changes enabled: immutable structural-validation reuse and
structured repair checkpoint/resume. Equal-score random tie handling, cycle-aware
parent selection and adaptive regional budgets remain disabled Python API
experiments (`neutral_ties`, `pair_selection`, `adaptive_regions` in
`smart_repair.solve`). Correctness alone does not establish search effectiveness.

The direct validation test used 16 valid and 16 deliberately invalid actual boards,
three alternating passes per mode. Median times were 0.703 seconds rebuilding the
structural problem and 0.016 seconds reusing it. This approximately 44-fold result
applies only to the validation microbenchmark, not the whole solver. All tested
legality outcomes agreed.

Seeded ties preserved the exact primary matching objective and CPU/device cost
agreement but produced no additional boards in the two elite neighborhoods tested.
Cycle-aware selection increased mean active components from 11 to 13 in six sampled
pair comparisons, but its returned boards scored 465 while elite-parent comparisons
retained 466. More components do not establish better search. Fixed and adaptive
regional schedules both retained 466 during their 15-second operator comparisons.
The adaptive schedule exercised all four profiles but showed no measured score gain.

Full-portfolio ablations use one fixed, independently validated 32-parent fixture,
six configurations, 30-second search budgets and two seeds with reversed order.
GPU/native warm-up is reported separately. Every accepted improvement and final
board is independently validated. These are short desktop trials: one tiny CPU
diagnostic overlapped part of the first-seed trials for about 0.8 seconds. The raw
reports preserve the actual timing, source hashes and interruption status.

Six finite continuation tests cover an uninterrupted eight-batch run versus a
three-plus-five resumed run, interrupted-batch replay, best preservation, callback
isolation, settings changes and the 64-parent limit. Matching checkpoints preserve
the complete working population and random state at a completed batch boundary.
They do not claim exhaustive puzzle coverage or serialize native SAT/CP-SAT state.

Evidence: [direct comparisons](evidence/repair-candidates-20261020.json),
[portfolio ablations](evidence/repair-candidate-portfolios-20261020.json),
[OpenCL agreement](evidence/repair-candidates-opencl-20261020.json),
[exact measured sources](evidence/repair-candidates-20261020.sources.zip),
[public parent fixture](evidence/repair-candidates-20261020.parents.json),
[evidence audit](evidence/repair-candidates-audit-20261020.json) and
[v2 computation manifest](evidence/repair-candidates-manifest.json).
The manifest audits the preserved experiment evidence; the raw reports retain the
original experiment commands and before/after measurements.


With validation reuse alone, the two 30-second portfolio trials completed 2,414
and 2,430 neighborhoods versus 670 and 750 for their baselines (3.60x and 3.24x
counts). Their scores remained 466. This isolates that optimization; it does not
include periodic checkpoint writes or measure the released app's time to solve.
The [release regression report](evidence/release-regressions-20261020.json)
records 470 application tests, seven result-validator tests and 16 tiny solver
diagnostics, all passing. An initial continuation-test failure was corrected by
explicitly enabling the nondefault feature in its fixture; production behavior
was unchanged. A separate isolated real CUDA-worker smoke run saved 47 completed
batches, resumed to 104 batches, and retained the independently verified 466 best.
