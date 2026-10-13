Version **0.3.1** adds optional search and repair methods while preserving the original pieces, gray frame and all five fixed clue orientations. No 480/480 solution or faster route to a full solution is claimed. The best verified board in the bundled data remains 466.

### Search methods

- **GPU costs + CPU matching:** CUDA/OpenCL computes batches of legal placement-cost matrices; one CPU worker solves exact assignments and validates complete boards. Its separate neighborhood batch control ranges from 1 to 64, default 16. Existing parallel GPU repair and GPU-hints-plus-DFS modes remain available.
- **Incremental SAT score optimization:** seeks a score above the saved best and tightens targets while reusing one SAT model and learned clauses. Reference-board hints affect ordering only.
- **SAT core repair:** uses verified conflict cores and bounded minimum hitting sets to choose cells to release. Fallback covers are checked and never described as minimum unless proved.
- **CP-SAT score optimization and matching/cycle repair:** save independently validated improvements immediately. Nonadjacent matching and whole-cycle recombination preserve physical piece uniqueness.
- **Partitioned DFS:** one to four workers share disjoint tasks and divide a yielded remaining frontier when workers become idle. Queue ownership and checkpoint replay are validated.
- **Strong, full and adaptive DFS filtering:** optional all-different matching filters, independent completions, alternative branching, probes, small-square support and color feasibility. Adaptive schedules extra checks from deterministic pruning utility. Baseline remains available and remains cheaper on some measured cases.

The release also reduces overhead in cycle optimization, matching arc loading, DFS support masks, queue setup and repeated GPU batch preparation. Reusing immutable puzzle constraints preserves per-board validation while avoiding repeated reconstruction. Short local comparisons used identical inputs, independent oracles and bounded runs. Actual-board CUDA/OpenCL costs agreed with scalar CPU costs on the tested NVIDIA GPU; some batched repair operations were faster. These measurements do not establish faster time to 480, full-puzzle optimality, or certification of AMD/Apple hardware. See **docs/SMART_SEARCH.md** and **docs/SEARCH_RESEARCH.md** for raw reports, exact measured source snapshots, accepted improvements and inconclusive experiments.

### Saved state and offline operation

DFS checkpoints now use **schema 4**, and parallel queues use **schema 2**. Older files are preserved and require their matching older application to resume. To start a new frontier, explicitly select **Restart DFS frontier**; this archives the old checkpoint. Changing pruning settings also requires an explicit restart. Native SAT/CP-SAT and repair methods restart from the saved best after stopping. Existing GPU checkpoints remain separate.

Opening the app starts the local dashboard; search requires **Start**. **Stop & save** preserves compatible DFS/GPU progress. Library updates remain manual single-archive downloads from the fixed public archive endpoint. No startup downloads, background per-board requests or automatic retries are added. Search workers remain offline: no boards, work counters, BOINC tasks or telemetry are uploaded. **Export best board** saves local files for manual review only.

### Packages and validation

Packages target Windows x64, Linux x64, macOS Intel and macOS Apple Silicon, plus a Python wheel/source archive. Windows/Linux include CUDA and OpenCL; macOS uses OpenCL. Compatible GPU drivers are required; macOS packages are not notarized. Physical AMD, Intel and Apple GPUs were not available for local certification. CI tests portable CPU logic and frozen commands on each target, with separate Linux source/frozen PoCL checks.

Public puzzle data, source, evidence and dependency notices are packaged. User checkpoints, downloaded board caches, local discoveries, logs and credentials are excluded. BOINC inspection examples remain research tools, not approved production project applications. All accepted boards receive independent piece, border, clue and score validation. Bounded timeouts and neighborhood optima are never evidence that the whole puzzle is impossible.
