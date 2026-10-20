Version **0.4.0** makes structured matching repair resumable and reduces repeated CPU validation setup. The release also includes the previously local preview's diverse cached parents, plateau diversification and connected-region repair. The bundled best remains **466/480**; no faster route to 480 is established.

### Changes

- **Faster validation:** reuse an immutable puzzle snapshot while checking each candidate's physical pieces, frame, domains, fixed clues and score. Bounded CUDA comparisons record completed neighborhoods, elapsed time and independently checked final boards.
- **Resume structured repair:** both matching engines save the working population, random state, recent-board history, counters and bounded neighborhood history. Start continues a compatible checkpoint. An interrupted batch may repeat from its previous complete boundary, with every verified improvement retained. Changed settings require **Restart repair population**, which archives the previous checkpoint.
- **Broader repair search:** select varied parents from the local cache, perturb stalled working boards while protecting the saved best, and attempt bounded connected-region CP-SAT repairs.
- **Tested experiments:** seeded equal-score matching choices, cycle-aware parent pairing and adaptive regional budgets are implemented as Python API experiments, disabled by default. They passed correctness checks but did not establish improved scores in the bounded comparisons; seeded ties added overhead.

### Validation and scope

The five candidates were compared with finite correctness tests, seeded CUDA trials on real cached boards, independent board validation and CPU/OpenCL cost checks. Deterministic split/resume tests cover interrupted batches, partial improvements, corrupt checkpoints and settings changes. Source hashes, exact measured source snapshots and raw reports are in **docs/SEARCH_RESEARCH.md**. Short desktop timings measure their stated operations, not expected time to solve the puzzle.

Native CI tests and packages Windows x64, Linux x64, macOS Intel and macOS Apple Silicon. Windows/Linux include CUDA and OpenCL; macOS uses OpenCL. Linux source and frozen diagnostics use PoCL to test the OpenCL code path. Physical AMD, Intel and Apple GPUs have not been locally certified. Compatible GPU drivers are required; Mac apps are not notarized.

Search remains offline. Library refresh is an explicitly requested single-archive download and local merge. No boards, counters, BOINC work or telemetry are uploaded. Export saves local files for manual review. Search starts only when requested, and existing state/cache directories are retained. Native SAT/CP-SAT search still restarts from the saved best; its internal state is not checkpointed.
