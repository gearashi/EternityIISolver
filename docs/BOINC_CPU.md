# Existing CPU workunits: inspection and DFS compatibility

The development command `inspect-cpu` reads the observed Eternity@Home CPU workunit format and preserves its ticket identities. It does not run a search, complete a ticket, write a checkpoint, or produce a BOINC result. The released v0.1.0 executable predates this command; run it from the development source checkout until a later release includes it.

## Inspect an existing assignment

From the source checkout, with Python 3.11 or later and the application dependencies installed:

```powershell
python -B launcher.py inspect-cpu --slot 'C:\ProgramData\BOINC\slots\0' --project-root 'C:\ProgramData\BOINC\projects\boinc.eternityathome.org_eternity' --tickets 3
```

On Linux or macOS, supply the corresponding BOINC data-directory paths. A directory containing ordinary copies of the six selected inputs does not require `--project-root`. `--no-journals` skips optional ticket-journal manifests. Do not redirect output into a running BOINC slot.

The inspector reads only `job.xml`, `256pieces.txt`, `campaign_catalog.txt`, `campaign_hints.txt`, `campaign_manifest.json`, `exploration_allow.txt`, and optional matching ticket-journal manifests. It resolves BOINC logical soft links within the explicitly supplied project directory, checks target filename families, bounds reads, and checks for input changes during capture. It never reads `init_data.xml`, account configuration, or client identity files. It makes no network requests and never invokes the CPU executable or a GPU backend.

Success means that these inputs fit the supported inspection profile. Exit 2 indicates unsupported, malformed, changing, or inaccessible inputs. Future campaigns and filenames may require explicit support. The inspector does not authenticate an installed executable or the server's configuration identity.

## What is established

The inspected Windows production profile used runner 2.37 and `solver_windows_x64_236.exe`. The pinned solver's SHA-256 is:

```text
7ea0889103958d0fd1af6365ed0fd14e109acd441e5aecbfa2fbe285513ef60c
```

The inspected downloaded BOINC project files contained executables and inputs, but no solver source files or source archives. The related [public CPU solver at commit 711e0c8](https://github.com/igorpejic/eternity-ii-dfs-solver/blob/711e0c8da38a21a7ec7452a0fde0430888caa46e/bw.cpp) contains a useful earlier DFS implementation. It lacks the production interleaving, sharding, root filtering, and journal extensions, so it is not a verified substitute for solver 2.36.

### Piece and root encoding

Piece IDs are one-based and rotations are clockwise, 0 through 3, from the U,D,L,R piece definitions. Catalog indices are zero-based and remain indices into the complete catalog even when the allowlist selects only some roots. Internally the inspector uses `4*(piece_id-1)+rotation`.

Each supported catalog row supplies nine pieces. The first nine search cells, expressed as top-down row-major board indices, are:

```text
240, 241, 242, 224, 225, 226, 208, 209, 210
```

This is the bottom-left 3-by-3 region, starting at the bottom row. The map is the inverse of the production `_ZL17BOARD_ORDER_CLUED` cell-to-depth array at file offset `0x1ca740` (preferred virtual address `0x1401cbf40`). The related public source agrees. All 3,782 actual catalog rows passed border, adjacency, piece uniqueness, and clue compatibility checks under this map. The actual frame-2 hints are the official five clues rotated 180 degrees clockwise.

### Ticket enumeration

For full catalog size `N`, shard index `s`, shard count `S`, and half-open jitter interval `[J0,J1)`, the inspected interleaved stream is:

```text
globalticket = jidx*N + cidx
J0 <= jidx < J1
(globalticket - J0*N) % S == s
cidx must occur in the allowlist
```

Filtering does not renumber the catalog or its tickets. The implementation solves the modular congruence for each allowed root, then merges the resulting arithmetic progressions. It avoids scanning a large jitter range just to locate assigned tickets. This accelerates assignment enumeration, not the DFS itself.

In the copied slot-0 inputs, `N=3782`, `S=4096`, `s=274`, and `[J0,J1)=[251861248,251863296)`. The 116 allowed roots produce 116 tickets. The first is `952539264786`; the last is `952546961170`. Existing complete journal records independently matched the partition rule. See [the live-batch findings](BOINC.md#verified-live-batch-8-october-2026) for the observed discrepancies between some progress denominators and formula-derived counts.

### Seed reconstruction

Static tracing of the pinned solver's configured interleaved worker gives this 64-bit seed material:

```text
((cidx+1)*0x9e3779b97f4a7c15 + (jidx+1)*0xd1b54a32d192ed03) modulo 2^64
```

The constructor expands the material through four SplitMix64 rounds before xoshiro256++ use. Seed material, initialized RNG state, and the first random output are different quantities. `unit.reconstruct_interleaved_seed(ticket, binary_contract_sha256=INTERLEAVED_SEED_CONTRACT_SHA256)` requires an explicit pinned binary contract and a ticket belonging to the inspected workunit. It returns the constructor seed and four SplitMix64 words. Supplying that hash selects the reconstruction model; it does not authenticate a local executable.

This reconstruction is conditional on the dispatcher selecting `solverThreadConfigured<true>`. The inspector has not established which dispatcher path a running job takes, and reports that limit explicitly. It also does not establish identical candidate-table history or a complete DFS execution trace. The job's retained `--seed` must not be substituted for the reconstructed formula in that particular branch.

### Input identity

`input_sha256` hashes the selected input files' raw SHA-256 values under their logical names. It describes a captured input set; it is not the production `cfg_hash`. Both raw and LF-normalized file hashes are reported, since CRLF and LF files can contain identical puzzle rows but have different byte identities.

A base campaign manifest may name hashes that differ from the transformed per-workunit files. The inspector reports these mismatches and uses the actual captured inputs for inspection. It does not claim that a mismatch is server-authorized.

Optional journal identities retain the original opaque `cfg_hash`, epoch, and run ID after checking the supported fields against the inputs. Those consistency checks do not recompute or authenticate `cfg_hash`, and cannot fully bind fields absent from the journal manifest, including the starting jitter and allowlist. The production deduplication tuple is `(epoch,cfg_hash,shard_index,globalticket)`; never replace it with the inspector's SHA-256.

## Integration boundary

Python callers can use `inspect_workunit(slot_dir, project_root=...)`, iterate `.iter_tickets()`, or call `.summary(ticket_limit=...)`. The returned workunit and ticket structures are immutable. Job options are retained as data, never executed. Input parsing and ticket reconstruction are suitable foundations for a future compatible worker, but the current GPU search remains the separate stochastic worker described in [BOINC.md](BOINC.md).

Before claiming a drop-in GPU DFS worker, establish:

- Exact candidate-table initialization, sorting, tie ordering, and persistent state between tickets.
- Production pruning, sigma-tail/main settings, endgame ladder, and search-order behavior.
- Node counting and cap boundaries, interruption, per-ticket CPU budgets, and checkpoint/resume semantics.
- Output archive contents, ticket completion statuses, server validation, and assimilation rules.

Production source would make this substantially easier. Without it, the alternative is further reconstruction plus reproducible reference traces for the same inputs, seeds, node budgets, and resumed states. A legal board or a matching ticket number alone cannot demonstrate equivalent search coverage. This inspector produces no coverage or completion claim, and an unsuccessful bounded search cannot establish that Eternity II is impossible.

## Offline checks

```text
python -B -m unittest test_boinc_cpu_workunit -v
```

The tests use temporary synthetic workunits, including public puzzle pieces and a few known legal prefix rows. They compare modular ticket enumeration with brute force, check malformed inputs and changing snapshots, verify soft-link confinement, test 64-bit seed arithmetic, and exercise the read-only launcher route. They neither read a user's live BOINC slots nor execute a production solver. The native-build workflow includes these tests and a frozen `inspect-cpu --help` smoke check.
