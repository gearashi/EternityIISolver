# BOINC research notes — integration on hold

The GPU DFS port is **on hold**, and no CPU replacement worker is being developed. The application remains an offline GPU board-repair experiment. Local swaps and rotations do not count as BOINC DFS nodes, completed workunits, or credit.

A project message supplied by the user reports that a GPU DFS implementation was about **7,000× slower in the project's tests** than its CPU implementation, and that tested board-repair approaches plateaued. We have not independently reproduced those measurements. They concern the tested implementations, not all possible GPU algorithms. The message requires project-controlled production-build validation and says anonymous-platform replacement is disabled.

Do not install these examples through `app_info.xml`, replace a project executable, or submit repair results as completed production tickets. This repository does not provide an approved BOINC application. Its tests do not certify production compatibility, search coverage, or a BOINC performance improvement.

For read-only inspection of existing assignments, see [CPU workunit inspection](BOINC_CPU.md). The retained headless worker and XML templates below use a separate bounded-repair contract and are research material for local controlled tests. They do not install an application on a project server or submit results. Their CUDA mode selects an explicit NVIDIA device on Windows/Linux. OpenCL remains available in the standalone repair app; `--standalone-diagnostic` is for local tests only.

The v0.1.1 application uses bundled and previously saved board data, with no automatic index/board downloads, telemetry, or result uploads. A cached snapshot can be old or incomplete. Legacy v0.1.0 had automatic library synchronization; upgrading is necessary to obtain offline behavior. Saved checkpoints and results should be retained.

**Export best board** in the dashboard, or `EternityIISolver export-best --state-dir PATH`, creates a locally validated candidate JSON, readable layout, and validation report under `PATH/exports/`. It works with the search stopped and never uploads files. These exports are for manual review and use library-style candidate fields. The project's accepted upload schema has not been confirmed; an export is not a production CPU workunit result, completed ticket, or evidence of equivalent DFS coverage.

## Relationship to the existing project

On 8 October 2026, the project's public [application list](https://boinc.eternityathome.org/eternity/apps.php) listed Windows and Linux CPU pilot applications, including version 2.37 and a Windows 2.38 canary. Its [research description](https://stats.eternityathome.org/solver-research) describes strict five-clue DFS work divided into roots and tickets. This GPU worker performs stochastic board repair; its steps and proposed moves are different from that solver's DFS nodes. It does not accept the existing CPU ticket format. The public pages inspected did not provide a complete production workunit/validator contract to implement as a drop-in replacement.

Any future project integration would need a project-controlled build, compatible search semantics, and independent production validation before deployment. That work is currently on hold. Finishing the prototype's move budget does not establish exhaustive search, and this implementation has not demonstrated a 480/480 solution.

### What is publicly observable about production batches

The [research page](https://stats.eternityathome.org/solver-research) describes a campaign root family, sharded tickets, integrity hashes, native DFS, exact-tail work, and independent board validation. It lists 4,096 shards, a two-billion-node main cap per ticket, `surv2.txt` plus a reloadable root allowlist, and completed-ticket accounting. Tickets are reproducible bounded slices, not certificates of exhaustive coverage. Its search frame is rotated 90 degrees; publication returns boards to the official orientation. These are documented design facts, not a downloaded production input schema.

A [public Windows log from 2 October, post 110](https://boinc.eternityathome.org/eternity/forum_thread.php?id=22&postid=110) gives a concrete older example:

| Observed item | Value in that log |
| --- | --- |
| Wrapped executable | `bw_runner_233.exe` |
| Partition | Shard 258 of 4,096 |
| Jitter command bounds | 2,061,351,209 and 2,061,353,257 |
| Main node limit | 2,000,000,000 |
| CPU threads | 1 |
| Allowed-root filename | `exploration_allow.txt` |
| Reported root total | 80 |

The administrator's [explanation in post 119](https://boinc.eternityathome.org/eternity/forum_thread.php?id=22&postid=119) says this task contained roughly two tickets per root, which had caused its old progress display to finish halfway through the actual work. That is evidence that a BOINC task can bundle multiple root/ticket searches. It does not define the jitter range's endpoint semantics or guarantee that today's application uses the same packaging. The [3 October generator announcement](https://boinc.eternityathome.org/eternity/forum_thread.php?id=25) describes automatic queue refilling but does not publish the generator's schema.

The inspected pages did not link a complete source archive or sample workunit. Although the download-directory URL returned a homepage rather than an index, direct downloads of filenames identified from the public log and binaries succeeded. The following files were inspected **statically**, without executing them, on 8 October 2026. They are AMD64 PE executables; hashes identify exactly the downloaded bytes, not a permanent server version.

| Official public artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| [bw_runner_233.exe](https://boinc.eternityathome.org/eternity/download/bw_runner_233.exe) | 3,144,379 | `6247fdd1c850e87c7949da3b7a0c1a5c35eb56999b05d9b4a24103f1dc3dd442` |
| [bw_runner_237.exe](https://boinc.eternityathome.org/eternity/download/bw_runner_237.exe) | 1,247,232 | `11ad8572e63cc01c0e0f58a40e2d6c95f34f6a7939a23fcc960450b672a56a7d` |
| [solver_windows_x64.exe](https://boinc.eternityathome.org/eternity/download/solver_windows_x64.exe) | 2,418,446 | `0a16ff13598f1d52947fa3adff18111b9ffed9ef22b004a07741b079d4992daa` |
| [solver_windows_x64_236.exe](https://boinc.eternityathome.org/eternity/download/solver_windows_x64_236.exe) | 2,426,916 | `7ea0889103958d0fd1af6365ed0fd14e109acd441e5aecbfa2fbe285513ef60c` |

Runner 2.37 contains the name `solver_windows_x64_236.exe`; the older runner contains the unversioned solver name. The versioned pair is therefore the relevant static evidence for the current runner. Runner strings identify campaign hint/catalog files, a root allowlist, `256pieces.txt`, per-thread ticket journals, durable progress, a resume guard, and `result.tar.gz`. These names support a runner→solver→journal/harvest→result-archive architecture; they do not establish which files a particular workunit actually supplied or which archive members its validator requires.

Both downloaded solver engines embed a version-2 journal manifest describing little-endian records of **40 bytes**. The versioned solver contains this manifest text at file offset `0x861e0` (the opening JSON brace), and the layout field begins at `0x86303`:

| Byte offset | Type | Manifest field |
| ---: | --- | --- |
| 0 | unsigned 64-bit | `globalticket` |
| 8 | unsigned 32-bit | `cidx` (root/catalog index) |
| 12 | unsigned 32-bit | `jidx` (jitter index) |
| 16 | unsigned 64-bit | `nodes` |
| 24 | unsigned 32-bit | `dur_ms` |
| 28 | unsigned 16-bit | `max_depth` |
| 30 | unsigned 8-bit | `status`: 0 exhausted, 1 capped, 2 hit, 3 interrupted |
| 31 | unsigned 8-bit | `thread_id` |
| 32 | unsigned 64-bit | reserved |

The manifest names a run ID, configuration hash, build ID, epoch, shard index/count, root count, jitter/interleave settings, node cap, seed, and minimum saved score. Its stated deduplication tuple is `(epoch, cfg_hash, shard_index, globalticket)`, and its files follow `seg_<run-id>.t<thread>.bin`. A diagnostic format describes shard tickets as `J0*nroots + shard_index + shard_count*local`. Root-allow messages describe filtering catalog indices while preserving their original ticket/seed identities. This supports partitioning a root-by-jitter search stream into disjoint shard indices, rather than distributing one nearly solved board per task.

Runner 2.37 additionally contains a 3,600-second per-ticket CPU-budget argument. Its referenced solver contains matching budget-stop messages, `hard_tickets.txt`, and per-thread budget-state filenames. Static strings alone do not establish how every budget stop is encoded in the journal status byte; do not invent an additional status or treat all capped tickets as exhausted. Harvest metadata strings include root, seed, epoch, shard, global ticket, catalog index, and jitter index.

No client was attached and no scheduler work was requested for the static inspection above. The later authorized live-task inspection below verifies a narrower concrete batch. Output archive contents and server validation/assimilation rules still require further evidence. The GPU examples define a separate contract; they do not impersonate completed CPU tickets or claim production DFS coverage.

### Verified live batch: 8 October 2026

A running CPU task was inspected read-only on 8 October 2026. BOINC soft-link files were resolved to their project files; account, host identity, and initialization credentials were excluded. No running file was changed. These observations describe that batch, not every future campaign:

- The active application was `eternity_cpu` version 2.37, allocating one CPU. Its logical runner used `bw_runner_237.exe`, which launched `solver_windows_x64_236.exe`. Legacy executables were also present, so directory presence alone did not identify the active version.
- `campaign_catalog.txt` contained **3,782 lines**, each exactly nine comma-separated `piece_id/rotation` pairs. `exploration_allow.txt` contained **116 unique zero-based catalog indices**, one integer per line. Their placement order is solver-specific; a catalog row is not a 256-cell board in this GPU adapter's encoding.
- `campaign_hints.txt` contained five numeric rows in `piece_id row column rotation` order, excluding comments. All five matched this repository's official clues after a **180-degree clockwise board rotation**. Thus this actual frame-2 campaign differs from the older research page's described 90-degree frame. Piece IDs are one-based; row/column/rotation are zero-based.
- The downloaded piece file's 256 numeric U,D,L,R rows exactly matched this repository's rows. Its raw hash differed because it used CRLF line endings while this repository uses LF. Raw-file hashes remain intentional byte identities and must not be interchanged merely because parsed pieces agree.

The task selected shard `s=274` of `S=4096`, starting jitter `J0=251861248`, ending bound `J1=251863296`, and `N=3782` catalog roots. Its journal verified `globalticket = jidx*N + cidx`. Enumerating `J0 <= jidx < J1` for allowed roots and retaining

```text
(globalticket - J0*N) % S == s
```

produced **116 tickets, exactly one per allowed root**, agreeing with the runner's reported total. The 2,048-value jitter interval therefore did not mean 2,048 searches for each of those 116 roots on this shard. The first two complete journal records independently satisfied the equation and partition rule; both used capped status 1 and reported 2,000,000,001 nodes. Do not interpret the nominal two-billion cap as an assertion that a journal counter can never exceed it by one.

A comparison of eight simultaneous tasks found five matching totals. Three tasks with wider, 4,096-value jitter intervals reported totals of 54, 12, and 92 allowed roots, while the partition formula implied 108, 24, and 184 tickets respectively. This is an observed mismatch between reported denominators and formula-derived ticket counts. Its cause and any broader defect remain unconfirmed without completed-task or control-flow validation; a progress percentage alone does not establish ticket coverage.

The supplied wrapper job declared a 7,500-second hard limit and passed a 7,200-second solver limit, plus the node limit and exact-endgame ladder. It used a progress file and checkpoint marker and directed runner output to `result.tar.gz`. These observed CPU limits do not change the new GPU adapter's step-based budget.

The per-workunit catalog hash was `25c37addeaaa8c0f2407a1a6bbf69662fb4815c90f336ff5c7bdd60563c53962`. A separately installed campaign manifest still described a different base catalog. Therefore an installed base manifest alone is insufficient to identify the files actually supplied to a task; resolve its logical input references and hash those bytes. The raw slot snapshot is not distributed with this repository. Exact source-comment paths, account details, and host identifiers are omitted from this report.

## Files supplied

| File | Purpose |
| --- | --- |
| `boinc_worker.py` / `EternityIISolver boinc` | Headless workunit runner; no dashboard or library networking |
| `boinc/workunit.sample.json` | Small, reproducible preflight input: 32 replicas and 1,000 steps per replica |
| `boinc/job.xml` | Official-wrapper job using an allocated CUDA device |
| `boinc/input_template.xml` | One workunit JSON input, copied into the slot |
| `boinc/output_template.xml` | One bounded result JSON upload |
| `boinc/validate_result.py` | Independent candidate and workunit-binding check for a source checkout |
| `boinc/test_validate_result.py` | Offline rejection and acceptance tests |

The candidate checker is an integration example, not a registered BOINC validator daemon. It needs the source checkout's `validator.py`, `app_paths.py`, and trusted `data/` directory; copying that script alone from a native archive is insufficient.

## Optional local prototype preflight

These commands exercise only the separate repair prototype. From an installed CUDA-enabled source checkout, use a writable test directory outside any BOINC slot:

```text
eternity-solver boinc --input /path/to/boinc/workunit.sample.json --output result.json --checkpoint checkpoint.npz --progress fraction_done.txt --backend cuda --cuda-device 0
```

On Windows, use normal Windows paths. Native packages use `EternityIISolver.exe boinc` on Windows and `./EternityIISolver boinc` on Linux. Keep the entire native distribution together. For a source install with CUDA dependencies, install the repository's CUDA extra (`pip install ".[cuda]"`). An actual compatible NVIDIA driver and device are required for this nonzero-step example.

Here `0` is an explicitly chosen **local preflight** CUDA ordinal. Under BOINC, replace it with the scheduler's allocation through the wrapper macro; do not hard-code zero for volunteer tasks. Repeat the same command with the same files to exercise resume. A zero-step input exercises the JSON/validation interface without allocating a GPU, and therefore does not demonstrate GPU compatibility.

Validate the returned result from a source checkout:

```text
python boinc/validate_result.py --input boinc/workunit.sample.json --result /path/to/result.json
python -m unittest discover -s boinc -p test_*.py -v
```

The checker prints one JSON report and returns zero on acceptance, one on rejection. It never uploads data. Use `--data-dir` only to select a trusted copy of this repository's puzzle bundle.

## Workunit contract: `eternity-gpu-workunit/v1`

Every workunit is a UTF-8 JSON object. Duplicate keys, non-finite numbers, unknown fields, and invalid ranges are rejected. Integers must be JSON integers, not booleans or floating-point equivalents.

| Field | Required | Meaning |
| --- | --- | --- |
| `schema` | Yes | Exactly `eternity-gpu-workunit/v1` |
| `workunit_id` | Yes | Printable string of 1–200 characters, assigned uniquely by the server |
| `pieces_sha256` | Yes | SHA-256 of the exact bundled `data/pieces.txt` bytes |
| `replicas` | Yes | Integer 32–32768; independent parallel search states |
| `seed` | Yes | Integer 0–4294967295 |
| `steps_per_replica` | Yes | Integer 0–1000000000; bounded algorithm steps for each replica |
| `board` | No | Legal starting board of 256 integer states; defaults to bundled strict five-clue 466 board |
| `backend` | No | `auto`, `cuda`, or `opencl`; scheduled runtime requires CUDA and refuses conflicting input |
| `encoding` | No | Exactly `4*(piece_id-1)+clockwise_rotation` |
| `board_order` | No | Exactly `row-major` |

Current piece hash:

```text
1f0ec5db754c3ac94b95656f303745b85347c14adff33a0fda8f762ca79024ae
```

Piece IDs are 1–256. Rotations are 0–3 clockwise from `data/pieces.txt`, whose face order is **U,D,L,R**. Board entries are row-major from the top-left, each an integer 0–1023. All physical pieces occur once, and gray edges face exactly outside the board. The fixed zero-based cell→state pairs are `34→831`, `45→1019`, `135→554`, `210→723`, `221→992`. A board from another color numbering, piece-ID convention, global orientation, or center-clue-only contest must be converted and independently validated before use.

The workunit identity digest is SHA-256 of the **parsed original object**, serialized in Python as:

```python
json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
```

Defaults are not inserted before hashing. Whitespace and object key order do not change this digest; explicitly adding a previously omitted default does. The checkpoint and result carry this digest and the piece hash. Changing a seed, budget, or other input requires a new workunit and checkpoint path.

Use a step budget, not a wall-clock deadline: BOINC suspension must not consume the task's intended search allocation. The worker reports accumulated GPU event time as diagnostic information. A step or proposed move is not a unique board visited; stochastic search can revisit states, and seeds alone do not promise identical results across drivers or architectures.

## Result, checkpoint, and progress

`result.json` uses schema `eternity-gpu-result/v1`. It includes the workunit ID and hashes, `mode: "five-clue"`, a legal `board`, independently checked `score`, backend/device allocation, step counters, and diagnostic timing/move counters.

Completion fields have separate meanings:

- `status: "completed"`: the step budget finished or a valid 480 board was found.
- `budget_complete: true`: `completed_steps_per_replica >= steps_per_replica`.
- `complete: true`: the board matches **all 480 edges**. A normal 466 result has `complete: false` even after successfully finishing its workunit.
- `exhausted: false`: this is a heuristic search, with no claim of exhaustive coverage.

A successful bounded task exits zero. An interrupted task writes `status: "interrupted"` and exits 75; malformed input/checkpoint exits 2; other worker failures exit nonzero. Exit 75 is a worker diagnostic, **not** an implementation of BOINC's native temporary-exit protocol. The server checker rejects incomplete budgets unless the independently validated board is 480.

`checkpoint.npz` atomically combines the engine state with input binding, completed steps, accumulated GPU time, and the incumbent board. The worker resumes a matching checkpoint and rejects one from another input. Keep checkpoints private to the slot and input. `fraction_done.txt` contains a scalar 0–1 based on completed steps, or 1 for a valid solution. Checkpoints occur periodically and at graceful completion; an abrupt kill can replay work since the last checkpoint.

The supplied job declares checkpoint/progress filenames for BOINC's official wrapper. The wrapper supervises the worker process, including suspension/resumption and client lifecycle events. This Python worker does not call native `boinc_init`, checkpoint, or critical-section APIs itself. Real-client testing must establish behavior for the chosen wrapper, executable packaging, and GPU driver. See the [wrapper interface](https://github.com/BOINC/boinc/wiki/WrapperApp) and [native application API](https://github.com/BOINC/boinc/wiki/API-for-native-apps).

## Device allocation and application version

The wrapper expands `$GPU_DEVICE_NUM` from BOINC's assigned device. The worker selects that CUDA ordinal directly and preserves the inherited CUDA visibility/order environment. It rejects an unavailable ordinal and will not fall back to another backend. Do not wrap it in a script that independently rewrites `CUDA_VISIBLE_DEVICES` or changes device order after BOINC selects the ordinal. Verify two simultaneous slots on a multi-GPU host before rollout. BOINC documents CUDA allocation through `APP_INIT_DATA.gpu_device_num` in [GPU application support](https://github.com/BOINC/boinc/wiki/AppCoprocessor).

OpenCL needs a separate, exact allocation path. BOINC's native `boinc_get_opencl_ids()` maps its GPU type and `gpu_opencl_dev_index` to a platform/device; an index into this application's flattened device list is not equivalent. A future adapter must use that API, or a carefully tested bridge reproducing the [official mapping](https://github.com/BOINC/boinc/blob/master/api/boinc_opencl.cpp), and pass the exact device into the engine. Until then, AMD/Intel/Apple OpenCL support is available in the standalone application only.

The retained XML examples assume an official BOINC wrapper, which this repository does not redistribute. Their logical names are `eternity_gpu_worker` and `job.xml`. Native packages are **onedir** distributions: a hypothetical project build would need the executable and its entire `_internal` runtime directory, with relative data/library paths preserved. The standalone ZIP is not an app-version manifest.

These are packaging assumptions, not instructions to deploy on the existing project. Any project-controlled experiment would also need measured resource limits, a supported GPU [plan class](https://github.com/BOINC/boinc/wiki/Specifying-plan-classes-in-XML), and project-owned validation/assimilation. The supplied XML follows BOINC's [job-template format](https://github.com/BOINC/boinc/wiki/Job-templates). No anonymous-platform override is supplied or supported here.

`job.xml` enables process-tree supervision for executable bootloaders and uses normal feeder priority. It deliberately omits the wrapper's hard `time_limit`: the worker owns its bounded step budget, while forced termination may occur before a final checkpoint. The wrapper and application must still obey ordinary BOINC suspension/abort policy.

## Server validation and deduplication

Run `boinc/validate_result.py` against the server's retained workunit and trusted puzzle files, or port the same checks into a project validator. It checks schema/identity/hash binding, integer states, unique pieces, exact gray frame, all five clues, independently counted edge score, and consistent completion fields. It rejects a worker's claimed 480 if edge counting disagrees. Treat a valid 480 candidate as requiring an additional independent verifier before a public record claim.

Candidate legality does not prove that a volunteer performed the claimed amount of search. The project must decide its credit, replication, host-trust, anomaly-detection, and replay policy. Do not use byte-identical output as the only quorum rule for a stochastic GPU search; distinct legal results may come from the same input on different implementations. Benchmark the tradeoff between deterministic replay, redundant work, and independently verifiable improvements.

Keep the library monitor disabled inside BOINC slots: this adapter imports no library client and performs no networking. Deduplicate scheduled work by workunit identity on the server, and deduplicate accepted boards with the project's established canonical board hash. A workunit ledger avoids resending completed workunits; neither that ledger nor the board library proves that the heuristic has never visited a board. Do not equate this repository's input hash with the public board-library hash.

## Unresolved production validation requirements

- Real BOINC client: suspend/resume, leave-memory policy, quit/restart, abort, heartbeat loss, and checkpoint recovery on each supported platform.
- Actual GPU: allocation isolation on multi-GPU hosts, short kernels under desktop use, score agreement with CPU validation, resource peaks, and measured workunit duration.
- Server: successful template resolution/upload, malformed-result rejection, validator/assimilator integration, and treatment of interrupted or replayed workunits.
- Packaged app: every dependent CUDA library present, source/license notices retained, and application-version checksums generated by the project's deployment process.

The existing project's [checkpoint/suspension release notes](https://boinc.eternityathome.org/eternity/forum_thread.php?id=16) illustrate why these lifecycle checks matter. Offline tests and native package builds establish narrower properties; they do not substitute for a BOINC volunteer-client trial.
