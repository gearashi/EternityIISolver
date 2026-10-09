# Eternity II Solver

An **offline puzzle-search application** for the **five-clue Eternity II puzzle**, with CPU exact search and GPU board repair in one local dashboard. Every returned solution or promoted board is independently checked against the pieces, frame, five clues, and all 480 shared edges.

**CPU exact search** offers a pruned Python DFS baseline, SAT through Glucose, and CP-SAT through OR-Tools. It seeks a complete **480/480** solution; the displayed saved best board does not improve incrementally during this search. DFS can resume a compatible saved frontier. SAT and CP-SAT restart their native search after Stop. See [Exact search](docs/EXACT_SEARCH.md) for the models, measured comparison scope, and limitations.

**Experimental CPU + GPU hybrid** adds a short GPU sampling stage before CPU DFS. It accepts low-scoring full arrangements, freezes at most 64 diverse CPU-validated hints, then releases the GPU. Hint votes change branch order only; the exact pruning rules remain responsible for rejecting impossible assignments. The requested target of 100,000 samples per second is a measurement target, not a guarantee or evidence of a solution speedup. DFS remains the default exact engine.

The [bounded sampler tests](docs/EXACT_SEARCH.md#measured-gpu-sampling-9-october-2026) measured about **12.4 million samples/second with CUDA** and **11.5 million with OpenCL** on an RTX 2060, excluding compilation and initialization. Samples may repeat and are mostly low-scoring; only the retained hint set is CPU-validated. A separate equal-budget CPU comparison ended in timeouts for both ordinary and hint-guided DFS, so no exact-search speedup is established. The verified best remains **466/480**.

**GPU board repair** starts from an independently verified **466/480** board and explores swaps and rotations in parallel. NVIDIA is supported through CUDA; AMD, Intel, Apple, and NVIDIA GPUs can use OpenCL when compatible drivers are available. Repair can plateau and does not guarantee a solution. Move counts, CPU branches, and conflicts are local search statistics, not BOINC nodes, completed workunits, or credit. Public records from other clue modes, including 470 boards, are not equivalent to a five-clue result.

BOINC-compatible worker development and the GPU DFS port remain **on hold**. Standalone CPU exact search does not execute assigned production tickets or submit results. The read-only workunit inspector remains available for examining saved inputs.

**Version boundary:** v0.2.0 adds CPU exact search. Offline operation was introduced in v0.1.1. Legacy **v0.1.0** can automatically synchronize the public library; it does not acquire offline behavior without upgrading.

## Download and start

Get the archive matching your computer from [Releases](https://github.com/gearashi/EternityIISolver/releases), then extract the **entire archive**.

| Asset | Architecture | Search engines | GPU repair / hybrid driver requirement |
| --- | --- | --- | --- |
| Windows | x64 | CPU DFS/SAT/CP-SAT; CUDA/OpenCL repair | NVIDIA CUDA-compatible driver, or vendor OpenCL GPU driver |
| Linux | x64 | CPU DFS/SAT/CP-SAT; CUDA/OpenCL repair | NVIDIA driver, or AMD/Intel/NVIDIA OpenCL ICD exposing a GPU |
| macOS Intel | x86-64 | CPU DFS/SAT/CP-SAT; OpenCL repair | A GPU exposed by the available macOS OpenCL runtime |
| macOS Apple Silicon | ARM64 | CPU DFS/SAT/CP-SAT; OpenCL repair | A GPU exposed by the available macOS OpenCL runtime |

Open `EternityIISolver.exe` on Windows, `EternityIISolver.app` on macOS, or `EternityIISolver` on Linux. This opens the dashboard at [127.0.0.1:8765](http://127.0.0.1:8765/) **without starting a search**. Choose **CPU exact / hybrid** or **GPU board repair**, then select the engine and click **Start**. CPU-only DFS, SAT, and CP-SAT need no GPU driver. DFS is the recommended fallback when native SAT or CP-SAT encoding is too costly; it is a Python baseline, with no claim to match an optimized native DFS implementation. The hybrid engine is experimental and needs a supported GPU for a fresh sampling stage.

DFS, SAT, and hybrid use one CPU worker. CP-SAT allows **1–4** workers. GPU repair retains its backend selector and **32–32,768** parallel searches (default **4096**); those are GPU replicas, not CPU workers. Hybrid exposes the same numeric control as **GPU sample batch** for its short sampling stage. More replicas, larger batches, or more workers are not guaranteed to be faster. Stop the active search before changing settings.

**Stop & save** requests a clean stop and saves local state. It preserves the best board and exports; DFS and GPU repair also save resumable search state. Hybrid reuses its frozen hints and separate DFS frontier on resume. SAT/CP-SAT native search state is not checkpointed. The dashboard stays available afterward. Closing the browser does not stop an active search.

Native archives include Python and application dependencies, but hardware drivers remain system supplied. Linux builds target the Ubuntu 22.04 runtime baseline and use the host system's `libstdc++` and `libgcc`, so newer GPU drivers load their matching C++ runtime. OpenCL support is conditional on the driver exposing a suitable GPU; normal search does not silently fall back to CPU OpenCL. AMD/Intel users should select **OpenCL** explicitly. Apple has [deprecated OpenCL](https://developer.apple.com/opencl/); macOS builds therefore depend on the runtime still exposed by the target Mac. The macOS archives are not Apple-notarized. They are built natively on macOS 15 Intel and macOS 14 ARM runners; older macOS versions are not certified.

## Commands

Use `EternityIISolver.exe` on Windows or `./EternityIISolver` on Linux. For a macOS app archive, the terminal executable is `EternityIISolver.app/Contents/MacOS/EternityIISolver`.

```text
EternityIISolver open
EternityIISolver start --method exact --exact-engine dfs --workers 1
EternityIISolver run --method exact --exact-engine cp-sat --workers 1 --seconds 60
EternityIISolver exact --engine sat --workers 1 --seconds 60
EternityIISolver start --method exact --exact-engine hybrid --backend auto --replicas 4096
EternityIISolver exact --engine hybrid --backend auto --replicas 4096 --seconds 60
EternityIISolver start --method gpu --backend auto --replicas 4096
EternityIISolver start --method gpu --backend opencl --replicas 2048
EternityIISolver stop
EternityIISolver status --json
EternityIISolver validate
EternityIISolver validate saved-board.json
EternityIISolver export-best --state-dir PATH
EternityIISolver diagnose --backend opencl --replicas 128
EternityIISolver diagnose-exact
EternityIISolver diagnose-hybrid --backend cuda --seconds 3 --replicas 4096
```

`open` and the default no-argument launch open the persistent local dashboard. `start` additionally starts the worker. `run` runs the worker in the foreground; `--seconds 60` makes it bounded, otherwise it continues until stopped, an error occurs, a validated solution is found, or an exact search exhausts its scope. `exact` invokes the headless CPU worker directly. For compatibility, `start`/`run` default to `--method gpu`; select exact search explicitly. Runtime library downloads are disabled. `--no-library` skips GPU repair's local cache checks; it is not needed to prevent networking. `--no-browser` is available with `start`. Use `--help` on a command for its options.

For AMD or Intel GPUs, use `--backend opencl`. When several OpenCL devices exist, `ETERNITY_OPENCL_DEVICE` and `ETERNITY_OPENCL_PLATFORM` can filter device/platform names. `--backend cuda` selects NVIDIA CUDA explicitly; `auto` tries the available supported backends.

## Install from source

Use Python **3.11 or newer** (release builds use Python 3.12) in a virtual environment:

```sh
git clone https://github.com/gearashi/EternityIISolver.git
cd EternityIISolver
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. Then choose one installation:

```sh
# CPU DFS, SAT and CP-SAT; also includes OpenCL support for compatible GPUs.
python -m pip install .

# Optional NVIDIA CUDA support on Windows/Linux, including the tested CUDA runtime and NVRTC component wheels.
python -m pip install ".[cuda]"

eternity-solver open
```

A GitHub release wheel can also be installed with `python -m pip install path/to/eternity_ii_solver-0.2.0-py3-none-any.whl` when that release is available. The core dependencies include OR-Tools 9.15 and python-sat; no separate exact-search extra is required. Install the CUDA extra from the source checkout if desired. Downloading releases, installing dependencies, and manually opening documentation links require internet access; normal application operation does not. The repository is not claimed to be published on PyPI.

PyOpenCL and a hardware-vendor OpenCL implementation are separate requirements. A successful Python installation alone does not supply an AMD/Intel/NVIDIA GPU driver. See [PyOpenCL installation](https://documen.tician.de/pyopencl/misc.html#installation) and [CuPy installation](https://docs.cupy.dev/en/stable/install.html) for supported runtime configurations.

## Saved state and offline cache

Application resources are read-only. Runtime state is stored separately:

| System | Default state directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%\EternityIISolver` |
| macOS | `~/Library/Application Support/EternityIISolver` |
| Linux | `$XDG_DATA_HOME/eternity-ii-solver`, or `~/.local/share/eternity-ii-solver` |

Use a command's `--state-dir PATH` option or the `ETERNITY_SOLVER_HOME` environment variable to choose another directory. Keep the state directory if you want to retain checkpoints and the board cache across upgrades. Dashboard/worker logs, current status, checkpoints, the library cache, and verified results live there; they are not written into the app bundle. GPU populations, ordinary DFS frontiers, and hybrid hints/frontiers use separate saved state and are validated before resuming. SAT/CP-SAT retain run status and the saved best board but rebuild their native search on each start.

The application uses bundled and previously saved board data only. It does not automatically refresh the public index, download board arrangements, upload results, or send telemetry. The dashboard connects only to its local server; its research link opens an external website only when selected. Cached metadata is **not** a complete set of cached arrangements, and the snapshot may be old. An absent board cannot establish worldwide novelty.

Click **Export best board**, or run `EternityIISolver export-best --state-dir PATH`, to validate and save the best available board under the state directory's `exports/` folder. Omit `--state-dir PATH` to use the normal state directory. Export works while the search is stopped and provides board JSON, a readable layout, and a validation report. The dashboard keeps local download links visible for review. Nothing is uploaded automatically. The JSON uses library-style candidate fields; the project's accepted upload schema is unconfirmed, and these files are not completed CPU workunits or proof of search coverage. Review the files before any manual sharing.

The [CPU workunit inspector](docs/BOINC_CPU.md) checks selected assignment inputs and enumerates ticket IDs without running a search. BOINC integration is on hold. The separate bounded-repair interface and wrapper examples in [BOINC.md](docs/BOINC.md) are research material, not an approved project application. Do not install them as an anonymous-platform replacement or submit their results as completed production CPU tickets.

A project message supplied by the user reports that the project's tested GPU DFS implementation was about **7,000× slower** than its CPU implementation, and that tested repair approaches plateaued. Those measurements have not been independently reproduced here and do not establish the performance of every GPU algorithm. The message also requires project-controlled production-build validation and says anonymous-platform replacement is disabled. Local move throughput is not evidence of a BOINC speedup.

## Search and validation

CPU exact search enforces one use of each piece, legal rotations, the gray frame, the five fixed clue states, and all 480 edge equalities. Its runtime is not bounded in advance: a timeout or manual stop is inconclusive. Exhausting a restored DFS frontier is not an independent unsatisfiability certificate. The [exact-search notes](docs/EXACT_SEARCH.md) distinguish whole-puzzle search from restricted regional benchmark tasks.

GPU repair evaluates swaps and rotations across many replicas, uses temperature schedules to explore beyond local optima, and periodically perturbs part of the population. The CPU independently verifies each promoted improvement. A legal partial arrangement is not a solved board: the validator's `complete` field requires 480 matched edges. Updating the application does not require deleting checkpoints or resetting a search.

The imported 466 board is credited to its public source, not presented as a discovery by this software. [DATA_PROVENANCE.md](DATA_PROVENANCE.md) records source attribution and hashes. Board states use `4 * (piece ID - 1) + clockwise rotation`, with zero-based row-major cells. Source piece edges are U,D,L,R; oriented solver edges are U,R,D,L.

## Tests and builds

CPU tests require no GPU and do not start a continuous search:

```sh
python -m unittest test_validator test_library_cache test_lifecycle test_gpu_backends test_boinc_worker test_boinc_cpu_workunit test_manual_export -v
python -m unittest test_exact_cp test_exact_sat test_exact_dfs test_exact_worker test_exact_integration test_exact_pruning test_hybrid_sampling test_hybrid_worker -v
python -m unittest discover -s boinc -p "test_*.py" -v
python launcher.py validate
python launcher.py diagnose-exact
```

Run the bounded hardware diagnostics explicitly:

```sh
python test_gpu.py --backend cuda --replicas 128 --output cuda-test.json
python test_gpu.py --backend opencl --replicas 128 --output opencl-test.json
python launcher.py diagnose-hybrid --backend cuda --seconds 3 --replicas 4096 --output hybrid-sampling.json
```

`diagnose-hybrid` checks two tiny GPU batches on the CPU, then measures a bounded sampling stage. Use `--backend opencl` for an OpenCL device. Its sample rate includes generation, GPU scoring, final selected transfers, and CPU screening, while compilation and initialization are timed separately. It measures sampled arrangements, not unique boards or exact-search speed.

Initial diagnostics passed on an NVIDIA RTX 2060 through both CUDA and OpenCL, covering score and adjacent-move deltas, legal states, inverse maps, reseeding, checkpoint/resume, and adaptation. AMD, Intel, and Apple GPU hardware has not been tested locally. The Linux CI diagnostics exercise both source and frozen executables using `--allow-opencl-cpu` with PoCL solely to verify the OpenCL code path; it is not a GPU performance test.

Native build commands:

```sh
python -m pip install -r requirements-build.txt
# Windows/Linux CUDA+OpenCL package only:
python -m pip install -r requirements-cuda.txt
python scripts/build_release.py --flavor cuda-opencl
# macOS, or an OpenCL-only package on another platform:
python scripts/build_release.py --flavor opencl
```

Build on the target operating system and architecture; PyInstaller does not cross-compile these archives. Each build validates the bundled 466 board, reads status, loads the diagnostic/BOINC command help, and runs tiny solvable and infeasible cases through all three CPU engines in the frozen executable before archiving. `release-assets/` receives the native archive and its SHA-256 file. To build Python packages, run `python -m build`; `scripts/check_wheel.py` checks that a wheel contains the curated resources and excludes runtime data.

GitHub Actions builds Windows x64, Linux x64, macOS Intel, and macOS ARM separately. Branch/PR runs expose downloadable workflow artifacts. A pushed version tag such as `v0.2.0` must match `pyproject.toml`; after every build succeeds, the workflow publishes the native archives, Python wheel/source archive, and checksums to a GitHub release. Native archives include the CPU exact-search dependencies. A built package and a successful CPU smoke test do not establish compatibility with every GPU or driver.

## License

The existing [GNU GPL version 3 license](LICENSE) is preserved. Third-party components and public puzzle data retain their respective notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This is an independent project, not an official Eternity II or BOINC client.
