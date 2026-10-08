# Eternity II Solver

A local GPU search application for the **five-clue Eternity II puzzle**. It starts from an independently verified **466/480** board, runs parallel searches, checks improvements on the CPU, and displays progress in a local browser dashboard. It supports NVIDIA through CUDA and AMD, Intel, Apple, or NVIDIA GPUs through OpenCL when compatible drivers are available.

The application does not guarantee a 480/480 solution. Public records from other clue modes, including 470 boards, are not equivalent to a five-clue result and are not silently substituted for the bundled starting board.

## Download and start

Get the archive matching your computer from [Releases](https://github.com/gearashi/EternityIISolver/releases), then extract the **entire archive**.

| Asset | Architecture | Compute backends | Driver requirement |
| --- | --- | --- | --- |
| Windows | x64 | CUDA, OpenCL | NVIDIA CUDA-compatible driver, or vendor OpenCL GPU driver |
| Linux | x64 | CUDA, OpenCL | NVIDIA driver, or AMD/Intel/NVIDIA OpenCL ICD exposing a GPU |
| macOS Intel | x86-64 | OpenCL | A GPU exposed by the available macOS OpenCL runtime |
| macOS Apple Silicon | ARM64 | OpenCL | A GPU exposed by the available macOS OpenCL runtime |

Open `EternityIISolver.exe` on Windows, `EternityIISolver.app` on macOS, or `EternityIISolver` on Linux. This opens the dashboard at [127.0.0.1:8765](http://127.0.0.1:8765/) **without starting a search**. Choose the backend and number of parallel searches (default **4096**), then click **Start**. To change those settings during a session, stop the search and start it again with the new values.

**Stop** requests a clean stop and checkpoint. The dashboard stays available afterward. Closing the browser does not stop an active search. On systems where a graphical launch is inconvenient, use the commands below.

Native archives include Python and application dependencies, but hardware drivers remain system supplied. OpenCL support is conditional on the driver exposing a suitable GPU; normal search does not silently fall back to CPU OpenCL. AMD/Intel users should select **OpenCL** explicitly. Apple has [deprecated OpenCL](https://developer.apple.com/opencl/); macOS builds therefore depend on the runtime still exposed by the target Mac. The macOS archives are not Apple-notarized. They are built natively on macOS 15 Intel and macOS 14 ARM runners; older macOS versions are not certified.

## Commands

Use `EternityIISolver.exe` on Windows or `./EternityIISolver` on Linux. For a macOS app archive, the terminal executable is `EternityIISolver.app/Contents/MacOS/EternityIISolver`.

```text
EternityIISolver open
EternityIISolver start --backend auto --replicas 4096
EternityIISolver start --backend opencl --replicas 2048
EternityIISolver stop
EternityIISolver status --json
EternityIISolver validate
EternityIISolver validate saved-board.json
EternityIISolver diagnose --backend opencl --replicas 128
```

`open` and the default no-argument launch open the persistent local dashboard. `start` additionally starts the worker. `run` runs the worker in the foreground; `--seconds 60` makes it bounded, otherwise it continues until stopped, an error occurs, or a validated solution is found. `--no-library` disables public-library synchronization. `--no-browser` is available with `start`. Use `--help` on a command for its options.

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
# OpenCL: AMD, Intel, Apple, or NVIDIA with an appropriate GPU driver.
python -m pip install .

# Optional NVIDIA CUDA support on Windows/Linux, including the tested CUDA runtime and NVRTC component wheels.
python -m pip install ".[cuda]"

eternity-solver open
```

A GitHub release wheel can also be installed with `python -m pip install path/to/eternity_ii_solver-0.1.0-py3-none-any.whl`. Install the CUDA extra from the source checkout if desired. The repository is not claimed to be published on PyPI.

PyOpenCL and a hardware-vendor OpenCL implementation are separate requirements. A successful Python installation alone does not supply an AMD/Intel/NVIDIA GPU driver. See [PyOpenCL installation](https://documen.tician.de/pyopencl/misc.html#installation) and [CuPy installation](https://docs.cupy.dev/en/stable/install.html) for supported runtime configurations.

## Saved state and public library

Application resources are read-only. Runtime state is stored separately:

| System | Default state directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%\EternityIISolver` |
| macOS | `~/Library/Application Support/EternityIISolver` |
| Linux | `$XDG_DATA_HOME/eternity-ii-solver`, or `~/.local/share/eternity-ii-solver` |

Use a command's `--state-dir PATH` option or the `ETERNITY_SOLVER_HOME` environment variable to choose another directory. Keep the state directory if you want to retain checkpoints and the board cache across upgrades. Dashboard/worker logs, current status, checkpoints, the library cache, and verified results live there; they are not written into the app bundle. The solver restores compatible checkpoints and validates them before use.

The library initially loads its public metadata index, then downloads full board arrangements gradually, highest scores first, at approximately one request per second. Indexed records are **not** fully cached arrangements. Duplicate detection uses exact cached placements; when the relevant score tier is incomplete, novelty remains unknown. Being absent from a cached index snapshot is not a claim of worldwide novelty. Normal desktop search does not submit results to BOINC. The separate [experimental BOINC wrapper adapter](docs/BOINC.md) accepts bounded workunits, writes checkpoint/progress files and independently checked results; project-side deployment still requires team testing.

## Search and validation

The GPU evaluates swaps and rotations across many replicas, uses temperature schedules to escape local optima, and periodically reseeds part of the population. The CPU independently verifies each promoted improvement against every piece, all 480 internal adjacencies, the gray frame, and the five fixed clue states. A legal partial arrangement is not a solved board: the validator's `complete` field requires 480 matched edges.

The imported 466 board is credited to its public source, not presented as a discovery by this software. [DATA_PROVENANCE.md](DATA_PROVENANCE.md) records source attribution and hashes. Board states use `4 * (piece ID - 1) + clockwise rotation`, with zero-based row-major cells. Source piece edges are U,D,L,R; oriented solver edges are U,R,D,L.

## Tests and builds

CPU tests require no GPU and do not start a continuous search:

```sh
python -m unittest test_validator test_library_cache test_lifecycle test_gpu_backends test_boinc_worker -v
python -m unittest discover -s boinc -p "test_*.py" -v
python launcher.py validate
```

Run the bounded hardware diagnostics explicitly:

```sh
python test_gpu.py --backend cuda --replicas 128 --output cuda-test.json
python test_gpu.py --backend opencl --replicas 128 --output opencl-test.json
```

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

Build on the target operating system and architecture; PyInstaller does not cross-compile these archives. Each build validates the bundled 466 board, reads status and loads the diagnostic/BOINC command help from the frozen executable before archiving. `release-assets/` receives the native archive and its SHA-256 file. To build Python packages, run `python -m build`; `scripts/check_wheel.py` checks that a wheel contains the curated resources and excludes runtime data.

GitHub Actions builds Windows x64, Linux x64, macOS Intel, and macOS ARM separately. Branch/PR runs expose downloadable workflow artifacts. A pushed version tag such as `v0.1.0` must match `pyproject.toml`; after every build succeeds, the workflow publishes the native archives, Python wheel/source archive, and checksums to a GitHub release. A built package and a successful CPU smoke test do not establish compatibility with every GPU or driver.

## License

The existing [GNU GPL version 3 license](LICENSE) is preserved. Third-party components and public puzzle data retain their respective notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This is an independent project, not an official Eternity II or BOINC client.
