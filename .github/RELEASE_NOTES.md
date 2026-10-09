This v0.1.1 update makes normal operation **offline**. The app uses bundled and previously saved board data, with no automatic library-index refreshes, board downloads, telemetry, or result uploads. Cached snapshots may be old or incomplete. Legacy v0.1.0 had automatic library synchronization and must be replaced to obtain this behavior. Keep existing checkpoints and results; no search reset is required.

Native packages cover Windows x64, Linux x64, macOS Intel, and macOS Apple Silicon. Windows/Linux include CUDA and OpenCL backends; macOS uses OpenCL. AMD and Intel GPU users select OpenCL and need their hardware vendor's compatible OpenCL driver.

Extract the entire archive. Open the executable (the .app on macOS) to open the local dashboard. Choose the number of parallel searches and click Start; Stop requests a clean checkpoint. Opening the application does not start a search automatically.

**Export best board** saves validated board JSON, a readable layout, and a validation report locally, with download links for manual review. The `export-best` command provides the same export without a running dashboard or search. Exports never upload automatically. The project's accepted upload schema remains unconfirmed; exported candidate files are not completed BOINC CPU workunits.

The bundled source board is verified at 466/480 under all five official clues. Scores reported under other clue modes, including 470 boards, are not interchangeable with this mode. This is GPU board repair and can plateau. Its local move counts measure swaps and rotations, not BOINC DFS nodes, completed tasks, or credit. There is no guarantee of 480/480.

Every native build runs frozen CPU validation and status smoke tests. Linux additionally runs the source and frozen OpenCL kernel diagnostics on PoCL's CPU device. Those checks do not certify real AMD, Intel, or Apple GPU hardware. Initial local CUDA and OpenCL diagnostics passed on an NVIDIA RTX 2060. macOS packages are not Apple-notarized. See README for driver, source-install, and data-location details.

No user checkpoints, downloaded library databases, local discoveries, logs, or credentials are included. Archives contain exact dependency versions, resource checksums, and third-party notices. SHA256SUMS.txt covers the downloadable archives and Python packages.

The GPU DFS port is on hold; no CPU replacement worker is being developed. The read-only CPU workunit inspector and bounded-repair wrapper examples remain research tools, not an approved BOINC application. Do not use them as anonymous-platform replacements or submit their outputs as completed production CPU tickets. A project message supplied by the user reports a roughly 7,000× slowdown in the project's GPU DFS tests and repair plateaus; these results have not been independently reproduced here. See docs/BOINC.md for the attribution and unresolved production-validation boundary.
