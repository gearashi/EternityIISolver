# Version 0.6.2

The Windows CUDA build checks the finished ZIP for the CuPy extensions, CUDA runtime, NVRTC, headers and wheel metadata. When CUDA startup fails, the worker reports the missing component in the dashboard. The ZIP includes its own Python. For a Windows source checkout with Python 3.15, run `scripts\setup_windows_cuda.cmd` to install the CUDA dependencies with a compatible private interpreter.

The local library shows how many records it validated and merged, saves the result in the library database and restores it when the dashboard reopens. A manual refresh makes one conditional request when the archive server provides validators. A matching snapshot skips board validation and SQLite writes. An HTTP rate limit appears in the panel.

Fast search saves validated run boards below the bundled reference. The board view and manual export can show the run's result while retaining the higher saved record.
