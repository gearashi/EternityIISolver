# CUDA fast-score experiment

From the repository root, run the experiment using the Windows source CUDA environment:

```powershell
.venv-cuda\Scripts\python.exe scripts\experimental\cuda_fast_score\cuda_fast_score.py --repo . --out cuda-trial --seed 20261124 --lanes 2048 --seconds 60
```

Use a fresh output directory for each run. Each CUDA lane performs a bounded portion of the row-major search between host checks. The solver validates complete boards on the CPU; accepted boards, elapsed time, placement counts and maximum depth are written locally. The search uses the official five clues, the same end-heavy mismatch schedule and the same depletion colors as the native score ladder. A time limit gives a bounded heuristic result.

To check CPU and GPU traversal on the small strict and score fixtures, point the test at the native helper from the extracted Windows release:

```powershell
.venv-cuda\Scripts\python.exe scripts\experimental\cuda_fast_score\check_cuda_fast_score.py --repo . --helper C:\path\to\EternityIISolver\_internal\native\eternity-fast.exe --out cuda-fixtures
```
