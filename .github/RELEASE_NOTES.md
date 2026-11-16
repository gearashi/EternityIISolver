Version **0.5.2** reduces the work DFS spends checking whether a lookahead placement completed the board. If a previous trial left a cell undecided, the next trial inspects that cell first. It returns a solution only after every cell has a single placement and the usual validator accepts the board.

With **Maximum filtering** and **one CPU worker**, this reduced individual cell-domain checks from **60,872 to 889** across 24 five-clue, original-piece regional tasks. Both versions made the same **6,910 search choices**, including 6,900 on the hard southeast 9×9, and returned identical independently validated boards on every task. Across nine small generated cases the checks fell from 193 to 169 overall, with one case requiring one additional check.

These counts measure work inside the completion check. The full puzzle’s time to solution is unknown. The best bundled five-clue board remains Jef’s imported **466/480**. [Settings and benchmark results](https://github.com/gearashi/EternityIISolver/blob/v0.5.2/docs/SEARCH_SETTINGS.md)

Native downloads are available for Windows x64, Linux x64, Intel Mac and Apple Silicon Mac. Windows and Linux include CUDA and OpenCL. macOS uses OpenCL. A Python wheel and source archive are also available.
