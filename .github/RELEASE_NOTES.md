Version **0.5.1** lets DFS return a verified regional or full-board solution as soon as a lookahead placement forces every cell to one state. The same validator checks the resulting board. When resuming, DFS first reconstructs the saved frontier so it can accept a solution only from a branch still owned by that search.

### What changed in the search

With **Maximum filtering** and **one CPU worker**, v0.5.0 and v0.5.1 each completed and validated all 24 original-piece regional tasks. On the southwest 9×9 interior, accepting a completed lookahead reduced the number of placements from two to one. The other 23 tasks tied, including the hard southeast 9×9 at 6,900 placements in each version. Both versions began each task from scratch with the same seed and a 17-minute limit.

Across all 24 tasks, placements fell from **6,911 to 6,910**. Both versions took 6,306 ordinary branches. The number of lookahead trials fell from 605 to 604. Recognizing completed trials added 508 scans covering 60,872 individual cell-domain checks. The observed benefit is one fewer trial on this workload. The best bundled five-clue board remains Jef’s imported **466/480**. [Search settings and benchmark details](https://github.com/gearashi/EternityIISolver/blob/v0.5.1/docs/SEARCH_SETTINGS.md)

Native downloads are available for Windows x64, Linux x64, Intel Mac and Apple Silicon Mac. Windows and Linux include CUDA and OpenCL. macOS uses OpenCL. A Python wheel and source archive are also available.
