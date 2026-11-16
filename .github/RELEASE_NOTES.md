Version **0.5.3** reduces repeated work when DFS checks whether the remaining cells can be filled independently. If a required edge still joins two undecided cells, that shortcut cannot finish the board. DFS remembers the edge and rechecks its two cells the next time. When either cell becomes decided, the ordinary completion check runs again.

On 24 five-clue regional tasks, DFS made **6,910 search choices**. Counting both checks along that same search path showed that the remembered edge avoided **2,942 of 2,944 matching validations**. Scans across the full set of cell domains went from **753,664 individual cell inspections to 512**, with **5,884 additional endpoint checks**.

The release candidate returned the same independently verified board as v0.5.2 in all 24 regions and used the same 6,910 placements, including lookahead trials. The best bundled five-clue board remains Jef’s imported **466/480**. For fewer placements on the tested workloads, use **Advanced search → DFS → Maximum filtering (experimental) → 1 worker**. [Settings and measurements](https://github.com/gearashi/EternityIISolver/blob/v0.5.3/docs/SEARCH_SETTINGS.md)

Downloads include native packages for Windows x64, Linux x64, Intel Mac and Apple Silicon Mac, plus a Python wheel and source archive.
