Version **0.6.1** adds a compiled search core and two engines that use it. Both run offline on the CPU with the five clues fixed, use one thread per worker (up to 64), and continue from their checkpoints after Stop.

- **Fast compiled search for the highest score** looks for complete boards with as many matching edges as possible. It allows a limited number of mismatching edges late in the board, uses up three colors early, and restarts with new random orders. Each complete board it finds tightens the allowance, so the next one must score higher.
- **Fast compiled DFS for all 480** is a plain exhaustive search. Its workers split one search tree, so no branch is searched twice.

| Bounded test, four threads | Time | Pieces tried | Result |
| --- | ---: | ---: | --- |
| Fast compiled DFS | 60 s | 14.1 billion | Deepest placement 200 of 256; no solution |
| Fast compiled score search | 1,500 s | 358.5 billion | Complete boards of 459 (1 s), 460 (344 s) and 461 (1,191 s) |

The engines try about 235 million pieces per second on four threads of a 2.1 GHz cloud processor. They apply none of the Python DFS pruning rules, so the speed does not make the puzzle tractable. Neither engine solved the puzzle, and the score search stayed below the verified **466/480** reference in this test. Each further point took several times longer than the one before. Every board is re-scored from the piece list and validated before it can replace your saved best.

To try it, select **Advanced search → Fast compiled search for the highest score**, choose one worker per CPU core and press Start. Your saved best board, exports and other saved searches are kept. [How it works, measurements and limits](https://github.com/gearashi/EternityIISolver/blob/v0.6.1/docs/FAST_SEARCH.md)

Version 0.6.0 was tagged but never released. Its Windows build check found that the compiled search could lose its final report when the app read the status file at the same moment, which ended the run with an error. On Windows the search now waits for the reader and then writes the report.
