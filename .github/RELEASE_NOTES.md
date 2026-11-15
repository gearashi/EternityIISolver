Version **0.5.0** adds independently checked local repair proofs and new exact-search and repair settings for the five-clue Eternity II puzzle. The best verified board bundled with this release is Jef’s imported **466/480**.

### Settings and results

- For an exact search with fewer tested placements, choose **Advanced search → DFS → Maximum filtering (experimental) → 1 worker**. On the difficult 9×9 official-piece benchmark it reached a checked completion in 6,941 decisions, including speculative probes and checkpoint replay. Baseline used 20,327 and finished sooner. These measurements concern the selected regional problem.
- The regional master now shares interior-edge colors, learns regional allocation conflicts and reuses verified witnesses. New SAT piece channels, explained inventory rules, impact ordering and refined diagrams are available as experimental alternatives.
- Matching repair can reuse positive witnesses and apply verified integer price bounds. Across 24 original-piece regional targets, combining the price and existing score bounds reduced placements from 239 to 75. It added 606 simplex iterations and more setup work.
- Four experimental whole-puzzle modes now get through their original-board setup. Focused tests reach a real search decision or transition. The dashboard reports DFS lookahead trials separately.
- `perfect-distance` checks how many final cell states a perfect board must change. For the bundled 466, the checked lower bound is **24**. The source includes a dependency-free verifier. The separate radius-eight enumeration excludes improvements to 467 that change eight or fewer oriented cells of the same parent.
- `repair-radius` searches for a strict score improvement within a chosen changed-cell radius. The native packages include a C++ helper. The new component bounds reduced complete cell sets checked at radius seven from 15,931,447 to 1,646,927 on the official reference. `repair-patches` combines compatible local candidates, and `repair-parent-pair` ranks local boards for recombination.

### Running the app

Extract the archive for Windows x64, Linux x64, Intel Mac or Apple Silicon Mac. Windows and Linux include CUDA and OpenCL backends. macOS uses OpenCL. Compatible system GPU drivers are required. The macOS builds are unsigned by a Developer ID certificate and are not notarized. The Python wheel and source archive are also included.

Search, proof checking and candidate exports run locally. The dashboard downloads the public board archive only when you choose **Download and merge**. See `docs/SEARCH_SETTINGS.md` for the benchmark scope, limitations, reproduction commands and settings.
