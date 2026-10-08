# Bundled puzzle data

The packaged data is an explicit, reviewed allowlist in `build_support.py`. Runtime status, user discoveries, checkpoints, downloaded board databases, logs, credentials, and local caches are never collected by that allowlist.

- `data/pieces.txt` contains the public 256-piece numerical definition distributed in the Igor Pejic solver / Blackwood data convention. SHA-256: `1f0ec5db754c3ac94b95656f303745b85347c14adff33a0fda8f762ca79024ae`.
- `data/record466.json` preserves the public record named `FiveClue_466_2c037e70f7e9`, attributed by its source to **Jef**. Original fetched document SHA-256: `17b715e0baae7a6b9e65eac0b75a27ce18246f441e84c034021bb8d94b983766`. The published bundle removes the unused `discoverer_key` metadata field; sanitized bundle SHA-256: `33b5d5199718a705e5ecf5e8fe46a340f555e961e2fe86c64368a693f6c6421c`. The board arrangement and attribution name are preserved.
- `data/puzzle.json` records the piece convention, five fixed clue states, color taxonomy, and source checksums. The validator independently verifies the record as 466/480.
- `data/catalog.json` and `data/catalog.csv` are derived catalogs of these public piece definitions and source arrangement.
- `data/seeds/` contains eight public library arrangements. The seed collection preserves public attribution names, including Jef, Benj, and Palace, and public board hashes. Unused `discoverer_key` fields were intentionally removed before publication.

Public sources: [Eternity@Home project](https://stats.eternityathome.org/), [solver research and clue modes](https://stats.eternityathome.org/solver-research), and the [open DFS solver](https://github.com/igorpejic/eternity-ii-dfs-solver). Attribution in a public board document is retained as source metadata; it is not independent authentication of a discoverer's identity.

The public library distinguishes puzzle/clue modes. A reported 470 board in another mode is not a 470 result under the five-clue constraints enforced here. This bundle starts from the verified five-clue 466 board and does not silently import a different mode as an equivalent seed.

Build archives include per-resource SHA-256 hashes in `BUILD_INFO.json`. `.gitattributes` treats the data files as byte-preserving resources so that platform newline conversion cannot invalidate their source hashes.
