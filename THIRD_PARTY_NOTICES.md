# Third-party components

This repository preserves its existing GNU General Public License version 3 in [LICENSE](LICENSE). The solver source is distributed under GPL-3.0-only. Dependencies and public puzzle references retain their respective rights and notices; this file does not relicense them.

The packaged application uses Python, NumPy, PyOpenCL, and their dependencies. Windows and Linux CUDA builds additionally include CuPy and NVIDIA CUDA component libraries supplied by their official Python wheels. NVIDIA GPU drivers and hardware-vendor OpenCL drivers are not bundled. macOS OpenCL is supplied by the operating system/device environment.

Native builds generate `BUILD_INFO.json` with exact installed distribution versions and `THIRD_PARTY_LICENSES/` containing license/notice files provided in the build environment's distribution records. This may include build tools as well as runtime dependencies. Preserve those notices when redistributing a binary archive. The corresponding application source is the tagged repository/source archive identified by the build's commit and version.

Upstream projects and license information:

- [Python](https://docs.python.org/3/license.html)
- [NumPy](https://numpy.org/doc/stable/license.html)
- [PyOpenCL](https://documen.tician.de/pyopencl/misc.html#license)
- [CuPy](https://github.com/cupy/cupy/blob/main/LICENSE)
- [NVIDIA CUDA Toolkit documentation and license](https://docs.nvidia.com/cuda/eula/index.html)
- [PyInstaller licensing and bootloader exception](https://pyinstaller.org/en/stable/license.html)

Puzzle and board attribution is recorded in [DATA_PROVENANCE.md](DATA_PROVENANCE.md). Eternity II is the name of the referenced puzzle; this project does not claim affiliation with its publisher or the BOINC project.
