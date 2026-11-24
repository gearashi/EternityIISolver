@echo off
setlocal
cd /d "%~dp0.." || exit /b 1
rem Use a private x64 Python 3.12 when the installed Python has no solver wheels.
set "TOOLS=%CD%\.venv-cuda-tools"
set "TARGET=%CD%\.venv-cuda"
if not exist "%TOOLS%\Scripts\python.exe" (
    py -m venv "%TOOLS%" || exit /b 1
)
if not exist "%TOOLS%\Scripts\uv.exe" (
    "%TOOLS%\Scripts\python.exe" -m pip install --only-binary=:all: "uv>=0.9,<1" || exit /b 1
)
set "UV_PYTHON_INSTALL_DIR=%TOOLS%\interpreters"
set "UV_PYTHON_INSTALL_REGISTRY=false"
set "UV_PYTHON_INSTALL_BIN=false"
if not exist "%TARGET%\Scripts\python.exe" (
    "%TOOLS%\Scripts\uv.exe" venv --python cpython-3.12-windows-x86_64-none "%TARGET%" || exit /b 1
)
"%TOOLS%\Scripts\uv.exe" pip install --only-binary=:all: --python "%TARGET%\Scripts\python.exe" -r requirements-cuda.txt || exit /b 1
echo To start the dashboard, run .venv-cuda\Scripts\python.exe launcher.py open
