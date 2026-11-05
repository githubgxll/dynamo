@echo off
cd /d "%~dp0..\.." || exit /b 1
if not defined H3_PRODUCTION_PY set "H3_PRODUCTION_PY=python"
"%H3_PRODUCTION_PY%" -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python 3.10 or newer is required. Set H3_PRODUCTION_PY to its executable path. 1>&2
  exit /b 1
)
exit /b 0
