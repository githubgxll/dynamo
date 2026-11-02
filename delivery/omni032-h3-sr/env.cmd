@echo off
if not defined H3_EXPERIMENT_PY set "H3_EXPERIMENT_PY=D:\AI\_workspace\SeedVR2_Optimization_20261030\cpu-venv\Scripts\python.exe"
if not exist "%H3_EXPERIMENT_PY%" (
  echo Set H3_EXPERIMENT_PY to the existing Python tools environment.
  exit /b 1
)
"%H3_EXPERIMENT_PY%" -B -c "import yaml, packaging"
exit /b %errorlevel%
