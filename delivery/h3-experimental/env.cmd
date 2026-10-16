@echo off
set "H3_EXPERIMENT_REPO=%~dp0..\.."
if defined H3_EXPERIMENT_PY goto check_python
set "H3_EXPERIMENT_PY=D:\AI\_workspace\H3_Omni030_Image_20260929\tools-venv\Scripts\python.exe"
:check_python
if not exist "%H3_EXPERIMENT_PY%" (
  echo Python tools environment is missing. Set H3_EXPERIMENT_PY to a Python with PyYAML, Jinja2 and packaging.
  exit /b 1
)
"%H3_EXPERIMENT_PY%" -B -c "import yaml, jinja2, packaging"
if errorlevel 1 exit /b 1
exit /b 0
