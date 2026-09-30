@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
if "%~1"=="" (
  echo Usage: 03-import-preparation.cmd "C:\path\vbench-prepare-artifact.zip"
  exit /b 1
)
"%VBENCH_PY%" -B -X utf8 "%~dp0ops.py" import-preparation "%~1"
exit /b %errorlevel%
