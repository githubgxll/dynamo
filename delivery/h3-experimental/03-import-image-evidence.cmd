@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
if "%~1"=="" (
  echo Usage: call delivery\h3-experimental\03-import-image-evidence.cmd "C:\path\actions-artifact.zip"
  exit /b 1
)
"%H3_EXPERIMENT_PY%" -B -X utf8 "%~dp0ops.py" import-evidence "%~1"
exit /b %errorlevel%
