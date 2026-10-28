@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
if "%~1"=="" (
  echo Usage: call delivery\sr-native\04-import-evidence.cmd "D:\path\one-backend-artifact.zip"
  exit /b 1
)
"%H3_EXPERIMENT_PY%" -B -X utf8 "%~dp0import_evidence.py" "%~1"
exit /b %errorlevel%
