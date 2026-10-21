@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
if "%~1"=="" (
  echo Usage: call delivery\omni-gateway-lite\04-import-image-evidence.cmd "C:\path\image-evidence.zip"
  exit /b 1
)
"%H3_EXPERIMENT_PY%" -B -X utf8 "%~dp0import_evidence.py" "%~1"
exit /b %errorlevel%
