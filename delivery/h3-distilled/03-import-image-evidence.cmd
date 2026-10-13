@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
if "%~1"=="" (
  echo Usage: 03-import-image-evidence.cmd "C:\path\image-evidence-h3-distilled.zip"
  exit /b 1
)
"%H3_DISTILLED_PY%" -B -X utf8 "%~dp0ops.py" import-evidence "%~1"
exit /b %errorlevel%
