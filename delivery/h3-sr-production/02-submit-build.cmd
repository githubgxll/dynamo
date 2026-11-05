@echo off
setlocal
call "%~dp0env.cmd"
if errorlevel 1 exit /b %errorlevel%
"%H3_PRODUCTION_PY%" -B -X utf8 "%~dp0ops.py" submit --confirm-build-and-publish
exit /b %errorlevel%
