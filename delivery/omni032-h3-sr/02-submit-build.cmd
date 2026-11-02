@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
"%H3_EXPERIMENT_PY%" -B -X utf8 "%~dp0ops.py" submit --confirm-build-and-publish
exit /b %errorlevel%
