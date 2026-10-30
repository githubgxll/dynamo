@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
echo This requests another SeedVR2 optimization build with a new empty commit on the EXISTING branch. No gh is needed.
"%H3_EXPERIMENT_PY%" -B -X utf8 "%~dp0ops.py" dispatch
exit /b %errorlevel%
