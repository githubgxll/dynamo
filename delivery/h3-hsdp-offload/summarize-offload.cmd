@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
"%H3_EXPERIMENT_PY%" -B -X utf8 "%~dp0summarize_offload_timings.py" %*
exit /b %errorlevel%
