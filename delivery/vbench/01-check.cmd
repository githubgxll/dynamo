@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
"%VBENCH_PY%" -B -X utf8 "%~dp0ops.py" check
exit /b %errorlevel%
