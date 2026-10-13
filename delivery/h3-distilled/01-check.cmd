@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
"%H3_DISTILLED_PY%" -B -X utf8 "%~dp0check_local.py" %*
exit /b %errorlevel%
