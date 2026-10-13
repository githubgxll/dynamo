@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
echo Optional Windows download: about 5.64 GB. Does not upload or start CI.
"%H3_DISTILLED_PY%" -B -X utf8 "%~dp0ops.py" fetch
exit /b %errorlevel%
