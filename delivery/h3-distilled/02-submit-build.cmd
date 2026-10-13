@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
echo This command checks, commits and pushes the distilled image branch.
echo The group Runner builds and publishes to the existing Harbor repository.
"%H3_DISTILLED_PY%" -B -X utf8 "%~dp0ops.py" submit --confirm-build-and-publish
exit /b %errorlevel%
