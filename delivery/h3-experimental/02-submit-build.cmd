@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
echo This commits and pushes to Dingorouter-H3-vllm-omni-jirx, creating it on first submission if absent.
echo The group Linux Runner will build and publish to the existing Harbor repository.
"%H3_EXPERIMENT_PY%" -B -X utf8 "%~dp0ops.py" submit --confirm-build-and-publish
exit /b %errorlevel%
