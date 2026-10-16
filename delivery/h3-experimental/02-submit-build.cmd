@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
echo This commits and pushes to the existing Dingorouter-H3-vllm-omni-jirx branch.
echo This recipe does not allow creation of a new remote branch.
echo The group Linux Runner will build and publish to the existing Harbor repository.
"%H3_EXPERIMENT_PY%" -B -X utf8 "%~dp0ops.py" submit --confirm-build-and-publish
exit /b %errorlevel%
