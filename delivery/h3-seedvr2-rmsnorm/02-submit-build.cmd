@echo off
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
echo This commits and pushes the sealed RMSNorm derivative to the EXISTING Dingorouter-H3-vllm-omni-jirx branch.
echo The group Runner will validate the candidate image before publishing a NEW srnormfix tag.
echo The successful parent image and current Pods are preserved.
"%H3_EXPERIMENT_PY%" -B -X utf8 "%~dp0ops.py" submit --confirm-build-and-publish
exit /b %errorlevel%
