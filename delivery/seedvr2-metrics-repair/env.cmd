@echo off
if not defined H3_EXPERIMENT_PY set "H3_EXPERIMENT_PY=D:\AI\_workspace\SeedVR2_Optimization_20261030\cpu-venv\Scripts\python.exe"
call "%~dp0..\h3-experimental\env.cmd"
if errorlevel 1 exit /b 1
"%H3_EXPERIMENT_PY%" -B -c "import av, numpy, fastapi, requests, regex, imageio_ffmpeg, multipart"
exit /b %errorlevel%
