@echo off
if not defined VBENCH_PY set "VBENCH_PY=D:\AI\_workspace\H3_Omni030_Image_20260929\tools-venv\Scripts\python.exe"
if not exist "%VBENCH_PY%" (
  echo Python not found. Set VBENCH_PY to the existing tools Python with PyYAML and pytest.
  exit /b 1
)
set "VBENCH_CONTEXT=server.teleport.hd-04.zetyun.cn-hd04-cci-k8s"
set "VBENCH_NAMESPACE=token-factory"
set "VBENCH_POD=jirx-vbench-smoke-0930-a"
if not defined VBENCH_DELIVERY set "VBENCH_DELIVERY=D:\AI\_workspace\VBench_Image_20260930"
exit /b 0
