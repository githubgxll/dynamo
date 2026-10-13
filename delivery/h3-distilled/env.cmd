@echo off
set "H3_DISTILLED_PY=D:\AI\_workspace\H3_Omni030_Image_20260929\tools-venv\Scripts\python.exe"
set "H3_DISTILLED_EVIDENCE=D:\AI\_workspace\H3_Distilled_Image_20261013"
if not exist "%H3_DISTILLED_PY%" (
  echo Missing existing preparation Python: %H3_DISTILLED_PY%
  exit /b 1
)
exit /b 0
