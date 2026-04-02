@echo off
setlocal

rem Copy all .py and .pyd files from this repo to dist\ocrapp_pureray\_internal
set "SRC=%~dp0"
set "DST=%SRC%dist\ocrapp_pureray\_internal"

if not exist "%DST%" (
  echo Target not found: %DST%
  exit /b 1
)

for /r "%SRC%" %%F in (*.py *.pyd) do (
  copy /y "%%F" "%DST%" >nul
)

echo Done.
endlocal
