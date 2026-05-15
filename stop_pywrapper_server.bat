@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
set "APP_NAMES=pywrapper_api_server.exe ocrapp_pureray.exe"

echo [INFO] Target exe names: %APP_NAMES%

set "FOUND=0"

for %%N in (%APP_NAMES%) do (
    for /f "skip=1 tokens=1 delims=" %%P in ('wmic process where "name='%%N'" get ProcessId 2^>nul') do (
        for /f "tokens=* delims= " %%Q in ("%%P") do (
            if not "%%Q"=="" (
                set "FOUND=1"
                echo [INFO] Stopping %%N PID=%%Q
                taskkill /PID %%Q /F
                if errorlevel 1 (
                    echo [ERROR] Failed to stop %%N PID=%%Q
                    exit /b 1
                )
            )
        )
    )
)

if "%FOUND%"=="0" (
    echo [INFO] No running target process found.
)

exit /b 0
