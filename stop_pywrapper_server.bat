@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
set "APP_NAME=pywrapper_api_server.exe"
set "EXE_PATH=%ROOT_DIR%dist\pywrapper_api_server\%APP_NAME%"

echo [INFO] Target exe: %EXE_PATH%

if not exist "%EXE_PATH%" (
    echo [WARN] Target exe file does not exist: %EXE_PATH%
)

set "FOUND=0"

for /f "skip=1 tokens=1 delims=" %%P in ('wmic process where "name='%APP_NAME%'" get ProcessId 2^>nul') do (
    for /f "tokens=* delims= " %%Q in ("%%P") do (
        if not "%%Q"=="" (
            set "FOUND=1"
            echo [INFO] Stopping %APP_NAME% PID=%%Q
            taskkill /PID %%Q /F
            if errorlevel 1 (
                echo [ERROR] Failed to stop PID=%%Q
                exit /b 1
            )
        )
    )
)

if "%FOUND%"=="0" (
    echo [INFO] No running %APP_NAME% process found.
)

exit /b 0
