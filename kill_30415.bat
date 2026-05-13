@echo off
setlocal enabledelayedexpansion

set "PORT=30415"
set "FOUND="

echo Looking for process listening on port %PORT%...

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    set "PID=%%P"
    set "FOUND=1"
    echo Killing PID !PID! on port %PORT%...
    taskkill /PID !PID! /F
    if errorlevel 1 (
        echo Failed to kill PID !PID!.
        exit /b 1
    )
)

if not defined FOUND (
    echo No LISTENING process found on port %PORT%.
    exit /b 0
)

echo Done.
exit /b 0
