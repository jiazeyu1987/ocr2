@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0package_backend.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Backend package failed with exit code %EXIT_CODE%.
  exit /b %EXIT_CODE%
)

echo.
echo Backend package finished successfully.
exit /b 0
