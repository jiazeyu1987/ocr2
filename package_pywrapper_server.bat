@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
set "SERVER_DIR=%ROOT_DIR%resource\pywrapper"
set "ENTRY=%SERVER_DIR%\api_server.py"
set "APP_NAME=pywrapper_api_server"
set "DIST_DIR=%ROOT_DIR%dist\%APP_NAME%"
set "ZIP_PATH=%ROOT_DIR%dist\%APP_NAME%.zip"

if not defined PYTHON_EXE (
    set "PYTHON_EXE=D:\miniconda3\envs\py39\python.exe"
)

echo [INFO] Root: %ROOT_DIR%
echo [INFO] Server: %SERVER_DIR%
echo [INFO] Python: %PYTHON_EXE%

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo [ERROR] Set PYTHON_EXE to the Python 3.9 environment that can import PyMobileComm.
    exit /b 1
)

if not exist "%ENTRY%" (
    echo [ERROR] Entry script not found: %ENTRY%
    exit /b 1
)

if not exist "%SERVER_DIR%\PyMobileComm.pyd" (
    echo [ERROR] Required PyMobileComm.pyd not found in %SERVER_DIR%
    exit /b 1
)

if not exist "%SERVER_DIR%\MobileCommunication.dll" (
    echo [ERROR] Required MobileCommunication.dll not found in %SERVER_DIR%
    exit /b 1
)

set "PYTHON_DIR="
for %%I in ("%PYTHON_EXE%") do set "PYTHON_DIR=%%~dpI"
set "CONDA_BIN=%PYTHON_DIR%Library\bin"
if not exist "%CONDA_BIN%\ffi.dll" (
    echo [ERROR] Required ffi.dll not found: %CONDA_BIN%\ffi.dll
    echo [ERROR] This DLL is required by Python ctypes in the packaged exe.
    exit /b 1
)

"%PYTHON_EXE%" -c "import PyInstaller" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] PyInstaller is not installed in %PYTHON_EXE%
    echo [ERROR] Install it in this environment first: "%PYTHON_EXE%" -m pip install pyinstaller
    exit /b 1
)

pushd "%ROOT_DIR%" || exit /b 1

"%PYTHON_EXE%" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --console ^
  --name "%APP_NAME%" ^
  --paths "%SERVER_DIR%" ^
  --distpath "%ROOT_DIR%dist" ^
  --workpath "%ROOT_DIR%build\%APP_NAME%" ^
  --specpath "%ROOT_DIR%build\%APP_NAME%" ^
  --hidden-import PyMobileComm ^
  --add-binary "%SERVER_DIR%\PyMobileComm.pyd;." ^
  --add-binary "%SERVER_DIR%\MobileCommunication.dll;." ^
  --add-binary "%SERVER_DIR%\AdbWinApi.dll;." ^
  --add-binary "%SERVER_DIR%\AdbWinUsbApi.dll;." ^
  --add-binary "%SERVER_DIR%\D3DX9_43.dll;." ^
  --add-binary "%SERVER_DIR%\DicomContol_Factory.dll;." ^
  --add-binary "%SERVER_DIR%\Ijwhost.dll;." ^
  --add-binary "%SERVER_DIR%\opencv_world440.dll;." ^
  --add-binary "%CONDA_BIN%\ffi.dll;." ^
  --add-binary "%CONDA_BIN%\libbz2.dll;." ^
  --add-binary "%CONDA_BIN%\libcrypto-3-x64.dll;." ^
  --add-binary "%CONDA_BIN%\liblzma.dll;." ^
  --add-data "%SERVER_DIR%\Company.ini;." ^
  --add-data "%SERVER_DIR%\license;." ^
  "%ENTRY%"

set "BUILD_RC=%ERRORLEVEL%"
popd

if not "%BUILD_RC%"=="0" (
    echo [ERROR] PyInstaller build failed with exit code %BUILD_RC%.
    exit /b %BUILD_RC%
)

if not exist "%DIST_DIR%\%APP_NAME%.exe" (
    echo [ERROR] Build finished but exe was not found: %DIST_DIR%\%APP_NAME%.exe
    exit /b 1
)

echo [OK] Server exe created:
echo      %DIST_DIR%\%APP_NAME%.exe

if exist "%ZIP_PATH%" (
    del /f /q "%ZIP_PATH%"
    if errorlevel 1 (
        echo [ERROR] Failed to remove old zip: %ZIP_PATH%
        exit /b 1
    )
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; Compress-Archive -LiteralPath '%DIST_DIR%' -DestinationPath '%ZIP_PATH%' -CompressionLevel Optimal"
if errorlevel 1 (
    echo [ERROR] Failed to create zip: %ZIP_PATH%
    exit /b 1
)

if not exist "%ZIP_PATH%" (
    echo [ERROR] Zip command finished but file was not found: %ZIP_PATH%
    exit /b 1
)

echo [OK] Zip created:
echo      %ZIP_PATH%
exit /b 0
