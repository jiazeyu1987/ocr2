@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
set "SERVER_DIR=%ROOT_DIR%resource\pywrapper"
set "ENTRY=%SERVER_DIR%\api_server.py"
set "APP_NAME=pywrapper_api_server"
set "DIST_DIR=%ROOT_DIR%dist\%APP_NAME%"
set "ZIP_PATH=%ROOT_DIR%dist\%APP_NAME%.zip"
set "COMPAT_APP_NAME=ocrapp_pureray"
set "COMPAT_DIST_DIR=%ROOT_DIR%dist\OCRSERVER"
set "COMPAT_ZIP_PATH=%ROOT_DIR%dist\OCRSERVER.zip"

if not defined VEIN_MAIN_DIR (
    set "VEIN_MAIN_DIR=D:\ProjectPackage\Vein\sqw\Vein"
)
if not defined DEPLOY_TO_VEIN_MAIN (
    set "DEPLOY_TO_VEIN_MAIN=1"
)
set "VEIN_OCRSERVER_DIR=%VEIN_MAIN_DIR%\OCRSERVER"

if not defined PYTHON_EXE (
    set "PYTHON_EXE=D:\miniconda3\envs\py39\python.exe"
)

echo [INFO] Root: %ROOT_DIR%
echo [INFO] Server: %SERVER_DIR%
echo [INFO] Python: %PYTHON_EXE%
echo [INFO] Main program dir: %VEIN_MAIN_DIR%
echo [INFO] Deploy to main program: %DEPLOY_TO_VEIN_MAIN%

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
  --windowed ^
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
  --add-binary "%CONDA_BIN%\libexpat.dll;." ^
  --add-binary "%CONDA_BIN%\liblzma.dll;." ^
  --add-binary "%CONDA_BIN%\libssl-3-x64.dll;." ^
  --add-data "%SERVER_DIR%\Company.ini;." ^
  --add-data "%SERVER_DIR%\license;." ^
  --add-data "%ROOT_DIR%settings;." ^
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

for %%F in (Company.ini AdbWinApi.dll AdbWinUsbApi.dll) do (
    if not exist "%SERVER_DIR%\%%F" (
        echo [ERROR] Required root runtime file not found: %SERVER_DIR%\%%F
        exit /b 1
    )
    copy /Y "%SERVER_DIR%\%%F" "%DIST_DIR%\%%F" >nul
    if errorlevel 1 (
        echo [ERROR] Failed to copy root runtime file: %%F
        exit /b 1
    )
    if not exist "%DIST_DIR%\%%F" (
        echo [ERROR] Root runtime file copy finished but file was not found: %DIST_DIR%\%%F
        exit /b 1
    )
)

echo [OK] Server exe created:
echo      %DIST_DIR%\%APP_NAME%.exe
echo [OK] Root runtime files copied beside server exe:
echo      Company.ini
echo      AdbWinApi.dll
echo      AdbWinUsbApi.dll

if exist "%COMPAT_DIST_DIR%" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop'; Remove-Item -LiteralPath '%COMPAT_DIST_DIR%' -Recurse -Force"
    if errorlevel 1 (
        echo [ERROR] Failed to remove old compatible package directory: %COMPAT_DIST_DIR%
        exit /b 1
    )
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; Copy-Item -LiteralPath '%DIST_DIR%' -Destination '%COMPAT_DIST_DIR%' -Recurse; Rename-Item -LiteralPath '%COMPAT_DIST_DIR%\%APP_NAME%.exe' -NewName '%COMPAT_APP_NAME%.exe'"
if errorlevel 1 (
    echo [ERROR] Failed to create main-program compatible package: %COMPAT_DIST_DIR%
    exit /b 1
)

if not exist "%COMPAT_DIST_DIR%\%COMPAT_APP_NAME%.exe" (
    echo [ERROR] Compatible exe was not found: %COMPAT_DIST_DIR%\%COMPAT_APP_NAME%.exe
    exit /b 1
)

echo [OK] Main-program compatible server exe created:
echo      %COMPAT_DIST_DIR%\%COMPAT_APP_NAME%.exe

if "%DEPLOY_TO_VEIN_MAIN%"=="1" (
    if not exist "%VEIN_MAIN_DIR%" (
        echo [ERROR] Main program directory not found: %VEIN_MAIN_DIR%
        exit /b 1
    )
    if exist "%VEIN_OCRSERVER_DIR%" (
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
          "$ErrorActionPreference='Stop'; Remove-Item -LiteralPath '%VEIN_OCRSERVER_DIR%' -Recurse -Force"
        if errorlevel 1 (
            echo [ERROR] Failed to remove old main-program OCRSERVER directory: %VEIN_OCRSERVER_DIR%
            echo [ERROR] Close running ocrapp_pureray.exe and retry.
            exit /b 1
        )
    )
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop'; Copy-Item -LiteralPath '%COMPAT_DIST_DIR%' -Destination '%VEIN_OCRSERVER_DIR%' -Recurse"
    if errorlevel 1 (
        echo [ERROR] Failed to deploy compatible server to: %VEIN_OCRSERVER_DIR%
        exit /b 1
    )
    if not exist "%VEIN_OCRSERVER_DIR%\%COMPAT_APP_NAME%.exe" (
        echo [ERROR] Main-program deployment finished but exe was not found: %VEIN_OCRSERVER_DIR%\%COMPAT_APP_NAME%.exe
        exit /b 1
    )
    echo [OK] Main-program OCRSERVER deployed:
    echo      %VEIN_OCRSERVER_DIR%\%COMPAT_APP_NAME%.exe
)

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

if exist "%COMPAT_ZIP_PATH%" (
    del /f /q "%COMPAT_ZIP_PATH%"
    if errorlevel 1 (
        echo [ERROR] Failed to remove old compatible zip: %COMPAT_ZIP_PATH%
        exit /b 1
    )
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; Compress-Archive -LiteralPath '%COMPAT_DIST_DIR%' -DestinationPath '%COMPAT_ZIP_PATH%' -CompressionLevel Optimal"
if errorlevel 1 (
    echo [ERROR] Failed to create compatible zip: %COMPAT_ZIP_PATH%
    exit /b 1
)

if not exist "%COMPAT_ZIP_PATH%" (
    echo [ERROR] Compatible zip command finished but file was not found: %COMPAT_ZIP_PATH%
    exit /b 1
)

echo [OK] Main-program compatible zip created:
echo      %COMPAT_ZIP_PATH%
exit /b 0
