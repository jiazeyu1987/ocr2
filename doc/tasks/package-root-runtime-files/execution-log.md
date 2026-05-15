# Execution Log

BDD: Runtime files beside exe -> Given the main program starts `OCRSERVER\ocrapp_pureray.exe`, When the package is built, Then `Company.ini`, `AdbWinApi.dll`, and `AdbWinUsbApi.dll` are present in the same directory as `ocrapp_pureray.exe`.

BDD: Original package also keeps root files -> Given the original package still outputs `pywrapper_api_server.exe`, When packaging completes, Then the same three files are present beside `pywrapper_api_server.exe`.

RED: Inspection -> FAIL, current PyInstaller packaging puts these files under `_internal`; they are not guaranteed to exist beside the exe.

GREEN: `.\package_pywrapper_server.bat` -> PASS, copied root runtime files beside both packaged executables and deployed main-program executable.

GREEN: root file inspection -> PASS, all three files exist in `dist\pywrapper_api_server`, `dist\OCRSERVER`, and deployed `D:\ProjectPackage\Vein\sqw\Vein\OCRSERVER`.

GREEN: `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` -> PASS, 19 tests.
