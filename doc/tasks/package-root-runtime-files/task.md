# Package Root Runtime Files

## Goal
Ensure `Company.ini`, `AdbWinApi.dll`, and `AdbWinUsbApi.dll` are packaged next to the generated executable as well as inside PyInstaller internals.

## Milestones
- [x] Identify requested root-level runtime files.
- [x] Copy root runtime files into original and main-program-compatible package roots.
- [x] Verify package output includes the files next to both executables.

## Expected Verification
- `D:\ocr2\dist\pywrapper_api_server\Company.ini` exists.
- `D:\ocr2\dist\pywrapper_api_server\AdbWinApi.dll` exists.
- `D:\ocr2\dist\pywrapper_api_server\AdbWinUsbApi.dll` exists.
- `D:\ocr2\dist\OCRSERVER\Company.ini` exists.
- `D:\ocr2\dist\OCRSERVER\AdbWinApi.dll` exists.
- `D:\ocr2\dist\OCRSERVER\AdbWinUsbApi.dll` exists.
- Deployed main-program `OCRSERVER` contains the same three root-level files.

## Status
Completed.

## Final Verification
- `.\package_pywrapper_server.bat` completed successfully.
- `D:\ocr2\dist\pywrapper_api_server\Company.ini`, `AdbWinApi.dll`, and `AdbWinUsbApi.dll` exist beside `pywrapper_api_server.exe`.
- `D:\ocr2\dist\OCRSERVER\Company.ini`, `AdbWinApi.dll`, and `AdbWinUsbApi.dll` exist beside `ocrapp_pureray.exe`.
- `D:\ProjectPackage\Vein\sqw\Vein\OCRSERVER\Company.ini`, `AdbWinApi.dll`, and `AdbWinUsbApi.dll` exist beside deployed `ocrapp_pureray.exe`.
- `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` passed with 19 tests.
