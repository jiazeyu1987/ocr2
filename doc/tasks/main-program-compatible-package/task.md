# Main Program Compatible Package

## Goal
Make the packaged pywrapper algorithm server usable by the existing Vein main program without changing main program code.

## Milestones
- [x] Identify the main program startup contract.
- [x] Package a main-program-compatible `OCRSERVER\ocrapp_pureray.exe` layout.
- [x] Verify the normal package and compatible alias both start on `127.0.0.1:30415`.
- [x] Record final verification evidence.

## Expected Verification
- `package_pywrapper_server.bat` creates the existing `dist\pywrapper_api_server\pywrapper_api_server.exe`.
- `package_pywrapper_server.bat` also creates `dist\OCRSERVER\ocrapp_pureray.exe`.
- The compatible exe starts and responds to a TCP request on the algorithm server port.

## Status
Completed.

## Final Verification
- `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` passed with 19 tests.
- `.\package_pywrapper_server.bat` created `D:\ocr2\dist\pywrapper_api_server\pywrapper_api_server.exe`.
- `.\package_pywrapper_server.bat` created `D:\ocr2\dist\OCRSERVER\ocrapp_pureray.exe`.
- `.\package_pywrapper_server.bat` deployed `D:\ProjectPackage\Vein\sqw\Vein\OCRSERVER\ocrapp_pureray.exe`.
- Smoke test against deployed `ocrapp_pureray.exe` on `127.0.0.1:30415` returned an ONLINE JSON response.
