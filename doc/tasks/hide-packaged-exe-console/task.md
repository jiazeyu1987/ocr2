# Hide Packaged Exe Console

## Goal
Package the pywrapper server as a windowless executable so running `ocrapp_pureray.exe` does not show a cmd console window.

## Milestones
- [x] Locate PyInstaller console setting.
- [x] Switch packaged exe to windowed mode.
- [x] Rebuild and verify server still starts through the generated exe.

## Expected Verification
- `package_pywrapper_server.bat` uses PyInstaller windowed mode.
- Rebuilt `ocrapp_pureray.exe` starts and listens on `127.0.0.1:30415`.
- ONLINE request still returns JSON.

## Status
Completed.

## Final Verification
- `package_pywrapper_server.bat` now uses `--windowed`.
- PyInstaller build log used the `runw.exe` bootloader.
- `.\package_pywrapper_server.bat` completed successfully.
- Deployed `D:\ProjectPackage\Vein\sqw\Vein\OCRSERVER\ocrapp_pureray.exe` accepted `ONLINE;31415\n` on `127.0.0.1:30415` and returned JSON.
- `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` passed with 19 tests.
