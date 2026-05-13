# OFFLINE ROI Debug Save

## Goal
Add OFFLINE debug image output for ROI1, ROI2, and ROI3 using device frames from `image_matrix`, controlled by `settings.offline_tmp_frames`.

## Milestones
- [x] Read ROI3 and debug-save settings from configuration.
- [x] Compute ROI1, ROI2, and ROI3 from the device frame and focus anchor.
- [x] Save before/after ROI images and `meta.json` when debug mode is enabled.
- [x] Fail fast with `debug_save_failed` if debug output cannot be written.
- [x] Run unit, compile, package, and smoke verification.

## Expected Verification
- Unit tests cover config parsing, debug enabled/disabled behavior, ROI3 bounds failure, and debug-save failure.
- `api_server.py` and tests compile.
- Server package builds into `dist\pywrapper_api_server` and zip artifact.
- Packaged exe starts and OFFLINE without a device frame reports `no_device_frame`.

## Status
Completed.

## Final Verification
- `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` passed with 19 tests.
- `D:\miniconda3\envs\py39\python.exe -m py_compile api_server.py test_api_server.py` passed.
- `.\package_pywrapper_server.bat` created `D:\ocr2\dist\pywrapper_api_server\pywrapper_api_server.exe` and `D:\ocr2\dist\pywrapper_api_server.zip`.
- Packaged exe smoke test returned `{"success": false, "info": "no_device_frame", "point_id": 123}` for OFFLINE without a device frame.
