# Execution Log

BDD: Debug mode saves OFFLINE ROI frames -> Given `offline_tmp_frames.enabled=true` and a valid focus point, When the same `point_id` sends OFFLINE start and stop, Then before/after ROI1, ROI2, ROI3 PNG files and `meta.json` are written under the configured debug root.

BDD: Debug mode fails fast on save errors -> Given debug saving is enabled, When the ROI image writer fails, Then OFFLINE returns `debug_save_failed` instead of silently continuing.

BDD: ROI3 is required for debug output -> Given ROI3 extension parameters place ROI3 outside the device frame, When OFFLINE starts, Then the response is `invalid_roi3_rect`.

BDD: Debug disabled leaves no output -> Given `offline_tmp_frames.enabled=false`, When OFFLINE completes, Then no debug directory is created and the response omits `debug_dir`.

RED: `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` -> FAIL, legacy green/red tests used the default ROI3 size on 20x20 frames and returned `invalid_roi3_rect`.

GREEN: `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` -> PASS, 19 tests.

GREEN: `D:\miniconda3\envs\py39\python.exe -m py_compile api_server.py test_api_server.py` -> PASS.

GREEN: `.\package_pywrapper_server.bat` -> PASS, created exe and zip artifacts.

GREEN: packaged exe smoke OFFLINE request -> PASS, returned `no_device_frame` with no screenshot fallback.
