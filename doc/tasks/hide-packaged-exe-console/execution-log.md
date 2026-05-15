# Execution Log

BDD: No cmd window for packaged server -> Given the main program starts `OCRSERVER\ocrapp_pureray.exe`, When the exe runs, Then it should run without a visible console window and continue logging to `ocrlog`.

RED: Inspection -> FAIL, `package_pywrapper_server.bat` currently passes `--console` to PyInstaller.

GREEN: `rg -n -- "--console|--windowed|--noconsole" package_pywrapper_server.bat` -> PASS, script now uses `--windowed`.

GREEN: `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` -> PASS, 19 tests.

GREEN: `.\package_pywrapper_server.bat` -> PASS, PyInstaller used `runw.exe`.

GREEN: deployed windowed exe smoke test -> PASS, `ONLINE;31415\n` returned JSON on `127.0.0.1:30415`.
