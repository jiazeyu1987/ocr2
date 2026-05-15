# Execution Log

BDD: Main program starts algorithm server by fixed name -> Given the main program calls `OCRSERVER\ocrapp_pureray.exe`, When the pywrapper server is packaged, Then a compatible `dist\OCRSERVER\ocrapp_pureray.exe` exists with the same runtime files.

BDD: Existing pywrapper package remains available -> Given existing tooling uses `dist\pywrapper_api_server\pywrapper_api_server.exe`, When packaging completes, Then that exe and zip still exist.

RED: Inspection -> FAIL, the current package only creates `dist\pywrapper_api_server\pywrapper_api_server.exe`; the main program fixed path `OCRSERVER\ocrapp_pureray.exe` is absent.

GREEN: `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` -> PASS, 19 tests.

GREEN: `.\package_pywrapper_server.bat` -> PASS, created original package, compatible `dist\OCRSERVER`, compatible zip, and deployed to main program `OCRSERVER`.

GREEN: deployed `ocrapp_pureray.exe` smoke test -> PASS, `ONLINE;31415\n` returned JSON on `127.0.0.1:30415`.
