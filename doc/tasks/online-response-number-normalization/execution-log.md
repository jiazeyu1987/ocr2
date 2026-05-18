# Execution Log

BDD: Missing ONLINE values remain null -> Given provider fields are missing or `None`, When ONLINE response is converted, Then response fields remain `None` / JSON `null`.

BDD: Numeric ONLINE values are rounded -> Given numeric strings or numbers contain decimals, When ONLINE response is converted, Then decimal values are rounded half-up to two places and integral values remain integers.

BDD: Non-numeric ONLINE values are preserved -> Given provider fields contain non-numeric strings, empty strings, or non-finite numeric strings, When ONLINE response is converted, Then those fields remain unchanged instead of being replaced by `0`.

RED: Inspection -> FAIL, current ONLINE conversion returns provider strings directly.

GREEN: `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` -> PASS, 21 tests.

GREEN: `D:\miniconda3\envs\py39\python.exe -m py_compile api_server.py test_api_server.py` -> PASS.

RED: User correction -> FAIL, `None -> 0` is no longer desired; missing values must stay as `None`.

GREEN: corrected `None` behavior -> PASS, `None` remains `None` while numeric values are still normalized.

GREEN: `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` -> PASS, 21 tests after correction.

GREEN: `D:\miniconda3\envs\py39\python.exe -m py_compile api_server.py test_api_server.py` -> PASS after correction.

GREEN: `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` -> PASS, 22 tests after preserving non-numeric values.

GREEN: `D:\miniconda3\envs\py39\python.exe -m py_compile api_server.py test_api_server.py` -> PASS after preserving non-numeric values.

GREEN: `.\package_pywrapper_server.bat` -> PASS, packaged server rebuilt after preserving `None`.
