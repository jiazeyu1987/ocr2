# ONLINE Response Number Normalization

## Goal
Normalize ONLINE numeric response values so decimals are rounded half-up to two decimal places and integer values remain integers. Missing values stay as `None` / JSON `null`.

## Milestones
- [x] Locate ONLINE provider-to-response conversion.
- [x] Add strict ONLINE value normalization.
- [x] Update tests for numeric rounding while preserving `None`.
- [x] Verify and rebuild packaged server.

## Expected Verification
- Numeric strings and numeric values are rounded using normal half-up rounding.
- Integral values are returned as JSON integers.
- `None` values remain `None` / JSON `null`.
- Boolean fields remain booleans.
- Non-numeric non-null strings, such as `FocusPoint`, remain unchanged.

## Status
Completed.

## Verification Evidence
- `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` passed with 22 tests.
- `D:\miniconda3\envs\py39\python.exe -m py_compile api_server.py test_api_server.py` passed.
- `.\package_pywrapper_server.bat` completed and rebuilt `D:\ProjectPackage\Vein\sqw\Vein\OCRSERVER\ocrapp_pureray.exe`.
