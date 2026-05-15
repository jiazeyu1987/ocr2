# ONLINE Timepoint Logging

## Goal
Add detailed ONLINE timepoint logs from receiving the ONLINE command through sending the JSON response back to the main program.

## Milestones
- [x] Locate ONLINE request handling and socket response send path.
- [x] Add absolute timepoint logs for each ONLINE step.
- [x] Verify unit tests and packaged server smoke behavior.

## Expected Verification
- ONLINE logs include a trace id and per-step `wall_time`.
- Logs record provider fetch, conversion, JSON encoding, and socket send timepoints.
- Existing API behavior remains unchanged.

## Status
Completed.

## Final Verification
- `D:\miniconda3\envs\py39\python.exe -m unittest test_api_server.py` passed with 20 tests.
- `D:\miniconda3\envs\py39\python.exe -m py_compile api_server.py test_api_server.py` passed.
- `.\package_pywrapper_server.bat` completed and deployed the updated `ocrapp_pureray.exe`.
- Smoke ONLINE request returned JSON and wrote `ONLINE timepoint` logs through `socket_send_completed`.
