# Draw Line A Error Popup

## Goal
Change the draw-line ONLINE validation so an unrecognized `A` value triggers the error popup instead of the previous skin-depth recognition check.

## Milestones
- [x] Locate current draw-line recognition guard.
- [x] Replace skin-depth guard with `A` recognition guard.
- [x] Verify syntax and record evidence.

## Expected Verification
- `handle_draw_line()` rejects responses where `A` is missing, empty, invalid, non-finite, or `-1`.
- The rejection path closes the progress/error dialog and shows the existing "未识别到数据" popup.
- Python syntax check passes.

## Status
Completed.

## Final Verification
- `D:\miniconda3\envs\py39\python.exe -m py_compile GLPyModule\JExtension\VeinTreat\VeinTreat.py` passed in `D:\ProjectPackage\Vein\sqw\Vein`.
- `handle_draw_line()` now rejects unrecognized `A` before normalization.
