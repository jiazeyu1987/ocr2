# Bug Regression Evidence

## Bug Summary And Expected Behavior
- Bug: the requested non-HiFu blocking behavior was initially implemented through a new helper file, but the deployed Slicer runtime did not have that file, so importing `VeinTreat` failed before the module could instantiate.
- Expected: treatment-related `ONLINE` guards should block when `isHIFU == False`, show `B模式下不能运行`, and keep loading successfully in the deployed runtime.

## Reproduction Command Or Path
- Runtime trace from the deployed application showed: `ModuleNotFoundError: No module named 'TreatTools.online_mode_guard'` while importing [ProgressButton.py](/D:/ProjectPackage/Vein/sqw/Vein/GLPyModule/JExtension/VeinTreat/TreatTools/ProgressButton.py:5) from [VeinTreat.py](/D:/ProjectPackage/Vein/sqw/Vein/GLPyModule/JExtension/VeinTreat/VeinTreat.py:12).
- RED command: `python -m unittest D:\ocr2\test_online_mode_guard.py`

## Root Cause
- The first fix introduced a new production dependency file, `online_mode_guard.py`.
- The deployed Slicer module tree did not include that file, so Python failed during module import and `VeinTreat` never instantiated.
- Because `VeinTreat` failed to instantiate, downstream code kept reporting that the module was registered but not instantiated.

## Regression Test Added Or Updated
- Updated [test_online_mode_guard.py](/D:/ocr2/test_online_mode_guard.py:1) to simulate a deployed module tree that includes only `VeinTreat.py` and `TreatTools/ProgressButton.py`, deliberately omitting any extra helper file.
- The test imports both modules under stubbed Qt/Slicer dependencies and asserts:
- `B_MODE_BLOCK_MESSAGE == "B模式下不能运行"`.
- `should_block_on_non_hifu_mode({"isHIFU": False})` is `True`.
- `should_block_on_non_hifu_mode({"isHIFU": True})` is `False`.

## RED Command And Expected Failure
RED:
- `python -m unittest D:\ocr2\test_online_mode_guard.py`
- Expected failure before the final fix: `ModuleNotFoundError` for `TreatTools.online_mode_guard`.

## GREEN Command And Passing Result
GREEN:
- `python -m unittest D:\ocr2\test_online_mode_guard.py`
- `python -m py_compile D:\ProjectPackage\Vein\sqw\Vein\GLPyModule\JExtension\VeinTreat\TreatTools\ProgressButton.py`
- `python -m py_compile D:\ProjectPackage\Vein\sqw\Vein\GLPyModule\JExtension\VeinTreat\VeinTreat.py`

## Risk And Regression Scope
- Scope is limited to treatment guard logic and the import path needed for `VeinTreat` module instantiation.
- The logic still fails fast if `response["isHIFU"]` is missing, which matches the no-fallback policy.

## Verification
- Both guards now keep their logic inside existing deployed files instead of relying on an extra helper module file.
- The warning text is `B模式下不能运行`.
- Deployment-style import simulation and syntax checks passed.

## Blockers And Follow-Up Actions
- No code blockers remain in the source tree.
- I could not patch `D:\v3_sqw\...` directly from this workspace because that path is not present in the current filesystem view, so that runtime copy still needs the updated existing files synced into it before relaunch.
