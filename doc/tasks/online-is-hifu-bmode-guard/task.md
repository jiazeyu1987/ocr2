# Online IsHiFu B-Mode Guard

## Goal
Confirm that the main program under `D:\ProjectPackage\Vein\sqw\Vein\GLPyModule\JExtension` blocks treatment after an `ONLINE` response when the HiFu mode flag is `False`, and shows `B模式下不能运行`. If the current logic is different, change it to match.

## Milestones
- [x] Locate the current `ONLINE` response handling and treatment guard logic.
- [x] Add a regression test that proves the original condition does not match the requested behavior.
- [x] Change the treatment-start and continue-treatment guards to block on non-HiFu mode and unify the warning message.
- [x] Fix the follow-up runtime import failure caused by introducing a new helper file that was missing from the deployed Slicer module path.
- [x] Run verification and record evidence.

## Expected Verification
- Treatment start blocks when the `ONLINE` response reports `isHIFU == False`.
- Continue treatment blocks when the `ONLINE` response reports `isHIFU == False`.
- The warning text is `B模式下不能运行`.
- The module still imports when only `VeinTreat.py` and `ProgressButton.py` are present in the deployed tree.

## Status
Completed.

## Final Verification
- Code inspection confirmed the original guards in [ProgressButton.py](/D:/ProjectPackage/Vein/sqw/Vein/GLPyModule/JExtension/VeinTreat/TreatTools/ProgressButton.py:126) and [VeinTreat.py](/D:/ProjectPackage/Vein/sqw/Vein/GLPyModule/JExtension/VeinTreat/VeinTreat.py:3073) had been using the opposite condition before the first fix.
- The final implementation now blocks on `not bool(response["isHIFU"])` in both existing files, with no extra helper-file dependency.
- `python -m unittest D:\ocr2\test_online_mode_guard.py` passed with a deployment-style simulation that intentionally omits any extra helper file.
- `python -m py_compile D:\ProjectPackage\Vein\sqw\Vein\GLPyModule\JExtension\VeinTreat\TreatTools\ProgressButton.py` passed.
- `python -m py_compile D:\ProjectPackage\Vein\sqw\Vein\GLPyModule\JExtension\VeinTreat\VeinTreat.py` passed.
