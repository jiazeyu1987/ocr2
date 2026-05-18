# Continue Treat Alpha None Guard

## Goal
Fix the continuous-treatment flow so clicking the treatment path does not appear unresponsive when the `ONLINE` response lacks `Alpha`. The flow must fail fast with a clear user-facing warning instead of crashing during path planning.

## Milestones
- [x] Confirm the exact failure point from current logs and source.
- [x] Add a regression test that proves `Alpha=None` is rejected before path planning math runs.
- [x] Implement a fail-fast guard with a clear warning and button-state recovery.
- [x] Run verification and record evidence.

## Expected Verification
- Continuous-treatment planning rejects `ONLINE` responses with missing `Alpha`.
- The user sees a clear warning instead of silent no-op behavior.
- `btnContinueCure` is restored to an actionable state after the rejection.
- Regression test fails before the fix and passes after it.

## Status
Completed.

## Final Verification
- Added [get_continue_treat_planning_error](/D:/ProjectPackage/Vein/sqw/Vein/GLPyModule/JExtension/VeinTreat/VeinTreat.py:45) to reject missing or invalid `Alpha` before any path-planning math runs.
- Updated [handle_ultra_freeze](/D:/ProjectPackage/Vein/sqw/Vein/GLPyModule/JExtension/VeinTreat/VeinTreat.py:3216) to restore `btnContinueCure`, keep the planning widget hidden, log the block reason, and show `ONLINE返回缺少Alpha，无法规划连续治疗路径`.
- `python -m unittest D:\ocr2\test_online_mode_guard.py` passed.
- `python -m py_compile` passed for [VeinTreat.py](/D:/ProjectPackage/Vein/sqw/Vein/GLPyModule/JExtension/VeinTreat/VeinTreat.py) and [ProgressButton.py](/D:/ProjectPackage/Vein/sqw/Vein/GLPyModule/JExtension/VeinTreat/TreatTools/ProgressButton.py).
