# Bug Regression Evidence

## Bug Summary And Expected Behavior
- Bug: after clicking continuous treatment, the flow could receive an `ONLINE` response where `Alpha` was `None`, then silently die in path-planning math, making the downstream treatment button appear unresponsive.
- Expected: if `Alpha` is missing or invalid, the flow must fail fast with a clear warning and restore the continue-treatment button state.

## Reproduction Command Or Path
- Runtime path: [handle_ultra_freeze](/D:/ProjectPackage/Vein/sqw/Vein/GLPyModule/JExtension/VeinTreat/VeinTreat.py:3180) stored `response['Alpha']`, then [generate_planned_poses_list](/D:/ProjectPackage/Vein/sqw/Vein/GLPyModule/JExtension/VeinTreat/VeinTreat.py:3320) and [_on_btnContinueCure](/D:/ProjectPackage/Vein/sqw/Vein/GLPyModule/JExtension/VeinTreat/VeinTreat.py:3272) used `math.radians(angle_deg)`.
- Runtime evidence showed `Alpha=None` immediately before `must be real number, not NoneType`.
- RED command: `python -m unittest D:\ocr2\test_online_mode_guard.py`

## Root Cause
- Continuous-treatment planning assumed `ONLINE` always returned a usable numeric `Alpha`.
- The code did not validate `Alpha` before entering path-planning math, so `math.radians(None)` raised and aborted the callback.
- Because the exception happened before queue preparation and before the treatment runtime button could become actionable, the user experience looked like “点击没反应”.

## Regression Test Added Or Updated
- Updated [test_online_mode_guard.py](/D:/ocr2/test_online_mode_guard.py:185) to require:
- `get_continue_treat_planning_error({"A": 1.5, "B": 1.5, "Alpha": None}) == "ONLINE返回缺少Alpha，无法规划连续治疗路径"`
- `get_continue_treat_planning_error({"A": 1.5, "B": 1.5, "Alpha": 12.0}) is None`

## RED Command And Expected Failure
RED:
- `python -m unittest D:\ocr2\test_online_mode_guard.py`
- Failure reason before the fix: `AttributeError` because `VeinTreatUnderTest` did not expose a planning guard for `Alpha=None`.

## GREEN Command And Passing Result
GREEN:
- `python -m unittest D:\ocr2\test_online_mode_guard.py`
- `python -m py_compile D:\ProjectPackage\Vein\sqw\Vein\GLPyModule\JExtension\VeinTreat\VeinTreat.py`
- `python -m py_compile D:\ProjectPackage\Vein\sqw\Vein\GLPyModule\JExtension\VeinTreat\TreatTools\ProgressButton.py`

## Risk And Regression Scope
- Scope is limited to continuous-treatment planning preconditions after `ONLINE` success.
- The change intentionally fails fast on invalid `Alpha` instead of guessing a fallback angle.

## Verification
- The flow now blocks before path-planning math when `Alpha` is missing or invalid.
- The user-facing warning is explicit.
- The continue-treatment button state is restored instead of leaving the UI in a confusing half-started state.

## Blockers And Follow-Up Actions
- No source-code blockers remain.
- The running `D:\v3_sqw\...` copy still needs the updated files synced into it before the fix is visible in that deployed environment.
