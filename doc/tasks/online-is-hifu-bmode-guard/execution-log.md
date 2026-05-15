# Execution Log

BDD: ONLINE returns non-HiFu mode -> Given the treatment flow receives an `ONLINE` response whose `isHIFU` flag is `False`, When the treatment-start or continue-treatment guard evaluates the response, Then it must block the action and show `B模式下不能运行`.

RED: `python -m unittest D:\ocr2\test_online_mode_guard.py` -> FAIL, a deployment-style import of `ProgressButton.py` without an extra helper file raised `ModuleNotFoundError: No module named 'TreatTools.online_mode_guard'`.

GREEN: `python -m unittest D:\ocr2\test_online_mode_guard.py` -> PASS

GREEN: `python -m py_compile D:\ProjectPackage\Vein\sqw\Vein\GLPyModule\JExtension\VeinTreat\TreatTools\ProgressButton.py` -> PASS

GREEN: `python -m py_compile D:\ProjectPackage\Vein\sqw\Vein\GLPyModule\JExtension\VeinTreat\VeinTreat.py` -> PASS
