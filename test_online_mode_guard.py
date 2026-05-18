import importlib.util
import pathlib
import shutil
import sys
import tempfile
import types
import unittest


SOURCE_ROOT = pathlib.Path(
    r"D:\ProjectPackage\Vein\sqw\Vein\GLPyModule\JExtension\VeinTreat"
)
SOURCE_PROGRESS_BUTTON = SOURCE_ROOT / "TreatTools" / "ProgressButton.py"
SOURCE_VEIN_TREAT = SOURCE_ROOT / "VeinTreat.py"


class DummyQtObject:
    def __init__(self, *args, **kwargs):
        pass


class DummyScriptedLoadableModule:
    def __init__(self, *args, **kwargs):
        pass


class DummyScriptedLoadableModuleWidget:
    def __init__(self, *args, **kwargs):
        pass


class DummyVTKObservationMixin:
    def __init__(self, *args, **kwargs):
        pass


def _build_qt_module():
    qt = types.ModuleType("qt")
    qt.QPushButton = DummyQtObject
    qt.QWidget = DummyQtObject
    qt.QBrush = DummyQtObject
    qt.QColor = DummyQtObject
    qt.QPen = DummyQtObject
    qt.QTimer = DummyQtObject
    qt.QPixmap = DummyQtObject
    qt.QIcon = DummyQtObject
    qt.QSize = DummyQtObject
    qt.QLabel = DummyQtObject
    qt.QRect = DummyQtObject
    qt.QPainter = DummyQtObject
    qt.QPointF = DummyQtObject
    qt.Qt = types.SimpleNamespace(
        NoPen=0,
        white=0,
        LeftButton=1,
        AlignCenter=0,
        FramelessWindowHint=0,
        WA_DeleteOnClose=0,
    )
    qt.__getattr__ = lambda name: DummyQtObject
    return qt


def _install_stub_modules(temp_root):
    inserted = {}

    def add_module(name, module):
        inserted[name] = sys.modules.get(name)
        sys.modules[name] = module

    qt = _build_qt_module()
    add_module("qt", qt)

    cv2 = types.ModuleType("cv2")
    add_module("cv2", cv2)

    controls = types.ModuleType("Controls")
    controls.__path__ = [str(temp_root / "Controls")]
    add_module("Controls", controls)

    control_view = types.ModuleType("Controls.ControlView")
    control_view.ControlView = DummyQtObject
    add_module("Controls.ControlView", control_view)

    therapy_view = types.ModuleType("Controls.TherapyView")
    therapy_view.TherapyView = DummyQtObject
    add_module("Controls.TherapyView", therapy_view)

    slicer = types.ModuleType("slicer")
    util = types.ModuleType("slicer.util")
    util.VTKObservationMixin = DummyVTKObservationMixin
    util.getModuleWidget = lambda name: None
    slicer.util = util
    add_module("slicer", slicer)
    add_module("slicer.util", util)

    scripted = types.ModuleType("slicer.ScriptedLoadableModule")
    scripted.ScriptedLoadableModule = DummyScriptedLoadableModule
    scripted.ScriptedLoadableModuleWidget = DummyScriptedLoadableModuleWidget
    scripted.__all__ = [
        "ScriptedLoadableModule",
        "ScriptedLoadableModuleWidget",
    ]
    add_module("slicer.ScriptedLoadableModule", scripted)

    treat_tools = types.ModuleType("TreatTools")
    treat_tools.__path__ = [str(temp_root / "TreatTools")]
    add_module("TreatTools", treat_tools)

    return inserted


def _restore_stub_modules(inserted):
    for name, original in inserted.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class OnlineModeGuardTestCase(unittest.TestCase):
    def _build_temp_tree(self):
        temp_dir = tempfile.TemporaryDirectory()
        temp_root = pathlib.Path(temp_dir.name)
        (temp_root / "TreatTools").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_PROGRESS_BUTTON, temp_root / "TreatTools" / "ProgressButton.py")
        shutil.copyfile(SOURCE_VEIN_TREAT, temp_root / "VeinTreat.py")
        return temp_dir, temp_root

    def test_progress_button_imports_without_extra_helper_file(self):
        temp_dir, temp_root = self._build_temp_tree()
        inserted = _install_stub_modules(temp_root)
        try:
            module = _load_module(
                "TreatTools.ProgressButton",
                temp_root / "TreatTools" / "ProgressButton.py",
            )
            self.assertEqual(module.B_MODE_BLOCK_MESSAGE, "B模式下不能运行")
            self.assertEqual(module.FREEZE_BLOCK_MESSAGE, "超声设备冻结中，请激活设备。")
            self.assertTrue(module.should_block_on_non_hifu_mode({"isHIFU": False}))
            self.assertFalse(module.should_block_on_non_hifu_mode({"isHIFU": True}))
            self.assertTrue(module.should_request_online_before_treat(0))
            self.assertTrue(module.should_request_online_before_treat(1))
            self.assertFalse(module.should_request_online_before_treat(2))
            self.assertTrue(module.should_check_freeze_in_online_response(0))
            self.assertTrue(module.should_check_freeze_in_online_response(1))
            self.assertFalse(module.should_check_freeze_in_online_response(2))
            self.assertEqual(
                module.get_treatment_online_block_message(
                    {"isHIFU": False, "IsFreeze": True},
                    freeze_check_enabled=True,
                ),
                "B模式下不能运行",
            )
            self.assertEqual(
                module.get_treatment_online_block_message(
                    {"isHIFU": True, "IsFreeze": True},
                    freeze_check_enabled=True,
                ),
                "超声设备冻结中，请激活设备。",
            )
            self.assertIsNone(
                module.get_treatment_online_block_message(
                    {"isHIFU": True, "IsFreeze": True},
                    freeze_check_enabled=False,
                )
            )
            progress_button_source = SOURCE_PROGRESS_BUTTON.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn("super().mousePressEvent(event)", progress_button_source)
            self.assertNotIn("super().mouseReleaseEvent(event)", progress_button_source)
        finally:
            _restore_stub_modules(inserted)
            temp_dir.cleanup()

    def test_vein_treat_imports_without_extra_helper_file(self):
        temp_dir, temp_root = self._build_temp_tree()
        inserted = _install_stub_modules(temp_root)
        try:
            _load_module(
                "TreatTools.ProgressButton",
                temp_root / "TreatTools" / "ProgressButton.py",
            )
            module = _load_module("VeinTreatUnderTest", temp_root / "VeinTreat.py")
            self.assertEqual(module.B_MODE_BLOCK_MESSAGE, "B模式下不能运行")
            self.assertEqual(
                module.get_treatment_online_block_message(
                    {"isHIFU": False, "IsFreeze": True},
                    freeze_check_enabled=True,
                ),
                "B模式下不能运行",
            )
            self.assertEqual(
                module.get_continue_treat_planning_error(
                    {"A": 1.5, "B": 1.5, "Alpha": None}
                ),
                "ONLINE返回缺少Alpha，无法规划连续治疗路径",
            )
            self.assertIsNone(
                module.get_continue_treat_planning_error(
                    {"A": 1.5, "B": 1.5, "Alpha": 12.0}
                )
            )
            self.assertEqual(
                module.build_continue_treat_debug_snapshot(
                    btn_more_enabled=True,
                    btn_more_visible=False,
                    btn_more_down=False,
                    widget_more_visible=True,
                    overlay_visible=True,
                    overlay_enabled=True,
                    continue_enabled=False,
                    continue_visible=False,
                    queue_pointer=3,
                    queue_length=9,
                    timer_active=True,
                    util_more_treat=True,
                    planned_pose_count=6,
                ),
                {
                    "btn_more_enabled": True,
                    "btn_more_visible": False,
                    "btn_more_down": False,
                    "widget_more_visible": True,
                    "overlay_visible": True,
                    "overlay_enabled": True,
                    "continue_enabled": False,
                    "continue_visible": False,
                    "queue_pointer": 3,
                    "queue_length": 9,
                    "timer_active": True,
                    "util_more_treat": True,
                    "planned_pose_count": 6,
                },
            )
        finally:
            _restore_stub_modules(inserted)
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
