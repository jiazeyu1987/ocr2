# -*- coding: utf-8 -*-
import argparse
import ctypes
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image


PASSWORD = "31415"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30415
PROVIDER_FIELDS = (
    "focus_depth",
    "guankuan_a",
    "guankuan_b",
    "depth",
    "focus_point",
    "isLive",
    "mode",
)


class StateInfo(ctypes.Structure):
    _fields_ = [
        ("Version", ctypes.c_int),
        ("AdbServer", ctypes.c_int),
        ("LicenseType", ctypes.c_int),
        ("ControlLinkState", ctypes.c_int),
        ("ImageInfoLinkState", ctypes.c_int),
        ("USBLinkState", ctypes.c_int),
        ("AppRunState", ctypes.c_int),
    ]


@dataclass(frozen=True)
class FrameSnapshot:
    image: np.ndarray
    seq: int
    ts: float


@dataclass(frozen=True)
class OfflineConfig:
    roi2_extension_params: dict = field(default_factory=lambda: {"left": 40, "right": 40, "top": 50, "bottom": 30})
    roi3_extension_params: dict = field(default_factory=lambda: {"left": 30, "right": 30, "top": 50, "bottom": 100})
    difference_threshold: float = 0.5
    debug_save_enabled: bool = False
    debug_save_dir: str = "D:/software_data/tmp"

    @staticmethod
    def default() -> "OfflineConfig":
        return OfflineConfig()


@dataclass
class OfflineSession:
    point_id: object
    before: np.ndarray
    before_seq: int
    before_ts: float
    focus_anchor: Tuple[int, int]
    roi2_rect: Tuple[int, int, int, int]
    roi3_rect: Tuple[int, int, int, int]
    before_mean: float
    debug_dir: Optional[str] = None
    meta: dict = field(default_factory=dict)


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint),
        ("wParam", ctypes.c_void_p),
        ("lParam", ctypes.c_void_p),
        ("time", ctypes.c_ulong),
        ("pt_x", ctypes.c_long),
        ("pt_y", ctypes.c_long),
    ]


@dataclass(frozen=True)
class ParsedRequest:
    req_type: str
    param: str
    arg: Optional[str]


def configure_runtime_paths() -> None:
    base_dir = Path(__file__).resolve().parent
    os.add_dll_directory(str(base_dir))

    env_prefix = Path(sys.executable).resolve().parent.parent
    env_bin = env_prefix / "Library" / "bin"
    if env_bin.exists():
        os.add_dll_directory(str(env_bin))
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        env_bin_text = str(env_bin)
        if env_bin_text not in path_parts:
            os.environ["PATH"] = env_bin_text + os.pathsep + os.environ.get("PATH", "")


def log_process_environment(logger: logging.Logger) -> None:
    base_dir = Path(__file__).resolve().parent
    exe_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else base_dir
    logger.info("process frozen: %s", bool(getattr(sys, "frozen", False)))
    logger.info("sys.executable: %s", sys.executable)
    logger.info("current working directory: %s", os.getcwd())
    logger.info("module directory: %s", base_dir)
    logger.info("exe directory: %s", exe_dir)
    for name in (
        "PyMobileComm.pyd",
        "MobileCommunication.dll",
        "DicomContol_Factory.dll",
        "AdbWinApi.dll",
        "AdbWinUsbApi.dll",
        "Company.ini",
        "license",
    ):
        path = base_dir / name
        logger.info("required file %s exists=%s path=%s", name, path.exists(), path)


def log_adb_devices(logger: logging.Logger) -> None:
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        logger.exception("adb devices failed: %s", exc)
        return

    logger.info("adb devices exit_code=%s", result.returncode)
    logger.info("adb devices stdout: %s", result.stdout.strip().replace("\n", " | "))
    stderr = result.stderr.strip()
    if stderr:
        logger.warning("adb devices stderr: %s", stderr.replace("\n", " | "))


def import_mobile_comm():
    configure_runtime_paths()
    import PyMobileComm

    return PyMobileComm


def parse_focus_point(value) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(float(value[0])), int(float(value[1]))
        except Exception:
            return None

    text = str(value).strip()
    match = re.search(r"PointF\(\s*([^,\s]+)\s*,\s*([^)]+?)\s*\)", text)
    if not match:
        return None
    try:
        return int(float(match.group(1))), int(float(match.group(2)))
    except Exception:
        return None


def compute_roi_region(
    frame_size: Tuple[int, int],
    anchor: Tuple[int, int],
    extension_params: dict,
) -> Optional[Tuple[int, int, int, int]]:
    width, height = frame_size
    ax, ay = anchor
    try:
        left = int(extension_params["left"])
        right = int(extension_params["right"])
        top = int(extension_params["top"])
        bottom = int(extension_params["bottom"])
    except Exception:
        return None

    x1 = int(ax) - left
    y1 = int(ay) - top
    x2 = int(ax) + right
    y2 = int(ay) + bottom
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def roi_gray_mean(image: np.ndarray, rect: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = rect
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("empty ROI")
    if roi.ndim == 3:
        roi = roi.astype(np.float32)
        gray = 0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 2]
        return float(np.mean(gray))
    return float(np.mean(roi.astype(np.float32)))


def crop_rect(image: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError(f"empty crop for rect={rect}")
    return np.array(cropped, copy=True)


def write_png(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image)
    if arr.ndim == 2:
        pil_image = Image.fromarray(arr)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pil_image = Image.fromarray(arr.astype(np.uint8))
    elif arr.ndim == 3 and arr.shape[2] == 4:
        pil_image = Image.fromarray(arr.astype(np.uint8))
    else:
        raise ValueError(f"unsupported image shape for png: {arr.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path, format="PNG")


class DebugFrameSaver:
    def create_session_dir(self, root_dir: str, point_id) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        day = datetime.now().strftime("%Y%m%d")
        safe_point = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(point_id))
        path = Path(root_dir) / "pywrapper_offline" / day / f"{safe_point}_{ts}"
        path.mkdir(parents=True, exist_ok=False)
        return str(path)

    def save_stage(
        self,
        debug_dir: str,
        stage: str,
        frame: np.ndarray,
        roi2_rect: Tuple[int, int, int, int],
        roi3_rect: Tuple[int, int, int, int],
    ) -> None:
        base = Path(debug_dir)
        write_png(base / f"{stage}_roi1.png", frame)
        write_png(base / f"{stage}_roi2.png", crop_rect(frame, roi2_rect))
        write_png(base / f"{stage}_roi3.png", crop_rect(frame, roi3_rect))

    def write_meta(self, debug_dir: str, meta: dict) -> None:
        path = Path(debug_dir) / "meta.json"
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def resolve_settings_path() -> Path:
    if getattr(sys, "frozen", False):
        candidates = [
            Path(sys.executable).resolve().parent / "settings",
            Path(__file__).resolve().parent / "settings",
        ]
    else:
        candidates = [Path(__file__).resolve().parents[2] / "settings"]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_offline_config(logger: logging.Logger) -> OfflineConfig:
    settings_path = resolve_settings_path()
    if not settings_path.exists():
        raise FileNotFoundError(f"required settings file not found: {settings_path}")

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    return parse_offline_config(settings, logger)


def parse_offline_config(settings: dict, logger: logging.Logger) -> OfflineConfig:
    peak = settings.get("peak_detect")
    if not isinstance(peak, dict):
        raise ValueError("settings.peak_detect is required for OFFLINE")
    roi2_ext = peak.get("roi2_extension_params")
    if not isinstance(roi2_ext, dict):
        raise ValueError("settings.peak_detect.roi2_extension_params is required for OFFLINE")
    roi3_ext = peak.get("roi3_extension_params")
    if not isinstance(roi3_ext, dict):
        raise ValueError("settings.peak_detect.roi3_extension_params is required for OFFLINE")
    threshold = peak.get("difference_threshold")
    if threshold is None:
        raise ValueError("settings.peak_detect.difference_threshold is required for OFFLINE")
    tmp = settings.get("offline_tmp_frames")
    if not isinstance(tmp, dict):
        raise ValueError("settings.offline_tmp_frames is required for OFFLINE debug saving")
    debug_save_dir = tmp.get("dir")
    if not debug_save_dir:
        raise ValueError("settings.offline_tmp_frames.dir is required for OFFLINE debug saving")

    config = OfflineConfig(
        roi2_extension_params=dict(roi2_ext),
        roi3_extension_params=dict(roi3_ext),
        difference_threshold=float(threshold),
        debug_save_enabled=bool(tmp.get("enabled", False)),
        debug_save_dir=str(debug_save_dir),
    )
    logger.info(
        "offline config loaded: roi2_extension_params=%s roi3_extension_params=%s difference_threshold=%s "
        "debug_save_enabled=%s debug_save_dir=%s",
        config.roi2_extension_params,
        config.roi3_extension_params,
        config.difference_threshold,
        config.debug_save_enabled,
        config.debug_save_dir,
    )
    return config


def create_hidden_window() -> int:
    user32 = ctypes.windll.user32
    hwnd = user32.CreateWindowExW(
        0,
        "STATIC",
        "pywrapper_api_server_hidden_d3d",
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        None,
    )
    if not hwnd:
        raise ctypes.WinError()
    return int(hwnd)


def destroy_window(hwnd: int) -> None:
    if not ctypes.windll.user32.DestroyWindow(hwnd):
        raise ctypes.WinError()


def pump_windows_messages() -> None:
    user32 = ctypes.windll.user32
    msg = MSG()
    while user32.PeekMessageW(ctypes.byref(msg), 0, 0, 0, 1):
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))


class MobileCommEngine:
    def __init__(
        self,
        comm,
        logger: logging.Logger,
        hwnd_factory: Callable[[], int] = create_hidden_window,
        hwnd_destroyer: Callable[[int], None] = destroy_window,
        stream_interval_s: float = 0.016,
    ):
        self._comm = comm
        self._logger = logger
        self._hwnd_factory = hwnd_factory
        self._hwnd_destroyer = hwnd_destroyer
        self._stream_interval_s = stream_interval_s
        self._hwnd: Optional[int] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[FrameSnapshot] = None
        self._frame_seq = 0

    def configure(self) -> None:
        self._logger.info("registering SetOnImageInfoOnceMsg callback")
        self._comm.SetOnImageInfoOnceMsg(self._on_image_info_received)
        self._logger.info("registering SetOnClientStateInfoOnceMsg callback")
        self._comm.SetOnClientStateInfoOnceMsg(self._on_state_info_received)

        self._hwnd = self._hwnd_factory()
        self._logger.info("created hidden D3D HWND=%s", self._hwnd)
        self._comm.SetD3DRenderHWND(self._hwnd)
        self._logger.info("SetD3DRenderHWND completed")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="MobileCommStreamRender", daemon=True)
        self._thread.start()
        self._logger.info("StreamRender loop started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
            self._logger.info("StreamRender loop stopped")
        if self._hwnd is not None:
            hwnd = self._hwnd
            self._hwnd = None
            self._hwnd_destroyer(hwnd)
            self._logger.info("destroyed hidden D3D HWND=%s", hwnd)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                pump_windows_messages()
                self._comm.StreamRender()
            except Exception:
                self._logger.exception("StreamRender loop failed")
                raise
            self._stop_event.wait(self._stream_interval_s)

    def _on_image_info_received(self, header_ptr, image_matrix) -> None:
        shape = getattr(image_matrix, "shape", None)
        self._logger.info("image callback received header_ptr=%s image_shape=%s", header_ptr, shape)
        try:
            frame = np.array(image_matrix, copy=True)
        except Exception:
            self._logger.exception("failed to copy image_matrix from image callback")
            return
        with self._frame_lock:
            self._frame_seq += 1
            self._latest_frame = FrameSnapshot(frame, self._frame_seq, time.time())

    def get_latest_frame(self) -> Optional[FrameSnapshot]:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return FrameSnapshot(np.array(self._latest_frame.image, copy=True), self._latest_frame.seq, self._latest_frame.ts)

    def _on_state_info_received(self, error_info_ptr) -> None:
        if error_info_ptr == 0:
            self._logger.warning("state callback received null pointer")
            return
        try:
            state = ctypes.cast(error_info_ptr, ctypes.POINTER(StateInfo)).contents
        except Exception:
            self._logger.exception("failed to parse state callback pointer")
            return

        self._logger.info(
            "device state: Version=%s AdbServer=%s LicenseType=%s ControlLinkState=%s "
            "ImageInfoLinkState=%s USBLinkState=%s AppRunState=%s",
            state.Version,
            state.AdbServer,
            state.LicenseType,
            state.ControlLinkState,
            state.ImageInfoLinkState,
            state.USBLinkState,
            state.AppRunState,
        )


class PyMobileCommProvider:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger("pywrapper_api_server")
        self._logger.info("initializing PyMobileComm provider")
        module = import_mobile_comm()
        self._logger.info("PyMobileComm module imported from: %s", getattr(module, "__file__", "<unknown>"))
        self._comm = module.CMobileCommunication()
        self._lock = threading.Lock()
        self._engine = MobileCommEngine(self._comm, self._logger)
        self._engine.configure()
        log_adb_devices(self._logger)
        self._logger.info("calling RestartAdbServer")
        self._comm.RestartAdbServer()
        log_adb_devices(self._logger)
        self._logger.info("calling Auto_Initialize")
        self._comm.Auto_Initialize()
        self._engine.start()
        self._logger.info("PyMobileComm provider initialized")

    def fetch(self) -> dict:
        with self._lock:
            self._logger.info("calling GetContentProvider")
            data = self._comm.GetContentProvider()
            self._logger.info("GetContentProvider returned type=%s", type(data).__name__)
            return data

    def get_latest_frame(self) -> Optional[FrameSnapshot]:
        return self._engine.get_latest_frame()

    def close(self) -> None:
        with self._lock:
            self._engine.stop()
            self._logger.info("calling Stop_AutoInitialize")
            self._comm.Stop_AutoInitialize()


class OfflineSessionManager:
    def __init__(
        self,
        provider_fetcher: Callable[[], dict],
        frame_fetcher: Callable[[], Optional[FrameSnapshot]],
        config: OfflineConfig,
        logger: Optional[logging.Logger] = None,
        debug_saver: Optional[DebugFrameSaver] = None,
    ):
        self._provider_fetcher = provider_fetcher
        self._frame_fetcher = frame_fetcher
        self._config = config
        self._logger = logger or logging.getLogger("pywrapper_api_server")
        self._debug_saver = debug_saver or DebugFrameSaver()
        self._lock = threading.Lock()
        self._sessions: dict[object, OfflineSession] = {}

    def handle(self, arg_json_text: Optional[str]) -> dict:
        arg_obj = self._parse_arg(arg_json_text)
        point_id = arg_obj.get("point_id")
        if point_id is None:
            return {"success": False, "info": "missing_point_id"}

        with self._lock:
            if point_id in self._sessions:
                return self._stop(point_id)
            return self._start(point_id)

    def _parse_arg(self, arg_json_text: Optional[str]) -> dict:
        if not arg_json_text:
            raise ValueError("OFFLINE requires JSON args")
        obj = json.loads(arg_json_text)
        if not isinstance(obj, dict):
            raise ValueError("OFFLINE args must be JSON object")
        return obj

    def _start(self, point_id) -> dict:
        frame = self._frame_fetcher()
        if frame is None:
            self._logger.warning("OFFLINE start failed: no_device_frame point_id=%s", point_id)
            return {"success": False, "info": "no_device_frame", "point_id": point_id}

        raw_provider = self._provider_fetcher()
        focus_point = raw_provider.get("focus_point") if isinstance(raw_provider, dict) else None
        if focus_point is None:
            self._logger.warning("OFFLINE start failed: missing_focus_point point_id=%s provider=%s", point_id, safe_json_text(raw_provider))
            return {"success": False, "info": "missing_focus_point", "point_id": point_id}

        anchor = parse_focus_point(focus_point)
        if anchor is None:
            self._logger.warning("OFFLINE start failed: invalid_focus_point point_id=%s focus_point=%r", point_id, focus_point)
            return {"success": False, "info": "invalid_focus_point", "point_id": point_id, "focus_point": focus_point}

        height, width = frame.image.shape[:2]
        roi2_rect = compute_roi_region((width, height), anchor, self._config.roi2_extension_params)
        if roi2_rect is None:
            self._logger.warning(
                "OFFLINE start failed: invalid_roi2_rect point_id=%s frame_shape=%s anchor=%s roi2_ext=%s",
                point_id,
                frame.image.shape,
                anchor,
                self._config.roi2_extension_params,
            )
            return {"success": False, "info": "invalid_roi2_rect", "point_id": point_id}
        roi3_rect = compute_roi_region((width, height), anchor, self._config.roi3_extension_params)
        if roi3_rect is None:
            self._logger.warning(
                "OFFLINE start failed: invalid_roi3_rect point_id=%s frame_shape=%s anchor=%s roi3_ext=%s",
                point_id,
                frame.image.shape,
                anchor,
                self._config.roi3_extension_params,
            )
            return {"success": False, "info": "invalid_roi3_rect", "point_id": point_id}

        before_mean = roi_gray_mean(frame.image, roi2_rect)
        debug_dir = None
        meta = {
            "point_id": point_id,
            "focus_anchor": [int(anchor[0]), int(anchor[1])],
            "roi2_rect": [int(v) for v in roi2_rect],
            "roi3_rect": [int(v) for v in roi3_rect],
            "before": {
                "frame_seq": int(frame.seq),
                "frame_ts": float(frame.ts),
                "frame_shape": [int(v) for v in frame.image.shape],
                "roi2_mean": round(float(before_mean), 6),
            },
        }
        if self._config.debug_save_enabled:
            try:
                debug_dir = self._debug_saver.create_session_dir(self._config.debug_save_dir, point_id)
                self._debug_saver.save_stage(debug_dir, "before", frame.image, roi2_rect, roi3_rect)
                self._debug_saver.write_meta(debug_dir, meta)
            except Exception as exc:
                self._logger.exception("OFFLINE debug save failed on start: point_id=%s", point_id)
                return {"success": False, "info": "debug_save_failed", "point_id": point_id, "error": str(exc)}

        session = OfflineSession(
            point_id=point_id,
            before=np.array(frame.image, copy=True),
            before_seq=frame.seq,
            before_ts=frame.ts,
            focus_anchor=anchor,
            roi2_rect=roi2_rect,
            roi3_rect=roi3_rect,
            before_mean=before_mean,
            debug_dir=debug_dir,
            meta=meta,
        )
        self._sessions[point_id] = session
        self._logger.info(
            "OFFLINE started: point_id=%s frame_seq=%s frame_shape=%s focus_point=%r anchor=%s "
            "roi2_rect=%s roi3_rect=%s before_mean=%.3f debug_dir=%s",
            point_id,
            frame.seq,
            frame.image.shape,
            focus_point,
            anchor,
            roi2_rect,
            roi3_rect,
            before_mean,
            debug_dir,
        )
        result = {"success": True, "info": "offline_started", "point_id": point_id}
        if debug_dir is not None:
            result["debug_dir"] = debug_dir
        return result

    def _stop(self, point_id) -> dict:
        session = self._sessions[point_id]
        frame = self._frame_fetcher()
        if frame is None:
            self._logger.warning("OFFLINE stop failed: no_device_frame point_id=%s", point_id)
            return {"success": False, "info": "no_device_frame", "point_id": point_id}

        after_mean = roi_gray_mean(frame.image, session.roi2_rect)
        diff = float(after_mean - session.before_mean)
        color = "green" if diff >= self._config.difference_threshold else "red"
        result = {
            "success": True,
            "info": "offline_stop_completed",
            "point_id": point_id,
            "roi2_color": color,
            "roi2_diff": round(diff, 6),
            "roi2_before_mean": round(float(session.before_mean), 6),
            "roi2_after_mean": round(float(after_mean), 6),
            "focus_anchor": [int(session.focus_anchor[0]), int(session.focus_anchor[1])],
            "roi2_rect": [int(v) for v in session.roi2_rect],
            "roi3_rect": [int(v) for v in session.roi3_rect],
        }
        if session.debug_dir is not None:
            result["debug_dir"] = session.debug_dir
            meta = dict(session.meta)
            meta["after"] = {
                "frame_seq": int(frame.seq),
                "frame_ts": float(frame.ts),
                "frame_shape": [int(v) for v in frame.image.shape],
                "roi2_mean": round(float(after_mean), 6),
            }
            meta["result"] = {
                "roi2_color": color,
                "roi2_diff": round(diff, 6),
                "roi2_before_mean": round(float(session.before_mean), 6),
                "roi2_after_mean": round(float(after_mean), 6),
            }
            try:
                self._debug_saver.save_stage(session.debug_dir, "after", frame.image, session.roi2_rect, session.roi3_rect)
                self._debug_saver.write_meta(session.debug_dir, meta)
            except Exception as exc:
                self._logger.exception("OFFLINE debug save failed on stop: point_id=%s", point_id)
                return {"success": False, "info": "debug_save_failed", "point_id": point_id, "error": str(exc)}

        self._sessions.pop(point_id, None)
        self._logger.info(
            "OFFLINE stopped: point_id=%s before_seq=%s after_seq=%s after_shape=%s anchor=%s roi2_rect=%s "
            "roi3_rect=%s before_mean=%.3f after_mean=%.3f diff=%.3f threshold=%.3f color=%s debug_dir=%s",
            point_id,
            session.before_seq,
            frame.seq,
            frame.image.shape,
            session.focus_anchor,
            session.roi2_rect,
            session.roi3_rect,
            session.before_mean,
            after_mean,
            diff,
            self._config.difference_threshold,
            color,
            session.debug_dir,
        )
        return result


def parse_request(text: str) -> ParsedRequest:
    parts = text.strip().split(";", 2)
    if len(parts) < 2:
        raise ValueError("request must use REQ_TYPE;PASSWORD[;JSON] format")
    req_type = parts[0].strip().upper()
    param = parts[1].strip()
    arg = parts[2].strip() if len(parts) == 3 and parts[2].strip() else None
    if not req_type:
        raise ValueError("request type is empty")
    return ParsedRequest(req_type=req_type, param=param, arg=arg)


def convert_provider_data(raw_data: dict) -> dict:
    if raw_data is None:
        raise ValueError("GetContentProvider returned None")
    if not isinstance(raw_data, dict):
        raise TypeError(f"GetContentProvider must return dict, got {type(raw_data).__name__}")

    is_live = raw_data.get("isLive")
    is_freeze = not is_live if is_live is not None else None
    is_hifu = raw_data.get("mode") == 2

    return {
        "SkinDepth": raw_data.get("focus_depth"),
        "A": raw_data.get("guankuan_a"),
        "B": raw_data.get("guankuan_b"),
        "Alpha": None,
        "Depth": raw_data.get("depth"),
        "IsFreeze": is_freeze,
        "isHIFU": is_hifu,
        "FocusPoint": raw_data.get("focus_point"),
    }


def json_response(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def safe_json_text(payload) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return repr(payload)


def log_online_diagnostics(
    logger: Optional[logging.Logger],
    raw_data: dict,
    converted_data: dict,
) -> None:
    if logger is None:
        return

    logger.info("ONLINE raw provider data: %s", safe_json_text(raw_data))
    logger.info("ONLINE converted response: %s", safe_json_text(converted_data))

    missing_provider_fields = [
        field for field in PROVIDER_FIELDS if not isinstance(raw_data, dict) or raw_data.get(field) is None
    ]
    if missing_provider_fields:
        logger.warning("ONLINE missing provider fields: %s", ", ".join(missing_provider_fields))

    null_response_fields = [
        field for field, value in converted_data.items() if value is None
    ]
    if null_response_fields:
        logger.warning("ONLINE null response fields: %s", ", ".join(null_response_fields))


def handle_request(
    request_text: str,
    provider_fetcher: Callable[[], dict],
    logger: Optional[logging.Logger] = None,
    offline_handler: Optional[Callable[[Optional[str]], dict]] = None,
) -> str:
    parsed = parse_request(request_text)
    if logger is not None:
        logger.info("request received: type=%s arg=%s", parsed.req_type, parsed.arg)

    if parsed.param != PASSWORD:
        if logger is not None:
            logger.warning("request rejected: invalid password for type=%s", parsed.req_type)
        return json_response({"success": False, "info": "invalid_password"})

    if parsed.req_type == "ONLINE":
        raw_data = provider_fetcher()
        converted_data = convert_provider_data(raw_data)
        log_online_diagnostics(logger, raw_data, converted_data)
        return json_response(converted_data)

    if parsed.req_type == "OFFLINE":
        if offline_handler is None:
            return json_response({"success": False, "info": "offline_not_configured"})
        return json_response(offline_handler(parsed.arg))

    return json_response(
        {"success": False, "info": "unknown_request_type", "req_type": parsed.req_type}
    )


def scan_json_end(text: str, start_idx: int = 0) -> int:
    i = start_idx
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    if i >= n or text[i] not in "{[":
        return -1

    stack = [text[i]]
    i += 1
    in_str = False
    esc = False
    while i < n:
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                opener = stack.pop()
                if (opener == "{" and ch != "}") or (opener == "[" and ch != "]"):
                    return -1
                if not stack:
                    return i + 1
        i += 1
    return -1


def try_parse_buffer(buffer: str) -> Optional[Tuple[str, str]]:
    stripped = buffer.lstrip("\r\n")
    offset = len(buffer) - len(stripped)
    first = stripped.find(";")
    if first < 0:
        return None
    second = stripped.find(";", first + 1)
    if second < 0:
        return None

    rest = stripped[second + 1 :]
    json_end = scan_json_end(rest)
    if json_end >= 0:
        request_text = stripped[: second + 1 + json_end]
        remaining = stripped[second + 1 + json_end :].lstrip("\r\n")
        return request_text, remaining

    if "\n" in stripped:
        line, remaining = stripped.split("\n", 1)
        return line.strip(), remaining

    if offset:
        return None
    return None


class ApiServer:
    def __init__(self, provider: PyMobileCommProvider, logger: logging.Logger, offline_manager: OfflineSessionManager):
        self._provider = provider
        self._logger = logger
        self._offline_manager = offline_manager

    def handle_client(self, client_socket: socket.socket, client_address) -> None:
        self._logger.info("client connected: %s", client_address)
        buffer = ""
        try:
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="strict")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._send_response(client_socket, line)

                while True:
                    parsed = try_parse_buffer(buffer)
                    if parsed is None:
                        break
                    request_text, buffer = parsed
                    self._send_response(client_socket, request_text)
        finally:
            client_socket.close()
            self._logger.info("client closed: %s", client_address)

    def _send_response(self, client_socket: socket.socket, request_text: str) -> None:
        try:
            response = handle_request(
                request_text,
                self._provider.fetch,
                logger=self._logger,
                offline_handler=self._offline_manager.handle,
            )
        except Exception as exc:
            self._logger.exception("request failed: %r", request_text)
            response = json_response({"success": False, "info": "request_failed", "error": str(exc)})
        self._logger.info("response sent: %s", response)
        client_socket.sendall((response + "\n").encode("utf-8"))

    def serve_forever(self, host: str, port: int) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.listen(5)
            self._logger.info("api server listening on %s:%s", host, port)
            print(f"api server listening on {host}:{port}", flush=True)

            while True:
                client_socket, client_address = server_socket.accept()
                threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, client_address),
                    daemon=True,
                ).start()


def build_logger() -> logging.Logger:
    if getattr(sys, "frozen", False):
        log_dir = Path(sys.executable).resolve().parent / "ocrlog"
    else:
        log_dir = Path(__file__).resolve().parents[2] / "ocrlog"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pywrapper_api_server")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_dir / "pywrapper_api_server.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.info("log file: %s", log_dir / "pywrapper_api_server.log")
    return logger


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PyMobileComm TCP API server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    logger = build_logger()
    log_process_environment(logger)
    offline_config = load_offline_config(logger)
    provider = PyMobileCommProvider(logger)
    offline_manager = OfflineSessionManager(
        provider_fetcher=provider.fetch,
        frame_fetcher=provider.get_latest_frame,
        config=offline_config,
        logger=logger,
    )
    try:
        ApiServer(provider, logger, offline_manager).serve_forever(args.host, args.port)
    finally:
        provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
