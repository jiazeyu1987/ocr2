# -*- coding: utf-8 -*-
import argparse
import ctypes
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple


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

    def configure(self) -> None:
        self._logger.info("registering SetOnClientOnceMsg callback")
        self._comm.SetOnClientOnceMsg(self._on_control_msg_received)
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

    def _on_control_msg_received(self, header_ptr, data_bytes) -> None:
        self._logger.info("control callback received header_ptr=%s data_type=%s", header_ptr, type(data_bytes).__name__)

    def _on_image_info_received(self, header_ptr, image_matrix) -> None:
        shape = getattr(image_matrix, "shape", None)
        self._logger.info("image callback received header_ptr=%s image_shape=%s", header_ptr, shape)

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

    def close(self) -> None:
        with self._lock:
            self._engine.stop()
            self._logger.info("calling Stop_AutoInitialize")
            self._comm.Stop_AutoInitialize()


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
        return json_response({"success": True, "info": "offline_ok"})

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
    def __init__(self, provider: PyMobileCommProvider, logger: logging.Logger):
        self._provider = provider
        self._logger = logger

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
            response = handle_request(request_text, self._provider.fetch, logger=self._logger)
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
    provider = PyMobileCommProvider(logger)
    try:
        ApiServer(provider, logger).serve_forever(args.host, args.port)
    finally:
        provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
