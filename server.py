# -*- coding: latin-1 -*-
import os

# Work around OpenMP runtime conflicts on Windows (common with MKL + Paddle/OpenCV).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import socket
import threading
# import multiprocessing
import json
import time
import sys
import os

from ocr_detect import OCRDetect
import logging
from datetime import datetime
from treat_compare_img import ComparePoints

class ImageProcessServer:
    def __init__(self):

        # logging.basicConfig(
        #     level=logging.INFO,  # 设置日志级别�?INFO，这意味着 INFO 及以上级别的日志会被记录
        #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
        #     filename='ocrserver.log',  # 指定日志输出文件，如果不指定则默认输出到控制�?
        #     filemode="a"
        # )


        self.init_logger()


        # 定义支持的请求类�?
        self.REQUEST_TYPES = {

            "Offline": "offline request",
            "Online": "online request",
            "OCR": "get OCR status",
            "CLOSEOCR": "close OCR loop",
            "OPENOCR": "open OCR loop",
        }

        self.setting = self.load_setting()
        # Verbose OFFLINE diagnostics switch (controlled by settings["peak_debug_log"]["enabled"]).
        try:
            pdl = (self.setting or {}).get("peak_debug_log")
            if isinstance(pdl, dict):
                self._peak_debug_enabled = bool(pdl.get("enabled", False))
            else:
                self._peak_debug_enabled = bool(pdl)
        except Exception:
            self._peak_debug_enabled = False
        self._offline_req_seq = 0
        self._offline_last_action = {}  # point_id -> {"action": "start"/"stop", "ts": float, "seq": int}
        self._offline_point_req_count = {}  # point_id -> accepted OFFLINE request count (max 2)
        # Bound offline history maps to avoid unbounded memory growth with ever-new point_id.
        try:
            self._offline_history_max_points = int((self.setting or {}).get("offline_history_max_points", 2000))
        except Exception:
            self._offline_history_max_points = 2000
        try:
            self._offline_history_ttl_seconds = float((self.setting or {}).get("offline_history_ttl_seconds", 6 * 3600))
        except Exception:
            self._offline_history_ttl_seconds = float(6 * 3600)

        self._client_state_lock = threading.Lock()
        self._active_client_connections = 0
        try:
            self._max_client_connections = int((self.setting or {}).get("max_client_connections", 64))
        except Exception:
            self._max_client_connections = 64
        self._online_wait_timeout_ms = 2500
        self._online_req_lock = threading.Lock()
        self._online_req_seq = 0

        self.logger.info(
            f"offline history guard enabled: max_points={self._offline_history_max_points}, ttl_s={self._offline_history_ttl_seconds}"
        )
        self.logger.info(
            f"client connection guard enabled: max_conn={self._max_client_connections}"
        )

        self.ocrserver = OCRDetect(self.setting, self.logger)

        self.start_ocr_server()
        self.start_watchdog()



        # for 治疗前后对比截图
        self.logger.info("compare tool loaded")

        self.point_id = None
        self.client_thread = None
        self.compare_client = None
        # OFFLINE compare is a two-signal session (start/stop). Use a per-session stop event.
        # Do NOT reuse/clear a shared Event across sessions; otherwise an old thread may resume.
        self.compare_stop_event = None
        self._offline_lock = threading.Lock()

        # OFFLINE session state:
        # - Each OFFLINE run gets its own ComparePoints instance to avoid state corruption when a previous
        #   session is still doing slow IO (saving images / DB insert).
        # - We keep references to "orphan" sessions so they can finish in background.
        self._offline_session = None  # dict(point_id, thread, stop_event, tool)
        self._offline_orphans = []
        self.compareTool = None
        # 为了首次调用服务器时不延迟，此处默认调用一�?
        # default_offline = {"point_id": 3141592653, "is_save": False, "time_out": 100}
        # self.get_offline(json.dumps(default_offline))
        # time.sleep(1)
        # self.get_offline(json.dumps(default_offline))


    def init_logger(self, dst='ocrlog'):
        # 如果没有指定日志文件，则使用当前日期作为文件�?
        if not os.path.exists(dst):
            os.makedirs(dst)
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(dst, f"ocrapp_{today}.log")

        # ?
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # 设置日志级别为DEBUG

        # 创建文件处理�?
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # 创建格式化器并添加到处理�?
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # 添加处理器到日志�?
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)


    def load_setting(self):
        # Default to the production config path, but fall back to local app directory (useful for packaging/debug).
        candidates = [r'D:\software_data\settings']
        try:
            app_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
            candidates.append(os.path.join(app_dir, 'settings'))
        except Exception:
            pass

        for setting_path in candidates:
            if not setting_path:
                continue
            if not os.path.exists(setting_path):
                continue
            try:
                with open(setting_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.logger.info(f"Loaded settings from: {setting_path}")
                return cfg
            except Exception as e:
                try:
                    self.logger.error(f"Failed to parse settings JSON: {setting_path}: {e}")
                except Exception:
                    pass
                continue

        try:
            self.logger.warning(f"No settings file found. Tried: {candidates}")
        except Exception:
            pass
        return None

    def _pdbg(self, msg: str):
        try:
            if getattr(self, "_peak_debug_enabled", False) and getattr(self, "logger", None):
                self.logger.info(f"[peakdbg] {msg}")
        except Exception:
            pass

    def _prune_offline_history_locked(self):
        """
        Prune in-memory OFFLINE history maps to keep memory bounded.
        Must be called under self._offline_lock.
        """
        try:
            max_points = max(100, int(getattr(self, "_offline_history_max_points", 2000)))
        except Exception:
            max_points = 2000
        try:
            ttl = max(60.0, float(getattr(self, "_offline_history_ttl_seconds", 6 * 3600)))
        except Exception:
            ttl = float(6 * 3600)

        now_ts = time.time()
        removed = 0

        try:
            stale_keys = []
            for k, v in list((self._offline_last_action or {}).items()):
                ts = None
                try:
                    ts = float((v or {}).get("ts"))
                except Exception:
                    ts = None
                if ts is None or (now_ts - ts) > ttl:
                    stale_keys.append(k)
            for k in stale_keys:
                if k in self._offline_last_action:
                    self._offline_last_action.pop(k, None)
                    removed += 1
                self._offline_point_req_count.pop(k, None)
        except Exception:
            pass

        try:
            if len(self._offline_last_action) > max_points:
                ordered = sorted(
                    self._offline_last_action.items(),
                    key=lambda kv: float(((kv[1] or {}).get("ts")) or 0.0),
                    reverse=True,
                )
                keep_keys = {k for (k, _) in ordered[:max_points]}
                drop_keys = [k for k in self._offline_last_action.keys() if k not in keep_keys]
                for k in drop_keys:
                    self._offline_last_action.pop(k, None)
                    self._offline_point_req_count.pop(k, None)
                    removed += 1
        except Exception:
            pass

        try:
            if len(self._offline_point_req_count) > max_points:
                count_keys = list(self._offline_point_req_count.keys())
                for k in count_keys[:-max_points]:
                    self._offline_point_req_count.pop(k, None)
                    removed += 1
        except Exception:
            pass

        if removed > 0:
            self._pdbg(
                f"offline history pruned: removed={removed}, last_action={len(self._offline_last_action)}, "
                f"req_count={len(self._offline_point_req_count)}"
            )
            self.logger.info(
                f"offline history prune applied: removed={removed}, last_action={len(self._offline_last_action)}, req_count={len(self._offline_point_req_count)}"
            )

    @staticmethod
    def _normalize_roi2_color(color):
        return "green" if str(color).strip().lower() == "green" else "red"

    def _get_tool_roi2_color(self, tool):
        if tool is None:
            return "red"
        try:
            color = getattr(tool, "final_roi2_color", None)
        except Exception:
            color = None
        if color is None:
            try:
                color = (getattr(tool, "response", {}) or {}).get("roi2_color")
            except Exception:
                color = None
        return self._normalize_roi2_color(color)

    def start_ocr_server(self):
        # 啓動實時識別
        # self.ocrthread = multiprocessing.Process(
        self.ocrthread = threading.Thread(
            target=self.ocrserver.start_ocr_server,
            args=(),
            daemon=True
        )

        self.ocrthread.start()

    def start_watchdog(self):
        setting = self.setting or {}
        enabled = bool(setting.get("watchdog_enable", False))
        if not enabled:
            return

        self._watchdog_cfg = {
            "check_interval_seconds": float(setting.get("watchdog_check_interval_seconds", 2.0)),
            "capture_stale_seconds": float(setting.get("watchdog_capture_stale_seconds", 15.0)),
            "ocr_stale_seconds": float(setting.get("watchdog_ocr_stale_seconds", 30.0)),
            "max_consecutive_failures": int(setting.get("watchdog_max_consecutive_failures", 10)),
            "exit_code": int(setting.get("watchdog_exit_code", 42)),
        }

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            args=(),
            daemon=True,
        )
        self._watchdog_thread.start()
        self.logger.info(f"watchdog enabled: {self._watchdog_cfg}")

    def _watchdog_loop(self):
        cfg = getattr(self, "_watchdog_cfg", None) or {
            "check_interval_seconds": 2.0,
            "capture_stale_seconds": 15.0,
            "ocr_stale_seconds": 30.0,
            "max_consecutive_failures": 10,
            "exit_code": 42,
        }

        while True:
            try:
                time.sleep(cfg["check_interval_seconds"])
                if not hasattr(self.ocrserver, "get_health"):
                    continue

                h = self.ocrserver.get_health()
                now = time.time()
                capture_stale = (now - float(h.get("last_capture_ok_ts", 0))) > cfg["capture_stale_seconds"]
                ocr_stale = (now - float(h.get("last_ocr_ok_ts", 0))) > cfg["ocr_stale_seconds"]
                too_many_failures = int(h.get("consecutive_failures", 0)) >= cfg["max_consecutive_failures"]

                if capture_stale or ocr_stale or too_many_failures:
                    reason = {
                        "capture_stale": capture_stale,
                        "ocr_stale": ocr_stale,
                        "too_many_failures": too_many_failures,
                        "health": h,
                        "cfg": cfg,
                    }
                    self.logger.critical(f"watchdog trigger, exiting process: {reason}")
                    for handler in list(self.logger.handlers):
                        try:
                            handler.flush()
                        except Exception:
                            pass

                    # sys.exit in a thread won't kill the whole process; use os._exit for supervisor restarts.
                    os._exit(cfg["exit_code"])
            except Exception as e:
                try:
                    self.logger.exception(f"watchdog loop error: {e}")
                except Exception:
                    pass
                time.sleep(1.0)


    def close_ocr_server(self):
        try:
            if hasattr(self.ocrserver, "stop_ocr_server"):
                self.ocrserver.stop_ocr_server()
            self.logger.info("CLOSEOCR applied: OCR loop paused")
            return True
        except Exception as e:
            self.logger.error(f"CLOSEOCR failed: {e}")
            return False

    def open_ocr_server(self):
        try:
            if hasattr(self.ocrserver, "open_ocr_server"):
                self.ocrserver.open_ocr_server()
            self.logger.info("OPENOCR applied: OCR loop resumed")
            return True
        except Exception as e:
            self.logger.error(f"OPENOCR failed: {e}")
            return False


    def _next_online_request_seq(self):
        with self._online_req_lock:
            self._online_req_seq += 1
            return self._online_req_seq

    @staticmethod
    def _build_online_results(measures):
        m = measures or {}
        return {
            'SkinDepth': m.get('skin_distance'),
            'A': m.get('A'),
            'B': m.get('B'),
            'Alpha': m.get('Alpha'),
            'CaptureTimestamp': m.get('CaptureTimestamp'),
            'RecognizeStartTimestamp': m.get('RecognizeStartTimestamp'),
            'Depth': m.get(r'\u6df1\u5ea6'.encode('ascii').decode('unicode_escape')),
            'IsFreeze': m.get('Is_Freeze'),
            'isHIFU': m.get('Is_HIFU', False),
            'Points_Per_MM': m.get('Points_Per_MM'),
        }

    def get_online(self, online_request_ts_ms):
        """Read OCR cache for ONLINE."""

        measures, wait_result = self.ocrserver.get_measures_after_capture_ts(
            min_capture_ts_ms=online_request_ts_ms,
            wait_timeout_ms=self._online_wait_timeout_ms
        )
        results = self._build_online_results(measures)

        return results, wait_result

    def get_offline(self, arg):

        results = ""
        self.logger.info("new offline method")

        if arg is None:
            results = {'success': False, 'info': 'invalid_input'}
            return results

        # Parse once so we can log the structured payload.
        arg_obj = json.loads(arg)

        point_id = arg_obj['point_id']
        time_out = arg_obj['time_out']
        is_save = arg_obj['is_save']

        try:
            self._offline_req_seq = int(self._offline_req_seq) + 1
        except Exception:
            self._offline_req_seq = 1
        seq = self._offline_req_seq
        self._pdbg(f"OFFLINE recv: seq={seq}, point_id={point_id}, time_out={time_out}, is_save={is_save}, raw={arg_obj}")

        """计算非实时的结果"""
        # if self.compareTool is None:
        #     from treat_compare_img import ComparePoints
        #     self.compareTool = ComparePoints(self.logger)
        self.logger.info("compare tool loaded")

        # Serialize OFFLINE start/stop to avoid starting two capture loops at the same time.
        with self._offline_lock:
            self._prune_offline_history_locked()
            try:
                point_key = int(point_id)
            except Exception:
                point_key = point_id

            accepted_count = int(self._offline_point_req_count.get(point_key, 0))
            if accepted_count >= 2:
                self._pdbg(
                    f"OFFLINE drop: seq={seq}, point_id={point_id}, reason=too_many_requests, accepted_count={accepted_count}"
                )
                self.logger.info(f"OFFLINE ignored: point_id={point_id}, accepted_count={accepted_count}, seq={seq}")
                return {"success": False, "info": "offline_ignored_extra_request", "point_id": point_id}

            # Snapshot current active session (helps diagnose duplicated/partial OFFLINE sequences).
            try:
                active0 = self._offline_session
                if active0 is None:
                    self._pdbg(f"OFFLINE state(before): seq={seq}, active=None, orphans={len(self._offline_orphans or [])}")
                else:
                    t0 = active0.get("thread")
                    tool0 = active0.get("tool")
                    cap0 = getattr(tool0, "_capture_done_event", None) if tool0 is not None else None
                    fin0 = getattr(tool0, "_finished_event", None) if tool0 is not None else None
                    stage0 = getattr(tool0, "_stage", None) if tool0 is not None else None
                    detail0 = getattr(tool0, "_stage_detail", None) if tool0 is not None else None
                    start_ts0 = getattr(tool0, "_session_start_ts", None) if tool0 is not None else None
                    stop_ts0 = getattr(tool0, "_stop_requested_ts", None) if tool0 is not None else None
                    now0 = time.time()
                    elapsed0 = (now0 - float(start_ts0)) if start_ts0 else None
                    since_stop0 = (now0 - float(stop_ts0)) if stop_ts0 else None
                    self._pdbg(
                        f"OFFLINE state(before): seq={seq}, active_point_id={active0.get('point_id')}, "
                        f"alive={t0.is_alive() if t0 is not None else None}, "
                        f"capture_done={cap0.is_set() if cap0 is not None else None}, finished={fin0.is_set() if fin0 is not None else None}, "
                        f"stage={stage0}, detail={detail0}, elapsed_s={elapsed0}, since_stop_s={since_stop0}, "
                        f"orphans={len(self._offline_orphans or [])}"
                    )
            except Exception:
                pass

            # Prune finished orphan sessions (best-effort).
            try:
                keep = []
                for s in list(self._offline_orphans or []):
                    t = (s or {}).get("thread")
                    if t is not None and getattr(t, "is_alive", None) and t.is_alive():
                        keep.append(s)
                self._offline_orphans = keep[-8:]
            except Exception:
                pass

            active = self._offline_session

            # Second OFFLINE signal (same point_id): stop current session.
            if active is not None and active.get("point_id") == point_id:
                t_stop_total = time.perf_counter()
                stop_recv_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[OFFLINE-PROFILE] step=stop_recv, seq={seq}, point_id={point_id}, ts={stop_recv_ts}")
                self._pdbg(f"OFFLINE action: seq={seq}, point_id={point_id}, action=stop (same as active)")
                self._offline_point_req_count[point_key] = accepted_count + 1
                self.logger.info(
                    f"OFFLINE accepted: point_id={point_id}, accepted_count={self._offline_point_req_count[point_key]}, action=stop, seq={seq}"
                )
                self.point_id = None
                tool = None
                t = None
                finished_ev = None
                finished_ok = False
                try:
                    tool = active.get("tool")
                    t = active.get("thread")
                    if tool is not None:
                        try:
                            tool._stop_requested_ts = time.time()
                        except Exception:
                            pass
                        try:
                            finished_ev = getattr(tool, "_finished_event", None)
                        except Exception:
                            finished_ev = None
                    ev = active.get("stop_event")
                    if ev is not None:
                        ev.set()
                        self.logger.info(f"[OFFLINE-PROFILE] step=stop_signal_set, seq={seq}, point_id={point_id}, elapsed_ms={(time.perf_counter()-t_stop_total)*1000.0:.1f}")
                except Exception:
                    pass
                try:
                    wait_timeout = float((self.setting or {}).get("offline_stop_wait_timeout_seconds", 20.0))
                except Exception:
                    wait_timeout = 20.0
                wait_timeout = max(1.0, min(wait_timeout, 120.0))
                self.logger.info(f"[OFFLINE-PROFILE] step=stop_wait_begin, seq={seq}, point_id={point_id}, wait_timeout_s={wait_timeout}")
                t_wait = time.perf_counter()
                try:
                    if finished_ev is not None:
                        finished_ok = bool(finished_ev.wait(timeout=wait_timeout))
                    elif t is not None:
                        t.join(timeout=wait_timeout)
                        finished_ok = not t.is_alive()
                except Exception:
                    finished_ok = False
                self.logger.info(f"[OFFLINE-PROFILE] step=stop_wait_done, seq={seq}, point_id={point_id}, finished_ok={bool(finished_ok)}, wait_elapsed_ms={(time.perf_counter()-t_wait)*1000.0:.1f}, total_elapsed_ms={(time.perf_counter()-t_stop_total)*1000.0:.1f}")
                roi2_color = self._get_tool_roi2_color(tool)
                stop_info = "offline_stop_completed" if finished_ok else "offline_stop_timeout"
                self.logger.info(
                    f"[OFFLINE-STOP-SEND] seq={seq}, point_id={point_id}, roi2_color={roi2_color}, "
                    f"roi2_final={bool(finished_ok)}, info={stop_info}"
                )
                self.logger.info(f"[OFFLINE-PROFILE] step=stop_response_send, seq={seq}, point_id={point_id}, total_elapsed_ms={(time.perf_counter()-t_stop_total)*1000.0:.1f}")
                try:
                    self._offline_orphans.append(active)
                except Exception:
                    pass
                self._offline_session = None
                try:
                    self._offline_last_action[int(point_id)] = {"action": "stop", "ts": time.time(), "seq": seq}
                except Exception:
                    pass
                return {
                    "success": True,
                    "info": stop_info,
                    "point_id": point_id,
                    "roi2_color": roi2_color,
                    "roi2_final": bool(finished_ok),
                }

            # New point_id (or no active session): stop previous capture loop if needed.
            if active is not None:
                self._pdbg(
                    f"OFFLINE action: seq={seq}, point_id={point_id}, action=switch_start (active_point_id={active.get('point_id')})"
                )
                try:
                    tool = active.get("tool")
                    if tool is not None:
                        try:
                            tool._stop_requested_ts = time.time()
                        except Exception:
                            pass
                    ev = active.get("stop_event")
                    if ev is not None:
                        ev.set()
                except Exception:
                    pass

                t = active.get("thread")
                tool = active.get("tool")
                capture_done = getattr(tool, "_capture_done_event", None) if tool is not None else None
                finished = getattr(tool, "_finished_event", None) if tool is not None else None

                # If the previous session is still in the screenshot loop, wait for it to exit.
                try:
                    if capture_done is not None:
                        capture_done.wait(timeout=2)
                except Exception:
                    pass

                try:
                    if t is not None and t.is_alive() and (capture_done is None or (not capture_done.is_set())):
                        t.join(timeout=5)
                except Exception:
                    pass

                if t is not None and t.is_alive() and (capture_done is None or (not capture_done.is_set())):
                    # Previous session did not exit in time (often due to slow IO like cv2.imwrite / sqlite insert).
                    # Since we no longer share the ComparePoints instance across sessions, it is safe to proceed.
                    try:
                        now = time.time()
                        start_ts = getattr(tool, "_session_start_ts", None) if tool is not None else None
                        stop_ts = getattr(tool, "_stop_requested_ts", None) if tool is not None else None
                        stage = getattr(tool, "_stage", None) if tool is not None else None
                        stage_detail = getattr(tool, "_stage_detail", None) if tool is not None else None
                        elapsed = (now - float(start_ts)) if start_ts else None
                        stop_elapsed = (now - float(stop_ts)) if stop_ts else None
                        capture_done_set = bool(capture_done.is_set()) if capture_done is not None else None
                        finished_set = bool(finished.is_set()) if finished is not None else None
                    except Exception:
                        elapsed = None
                        stop_elapsed = None
                        stage = None
                        stage_detail = None
                        capture_done_set = None
                        finished_set = None

                    self.logger.warning(
                        "OFFLINE switch: previous compare thread still alive; "
                        f"prev_point_id={active.get('point_id')}, prev_alive={t.is_alive() if t is not None else None}, "
                        f"capture_done={capture_done_set}, finished={finished_set}, stage={stage}, detail={stage_detail}, "
                        f"elapsed_s={elapsed}, since_stop_s={stop_elapsed}, orphans={len(self._offline_orphans or [])}; "
                        "starting new session anyway"
                    )

                # Previous session is either finished, or only doing slow post-processing IO.
                # Allow a new OFFLINE session by creating a fresh ComparePoints instance.
                try:
                    self._offline_orphans.append(active)
                except Exception:
                    pass
                self._offline_session = None

            # Start a new OFFLINE session with a fresh stop event and a fresh ComparePoints instance.
            # If user sends multiple OFFLINE with the same point_id but there's no active session,
            # this will be treated as START (not STOP). Log history to make it explicit.
            try:
                last = self._offline_last_action.get(int(point_id))
                if last is not None:
                    self._pdbg(
                        f"OFFLINE history: seq={seq}, point_id={point_id}, last_action={last.get('action')}, "
                        f"last_seq={last.get('seq')}, last_age_s={(time.time() - float(last.get('ts'))) if last.get('ts') else None}"
                    )
            except Exception:
                pass
            self._pdbg(f"OFFLINE action: seq={seq}, point_id={point_id}, action=start")
            t_start_total = time.perf_counter()
            start_recv_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logger.info(
                f"[OFFLINE-PROFILE] step=start_recv, seq={seq}, point_id={point_id}, ts={start_recv_ts}, "
                f"time_out={time_out}, is_save={is_save}"
            )
            self.point_id = point_id
            stop_event = threading.Event()
            tool = ComparePoints(self.setting, self.logger)
            t = threading.Thread(
                target=tool.detect,
                args=(point_id, float(time_out), is_save, stop_event),
                daemon=True,
            )
            self._offline_session = {"point_id": point_id, "thread": t, "stop_event": stop_event, "tool": tool}
            self._offline_point_req_count[point_key] = accepted_count + 1
            self.logger.info(
                f"OFFLINE accepted: point_id={point_id}, accepted_count={self._offline_point_req_count[point_key]}, action=start, seq={seq}"
            )
            self.compare_stop_event = stop_event
            self.compare_client = t
            t.start()
            self.logger.info(
                f"[OFFLINE-PROFILE] step=start_thread_started, seq={seq}, point_id={point_id}, "
                f"total_elapsed_ms={(time.perf_counter()-t_start_total)*1000.0:.1f}"
            )
            try:
                self._offline_last_action[int(point_id)] = {"action": "start", "ts": time.time(), "seq": seq}
            except Exception:
                pass
            return {"success": True, "info": "offline_started", "point_id": point_id}



        return {"success": False, "info": "offline_unexpected_state"}

    def handle_client(self, client_socket, client_address):
        """处理客户端请求的函数"""

        self.logger.info(f"client connected: {client_address}")


        try:
            # IMPORTANT: TCP is a byte stream. A recv() may contain multiple requests (coalesced),
            # or one request may be split across recv() calls. If we directly json.loads() the payload,
            # we can hit errors like: '{"..."}OFFLINE{"..."}' -> "Extra data".
            #
            # Recommended client framing: append '\n' after each request.
            # This server also supports concatenated OFFLINE requests without '\n' (best-effort) by
            # detecting JSON object boundaries.
            buffer = ""

            def scan_json_end(s, start_idx):
                i = start_idx
                n = len(s)
                while i < n and s[i].isspace():
                    i += 1
                if i >= n:
                    return -1
                if s[i] not in "{[":
                    return -1
                stack = [s[i]]
                i += 1
                in_str = False
                esc = False
                while i < n:
                    ch = s[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == "\"":
                            in_str = False
                    else:
                        if ch == "\"":
                            in_str = True
                        elif ch in "{[":
                            stack.append(ch)
                        elif ch in "}]":
                            if not stack:
                                return -1
                            opener = stack.pop()
                            if (opener == "{" and ch != "}") or (opener == "[" and ch != "]"):
                                return -1
                            if not stack:
                                return i + 1
                    i += 1
                return -1

            def try_parse_one(buf):
                b = buf.lstrip("\r\n")
                if not b:
                    return None
                i1 = b.find(";")
                if i1 < 0:
                    return None
                i2 = b.find(";", i1 + 1)
                if i2 < 0:
                    return None
                req_type = b[:i1].strip().upper()
                param = b[i1 + 1 : i2].strip()
                rest = b[i2 + 1 :]

                arg = None
                # Many clients send a JSON object as the 3rd field (especially for OFFLINE).
                # Consume it (and pass it to handler) when present to prevent buffer leftovers.
                j_end = scan_json_end(rest, 0)
                if j_end >= 0:
                    arg = rest[:j_end].strip()
                    rest = rest[j_end:]
                elif req_type == "OFFLINE":
                    # OFFLINE requires a JSON payload; if incomplete, wait for more data.
                    return None

                rest = rest.lstrip("\r\n")
                return req_type, param, arg, rest

            def handle_one(req_type, param, arg):
                response = None
                code = '31415'
                if param != code:
                    self.logger.info("密码错误")
                    response = '密码错误'
                else:
                    if req_type == 'OFFLINE':
                        try:
                            self.logger.info(req_type + (arg or ""))
                            response = self.get_offline(arg)
                        except Exception as e:
                            self.logger.error("offline返回错误:如下")
                            self.logger.error(e)
                            try:
                                self._pdbg(f"OFFLINE failed: raw_arg={arg!r}, buffer_tail={buffer[-120:]!r}")
                            except Exception:
                                pass
                            response = None
                    elif req_type == 'ONLINE':
                        try:
                            online_seq = self._next_online_request_seq()
                            online_request_ts_ms = int(time.time() * 1000)
                            cached_measures = self.ocrserver.get_measures()
                            self.logger.info(req_type)
                            self.logger.info(
                                f"[ONLINE-WAIT] seq={online_seq}, "
                                f"request_ts_ms={online_request_ts_ms}, "
                                f"cached_capture_ts_ms={cached_measures.get('CaptureTimestamp')}, "
                                f"cached_recognize_start_ts_ms={cached_measures.get('RecognizeStartTimestamp')}, "
                                f"action=request_received"
                            )
                            response, wait_result = self.get_online(online_request_ts_ms=online_request_ts_ms)
                            self.logger.info(
                                f"[ONLINE-WAIT] seq={online_seq}, "
                                f"request_ts_ms={online_request_ts_ms}, "
                                f"cached_capture_ts_ms={wait_result.get('capture_ts_ms')}, "
                                f"cached_recognize_start_ts_ms={wait_result.get('recognize_start_ts_ms')}, "
                                f"action={wait_result.get('action')}, "
                                f"wait_elapsed_ms={wait_result.get('wait_elapsed_ms')}, "
                                f"had_cache={wait_result.get('had_cache')}, "
                                f"fallback={wait_result.get('fallback')}"
                            )
                            self.logger.info(f"ONLINE response: {response}")
                        except Exception as e:
                            self.logger.error("online error:")
                            self.logger.error(e)
                            response = None
                    elif req_type == 'CLOSEOCR':
                        self.close_ocr_server()
                        response = {'success': True,  'info': "close ocr successfully"}
                    elif req_type == 'OPENOCR':
                        self.open_ocr_server()
                        response = {'success': True, 'info': "open ocr successfully"}
                    elif req_type == 'OCR':
                        response = {'success': True, 'health': self.ocrserver.get_health()}
                    elif req_type == 'CLOSE':
                        self.close_ocr_server()
                        response = {'success': True,  'info': "close ocr successfully"}
                    else:
                        response = {'success': False, 'info': f"错误: 未知请求类型 '{req_type}'。支持的类型: {', '.join(self.REQUEST_TYPES.keys())}"}

                if response:
                    response = json.dumps(response)
                    client_socket.sendall((response + "\n").encode('utf-8'))

            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                buffer += chunk.decode('utf-8', errors='replace')

                # Preferred: newline-delimited requests
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(";")
                    rt = parts[0].strip().upper() if len(parts) > 0 else ""
                    pm = parts[1].strip() if len(parts) > 1 else ""
                    ar = parts[2].strip() if len(parts) > 2 else None
                    handle_one(rt, pm, ar)

                # Best-effort: concatenated OFFLINE requests without '\n'
                while True:
                    parsed = try_parse_one(buffer)
                    if not parsed:
                        break
                    rt, pm, ar, rest = parsed
                    buffer = rest
                    handle_one(rt, pm, ar)


        except Exception as e:
            self.logger.error(f"处理客户�?{client_address} 时发生错�? {e}")
        finally:
            try:
                client_socket.close()
            except Exception:
                pass
            try:
                with self._client_state_lock:
                    self._active_client_connections = max(0, int(self._active_client_connections) - 1)
                    self.logger.info(
                        f"client released: addr={client_address}, active_conn={self._active_client_connections}/{self._max_client_connections}"
                    )
            except Exception:
                pass
            print(f"connection closed: {client_address}")

    def start_server(self, host='localhost', port=12345):
        """Start TCP server."""
        # 创建TCP套接�?
        # self.logger.info(f"start server: host={host}, port={port}")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            # 设置套接字选项，�
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # 绑定地址和端�?
            server_socket.bind((host, port))

            # 开始监听，最大连接数�?
            server_socket.listen(5)
            print(f"服务器已启动，监听地址: {host}:{port}")
            print(f"支持的请求类�? {', '.join([f'{k}({v})' for k, v in self.REQUEST_TYPES.items()])}")
            self.logger.info(f"服务器已启动，监听地址: {host}:{port}")

            try:
                while True:
                    # 接受客户端连�?
                    client_socket, client_address = server_socket.accept()
                    reject = False
                    try:
                        with self._client_state_lock:
                            if int(self._active_client_connections) >= int(self._max_client_connections):
                                reject = True
                            else:
                                self._active_client_connections += 1
                                self.logger.info(
                                    f"client accepted: addr={client_address}, active_conn={self._active_client_connections}/{self._max_client_connections}"
                                )
                    except Exception:
                        pass

                    if reject:
                        try:
                            self.logger.warning(
                                f"too many client connections ({self._active_client_connections}/{self._max_client_connections}), rejecting: {client_address}"
                            )
                        except Exception:
                            pass
                        try:
                            client_socket.close()
                        except Exception:
                            pass
                        continue

                    if self.client_thread is not None:
                        del self.client_thread

                    # 为每个客户端创建一个新线程
                    # client_thread = multiprocessing.Process(
                    self.client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    self.client_thread.start()
                    # self.client_thread.join()

            except KeyboardInterrupt:
                print("\n服务器已停止")
                self.logger.info("\n服务器已停止")


def run(host="localhost", port=30415):
    imgProcess = ImageProcessServer()
    imgProcess.start_server(host=host, port=port)

if __name__ == "__main__":
    run("127.0.0.1")

