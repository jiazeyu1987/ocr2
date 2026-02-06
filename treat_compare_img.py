# encoding=utf-8
import os
import sqlite3
import time
from datetime import datetime
from typing import Optional
import json

import cv2
import image_difference
import numpy as np
import pyautogui
import threading
from pathlib import Path

try:
    from screenshot_lock import SCREENSHOT_LOCK
except Exception:  # pragma: no cover
    # Backward-compatible fallback: allow running even if `screenshot_lock.py` was not deployed.
    # This will lose cross-thread screenshot serialization, but avoids hard crash.
    SCREENSHOT_LOCK = threading.Lock()

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import simplefem_focus
except Exception:  # pragma: no cover
    simplefem_focus = None


# Module-level cache: when focus detection fails for a session, reuse last successful location.
_LAST_FOCUS_ANCHOR = None  # type: ignore
_LAST_ROI2_RECT = None  # type: ignore
_LAST_ROI3_RECT = None  # type: ignore


class ComparePoints:
    def __init__(self, setting=None, logger=None):
        if setting is None:
            setting = {}

        self._raw_setting = dict(setting) if isinstance(setting, dict) else {}
        self.logger = logger

        self._debug_log_enabled = False
        self._debug_every_n_frames = 60
        try:
            dbg_cfg = (self._raw_setting or {}).get("offline_debug_log")
            if isinstance(dbg_cfg, dict):
                self._debug_log_enabled = bool(dbg_cfg.get("enabled", False))
                self._debug_every_n_frames = int(dbg_cfg.get("every_n_frames", self._debug_every_n_frames))
            else:
                self._debug_log_enabled = bool(dbg_cfg)
        except Exception:
            self._debug_log_enabled = False

        # Verbose diagnostics controlled by settings["peak_debug_log"]["enabled"].
        self._peak_debug_enabled = False
        try:
            pdl = (self._raw_setting or {}).get("peak_debug_log")
            if isinstance(pdl, dict):
                self._peak_debug_enabled = bool(pdl.get("enabled", False))
            else:
                self._peak_debug_enabled = bool(pdl)
        except Exception:
            self._peak_debug_enabled = False

        self.setting = {
            "width_x": setting["width_x"] if "width_x" in setting else 2,
            "height_y": setting["height_y"] if "height_y" in setting else 4,
            "binary_threshold": setting["binary_threshold"] if "binary_threshold" in setting else 10,
            "drawcontour": setting["drawcontour"] if "drawcontour" in setting else False,
            "if_align": setting["if_align"] if "if_align" in setting else False,
        }

        self._capture_roi = None  # (x1, y1, x2, y2) in screen coordinates
        self._roi_offset = (0, 0)  # (x1, y1) when capture_roi enabled
        try:
            roi_cfg = (setting or {}).get("roi1_capture") or (setting or {}).get("capture_roi") or {}
            if isinstance(roi_cfg, dict) and bool(roi_cfg.get("enabled", False)):
                x1 = int(roi_cfg.get("x1", 0))
                y1 = int(roi_cfg.get("y1", 0))
                x2 = int(roi_cfg.get("x2", 0))
                y2 = int(roi_cfg.get("y2", 0))
                if x2 > x1 and y2 > y1:
                    self._capture_roi = (x1, y1, x2, y2)
                    self._roi_offset = (x1, y1)
                    if self.logger:
                        self.logger.info(f"Offline capture ROI enabled: {self._capture_roi}")
        except Exception:
            if self.logger:
                self.logger.exception("Failed to parse roi1_capture config; using legacy capture region")

        self.point_id = None
        self.save_point_id = None
        self.is_save = True
        self.db_dir = "D:/software_data"
        self.response = {"success": True}

        self._stop_event = None
        # Lifecycle signals for server-side coordination.
        # `_capture_done_event` becomes set as soon as the screenshot loop ends (so a new OFFLINE session can start),
        # even if this thread is still doing slow IO (saving images / DB insert).
        self._capture_done_event = threading.Event()
        self._finished_event = threading.Event()
        # Session timing / stage markers for diagnostics.
        self._session_start_ts = None
        self._stop_requested_ts = None
        self._stage = "init"
        self._stage_detail = None

        self.compare_before = None
        self.compare_after = None
        self.before_name = ""
        self.after_name = ""

        # Optional: save all frames between before->after into tmp dir for debugging.
        self._tmp_frames_enabled = False
        self._tmp_frames_dir = r"D:/software_data/tmp"
        self._tmp_max_buffer_frames = 2500
        try:
            tmp_cfg = (self._raw_setting or {}).get("offline_tmp_frames")
            if isinstance(tmp_cfg, dict):
                self._tmp_frames_enabled = bool(tmp_cfg.get("enabled", False))
                if tmp_cfg.get("dir"):
                    self._tmp_frames_dir = str(tmp_cfg.get("dir"))
                if tmp_cfg.get("max_buffer_frames") is not None:
                    self._tmp_max_buffer_frames = int(tmp_cfg.get("max_buffer_frames"))
            else:
                self._tmp_frames_enabled = bool(tmp_cfg)
        except Exception:
            self._tmp_frames_enabled = False

        self._tmp_session_dir = None
        self._tmp_session_id = None
        self._tmp_day_dir = None
        self._tmp_meta_path = None
        self._tmp_frame_buffer = []

        # SimpleFEM-style focus/ROI2/ROI3 (cached per OFFLINE session)
        self._sf_enabled = False
        self._sf_roi2_ext = {"left": 40, "right": 40, "top": 50, "bottom": 30}
        self._sf_roi3_ext = {"left": 30, "right": 30, "top": 50, "bottom": 100}
        self._sf_difference_threshold = 0.5
        # ROI3 override rules (SimpleFEM-style): can override ROI2 "red" -> "green"
        self._sf_roi3_g1_g2_override = {"enabled": True, "g1_threshold": 98.0, "g2_threshold": 20.0, "use_peak_max": True}
        self._sf_roi3_column_diff_override = {"enabled": True, "g1_threshold": 99.0, "threshold": 15.0, "use_peak_max": True}
        self._sf_debug_log_enabled = False
        try:
            sf_cfg = (self._raw_setting or {}).get("peak_detect") or {}
            if isinstance(sf_cfg, dict):
                self._sf_enabled = bool(sf_cfg.get("enabled", False))
                if isinstance(sf_cfg.get("roi2_extension_params"), dict):
                    self._sf_roi2_ext.update(sf_cfg.get("roi2_extension_params") or {})
                if isinstance(sf_cfg.get("roi3_extension_params"), dict):
                    self._sf_roi3_ext.update(sf_cfg.get("roi3_extension_params") or {})
                if sf_cfg.get("difference_threshold") is not None:
                    self._sf_difference_threshold = float(sf_cfg.get("difference_threshold"))

                # ROI3 override configs (accept both keys for compatibility)
                g1g2 = sf_cfg.get("roi3_g1_g2_override") or sf_cfg.get("g1_g2_override")
                if isinstance(g1g2, dict):
                    self._sf_roi3_g1_g2_override.update(g1g2 or {})

                col = sf_cfg.get("roi3_column_diff_override")
                if isinstance(col, dict):
                    self._sf_roi3_column_diff_override.update(col or {})

            dbg_cfg2 = (self._raw_setting or {}).get("peak_debug_log")
            if isinstance(dbg_cfg2, dict):
                self._sf_debug_log_enabled = bool(dbg_cfg2.get("enabled", False))
            else:
                self._sf_debug_log_enabled = bool(dbg_cfg2)
        except Exception:
            self._sf_enabled = False
            self._sf_debug_log_enabled = False

        self._sf_anchor = None
        self._sf_roi2_rect = None
        self._sf_roi3_rect = None
        self._sf_roi2_before_mean = None
        self._sf_roi2_after_mean = None
        self._sf_roi2_diff = None
        self._sf_color = None
        self._sf_base_color = None
        self._sf_green_case = None  # 1/2/3 when final green, else None
        self._sf_roi3_g1 = None
        self._sf_roi3_g2 = None
        self._sf_roi3_column_diff = None
        self._sf_roi3_override_applied = False
        self._sf_roi3_override_method = None
        self._sf_roi3_override_frame_index = None
        self._sf_roi3_override_tag = None

    def _dbg(self, msg):
        if self._debug_log_enabled and self.logger:
            self.logger.info(f"[offline] {msg}")

    def _pdbg(self, msg):
        if self._peak_debug_enabled and self.logger:
            self.logger.info(f"[peakdbg] {msg}")

    def _sf_dbg(self, msg):
        if self._sf_debug_log_enabled and self.logger:
            self.logger.info(f"[peak] {msg}")

    def _tmp_dbg(self, msg):
        if self._tmp_frames_enabled and self.logger:
            self.logger.info(f"[tmp] {msg}")

    def _tmp_init_session(self, point_id: Optional[int], before_ts: str):
        if not self._tmp_frames_enabled:
            return
        try:
            pid = str(point_id) if point_id is not None else "unknown"
            self._tmp_session_id = f"{pid}_{before_ts}"
            day = before_ts.split("_", 1)[0]  # YYYY-MM-DD from convert_timestamp2str
            self._tmp_day_dir = str(Path(self._tmp_frames_dir) / day)
            self._tmp_session_dir = str(Path(self._tmp_day_dir) / self._tmp_session_id)
            Path(self._tmp_session_dir).mkdir(parents=True, exist_ok=True)
            self._tmp_meta_path = str(Path(self._tmp_day_dir) / "offline_frames_meta.jsonl")
            self._tmp_frame_buffer = []
            self._tmp_dbg(f"enabled: day_dir={self._tmp_day_dir} session_dir={self._tmp_session_dir}")
        except Exception as e:
            self._tmp_frames_enabled = False
            self._tmp_session_dir = None
            self._tmp_day_dir = None
            self._tmp_meta_path = None
            self._tmp_frame_buffer = []
            self._tmp_dbg(f"disabled due to init error: {e}")

    def _tmp_record_frame(self, frame_rgb: np.ndarray, ts: datetime, frame_index: int, tag: str, roi1_gray=None):
        """
        First OFFLINE signal: start reading/buffering.
        Second OFFLINE signal (stop): flush to disk.
        """
        if not self._tmp_frames_enabled or not self._tmp_session_dir:
            return
        try:
            if len(self._tmp_frame_buffer) >= int(self._tmp_max_buffer_frames):
                self._tmp_frame_buffer.pop(0)

            item = {
                "frame_index": int(frame_index),
                "ts": self.convert_timestamp2str(ts),
                "tag": str(tag),
                "roi1_gray": float(roi1_gray) if roi1_gray is not None else None,
                "roi2_mean": None,
                "roi3_mean": None,
            }
            if self._sf_enabled and self._sf_roi2_rect is not None:
                item["roi2_mean"] = self._sf_gray_mean(frame_rgb, self._sf_roi2_rect)
            if self._sf_enabled and self._sf_roi3_rect is not None:
                item["roi3_mean"] = self._sf_gray_mean(frame_rgb, self._sf_roi3_rect)

            item["_frame_rgb"] = frame_rgb  # keep for later write
            self._tmp_frame_buffer.append(item)
        except Exception as e:
            self._tmp_dbg(f"record failed: frame={frame_index}, tag={tag}, err={e}")

    def _tmp_append_meta(self, obj: dict):
        if not self._tmp_frames_enabled or not self._tmp_meta_path:
            return
        try:
            Path(self._tmp_day_dir).mkdir(parents=True, exist_ok=True)
            with open(self._tmp_meta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception as e:
            self._tmp_dbg(f"append meta failed: {e}")

    def _tmp_flush_on_stop(self, point_id: Optional[int]):
        if not self._tmp_frames_enabled or not self._tmp_session_dir:
            return
        if not self._tmp_frame_buffer:
            return
        try:
            before_event_written = False
            after_event_written = False
            for item in self._tmp_frame_buffer:
                frame_rgb = item.pop("_frame_rgb", None)
                if frame_rgb is None:
                    continue

                ts = item.get("ts", "")
                tag = item.get("tag", "frame")
                idx = int(item.get("frame_index", 0))

                name = f"{idx:05d}_{ts}_{tag}.png".replace(":", "-")
                out_path = str(Path(self._tmp_session_dir) / name)
                cv2.imwrite(out_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

                # Keep JSONL minimal: only filename + ROI2/ROI3 means.
                # Events ("before saved", "after saved") are represented by filename suffix: *_before.png / *_after_*.png
                self._tmp_append_meta(
                    {
                        "filename": os.path.basename(out_path),
                        "roi1_mean": item.get("roi1_gray"),
                        "roi2_mean": item.get("roi2_mean"),
                        "roi3_mean": item.get("roi3_mean"),
                    }
                )

                # Optional explicit event lines (different schema allowed):
                # - before_saved: first frame tagged "before"
                # - after_saved: first frame tagged "after_*"
                # These lines help downstream tooling without breaking the minimal-per-frame records.
                try:
                    if (not before_event_written) and str(tag) == "before":
                        self._tmp_append_meta(
                            {
                                "event": "before_saved",
                                "filename": os.path.basename(out_path),
                                "point_id": point_id,
                                "session": self._tmp_session_id,
                                "frame_index": idx,
                                "ts": ts,
                            }
                        )
                        before_event_written = True
                    if (not after_event_written) and str(tag).startswith("after"):
                        self._tmp_append_meta(
                            {
                                "event": "after_saved",
                                "filename": os.path.basename(out_path),
                                "point_id": point_id,
                                "session": self._tmp_session_id,
                                "frame_index": idx,
                                "ts": ts,
                                "tag": tag,
                            }
                        )
                        after_event_written = True
                except Exception:
                    # Never break frame flushing for event logging
                    pass
            self._tmp_dbg(f"flush done: frames={len(self._tmp_frame_buffer)} dir={self._tmp_session_dir}")
        except Exception as e:
            self._tmp_dbg(f"flush failed: {e}")
        finally:
            self._tmp_frame_buffer = []

    def _sf_gray_mean(self, frame_rgb: np.ndarray, rect):
        if frame_rgb is None or rect is None:
            return None
        x1, y1, x2, y2 = rect
        h, w = frame_rgb.shape[:2]
        x1 = max(0, min(int(x1), w))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        return float(np.mean(gray))

    def _sf_roi3_metrics(self, frame_rgb: np.ndarray):
        """
        Compute ROI3 metrics on a single frame using current ROI3 rect:
        - roi3_mean: mean grayscale (0-255)
        - g1: % pixels in [80,255]
        - g2: % pixels in [150,255]
        - column_diff: max(column_mean) - min(column_mean)
        """
        if frame_rgb is None or self._sf_roi3_rect is None:
            return {"roi3_mean": None, "g1": None, "g2": None, "column_diff": None}

        x1, y1, x2, y2 = self._sf_roi3_rect
        h, w = frame_rgb.shape[:2]
        x1 = max(0, min(int(x1), w))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return {"roi3_mean": None, "g1": None, "g2": None, "column_diff": None}

        roi = frame_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return {"roi3_mean": None, "g1": None, "g2": None, "column_diff": None}

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        total = int(gray.size)
        if total <= 0:
            return {"roi3_mean": None, "g1": None, "g2": None, "column_diff": None}

        roi3_mean = float(np.mean(gray))

        # Histogram-based counts for ranges (matches SimpleFEM image_metrics.py)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
        g1_count = float(np.sum(hist[80:256]))
        g2_count = float(np.sum(hist[150:256]))
        g1 = float((g1_count / total) * 100.0)
        g2 = float((g2_count / total) * 100.0)

        # Column mean diff (matches SimpleFEM compute_roi3_column_mean_diff)
        col_means = np.mean(gray.astype(np.float32), axis=0)
        column_diff = float(np.max(col_means) - np.min(col_means)) if col_means.size else 0.0

        return {"roi3_mean": roi3_mean, "g1": g1, "g2": g2, "column_diff": column_diff}

    def _sf_select_frame_by_roi3_g1_peak(self):
        """
        Select a frame for ROI3 override by maximizing ROI3 G1 (% pixels in [80,255]).
        Requires offline_tmp_frames enabled so we have an in-memory buffer.
        """
        if not self._tmp_frames_enabled or not self._tmp_frame_buffer:
            return None

        best = None
        best_g1 = None

        for item in self._tmp_frame_buffer:
            frame_rgb = item.get("_frame_rgb")
            if frame_rgb is None:
                continue
            m = self._sf_roi3_metrics(frame_rgb)
            g1 = m.get("g1")
            if g1 is None:
                continue
            if best_g1 is None or float(g1) > float(best_g1):
                best_g1 = float(g1)
                best = (frame_rgb, int(item.get("frame_index", 0)), str(item.get("tag", "frame")))

        return best

    def _sf_apply_roi3_overrides(self):
        """
        Apply ROI3-based override rules (SimpleFEM style):
        - Only when base ROI2 result is "red".
        - Rule1 (G1/G2): if G1>g1_threshold and G2>g2_threshold => red->green
        - Rule2 (Column diff): if G1>g1_threshold and column_diff>threshold => red->green
        Frame selection:
        - if use_peak_max and offline_tmp_frames is enabled => use frame with max ROI3 G1
        - else => use AFTER frame
        """
        if not self._sf_enabled or self._sf_color != "red" or self._sf_roi3_rect is None:
            return

        # Reset per-run fields
        self._sf_roi3_g1 = None
        self._sf_roi3_g2 = None
        self._sf_roi3_column_diff = None
        self._sf_roi3_override_applied = False
        self._sf_roi3_override_method = None
        self._sf_roi3_override_frame_index = None
        self._sf_roi3_override_tag = None

        def metrics_on_selected_frame(use_peak_max: bool):
            if use_peak_max:
                sel = self._sf_select_frame_by_roi3_g1_peak()
                if sel is not None:
                    frame_rgb, frame_index, tag = sel
                    m = self._sf_roi3_metrics(frame_rgb)
                    return m, frame_index, tag
            # fallback to after
            if self.compare_after is None:
                return {"roi3_mean": None, "g1": None, "g2": None, "column_diff": None}, None, None
            m = self._sf_roi3_metrics(self.compare_after)
            return m, None, "after"

        # 1) G1/G2 override (configurable thresholds)
        try:
            g1g2_conf = dict(self._sf_roi3_g1_g2_override or {})
            g1g2_enabled = bool(g1g2_conf.get("enabled", True))
            g1_thr = float(g1g2_conf.get("g1_threshold", 98.0))
            g2_thr = float(g1g2_conf.get("g2_threshold", 20.0))
            g1g2_use_peak_max = bool(g1g2_conf.get("use_peak_max", True))

            m, frame_index, tag = metrics_on_selected_frame(g1g2_use_peak_max)
            g1 = m.get("g1")
            g2 = m.get("g2")
            self._sf_roi3_g1 = g1
            self._sf_roi3_g2 = g2
            self._sf_roi3_column_diff = m.get("column_diff")

            if g1g2_enabled and g1 is not None and g2 is not None:
                self._sf_dbg(
                    f"roi3 g1/g2 check: g1={float(g1):.2f}% (>{g1_thr}%), g2={float(g2):.2f}% (>{g2_thr}%), "
                    f"use_peak_max={g1g2_use_peak_max}, selected_frame={frame_index}, tag={tag}"
                )
                if float(g1) > g1_thr and float(g2) > g2_thr:
                    self._sf_color = "green"
                    self._sf_green_case = 2
                    self._sf_roi3_override_applied = True
                    self._sf_roi3_override_method = "roi3_g1_g2"
                    self._sf_roi3_override_frame_index = frame_index
                    self._sf_roi3_override_tag = tag
                    self._sf_dbg(f"[roi3 override] RED->GREEN by G1/G2: g1={float(g1):.2f}%, g2={float(g2):.2f}%")
                    return
        except Exception as e:
            self._sf_dbg(f"roi3 g1/g2 override error: {e}")

        # 2) Column diff override (configurable thresholds; requires G1 + column_diff)
        try:
            col_conf = dict(self._sf_roi3_column_diff_override or {})
            col_enabled = bool(col_conf.get("enabled", True))
            col_g1_thr = float(col_conf.get("g1_threshold", 99.0))
            col_thr = float(col_conf.get("threshold", 15.0))
            col_use_peak_max = bool(col_conf.get("use_peak_max", True))

            m, frame_index, tag = metrics_on_selected_frame(col_use_peak_max)
            g1 = m.get("g1")
            column_diff = m.get("column_diff")
            # keep last computed metrics for response (even if not applied)
            self._sf_roi3_g1 = g1
            self._sf_roi3_g2 = m.get("g2")
            self._sf_roi3_column_diff = column_diff

            if col_enabled and g1 is not None and column_diff is not None:
                self._sf_dbg(
                    f"roi3 column_diff check: g1={float(g1):.2f}% (>{col_g1_thr}%), "
                    f"column_diff={float(column_diff):.2f} (>{col_thr}), use_peak_max={col_use_peak_max}, "
                    f"selected_frame={frame_index}, tag={tag}"
                )
                if float(g1) > col_g1_thr and float(column_diff) > col_thr:
                    self._sf_color = "green"
                    self._sf_green_case = 3
                    self._sf_roi3_override_applied = True
                    self._sf_roi3_override_method = "roi3_column_diff"
                    self._sf_roi3_override_frame_index = frame_index
                    self._sf_roi3_override_tag = tag
                    self._sf_dbg(f"[roi3 override] RED->GREEN by column_diff: g1={float(g1):.2f}%, column_diff={float(column_diff):.2f}")
        except Exception as e:
            self._sf_dbg(f"roi3 column_diff override error: {e}")

    def _sf_locate_roi2_roi3_on_before(self, before_rgb: np.ndarray):
        """
        Locate anchor + ROI2/ROI3 on the BEFORE frame once per OFFLINE session.
        If focus fails, reuse module-level last successful rectangles.
        """
        global _LAST_FOCUS_ANCHOR, _LAST_ROI2_RECT, _LAST_ROI3_RECT

        if not self._sf_enabled:
            return False

        if simplefem_focus is None:
            if self.logger:
                self.logger.error("peak_detect enabled but simplefem_focus is not available")
            return False

        if before_rgb is None:
            return False

        h, w = before_rgb.shape[:2]
        self._sf_dbg(f"before image size: w={w}, h={h}")

        anchor = None
        try:
            before_bgr = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR)
            anchor = simplefem_focus.detect_green_intersection(before_bgr)
        except Exception as e:
            self._sf_dbg(f"focus detect error: {e}")
            anchor = None

        used_last = False
        if anchor is None:
            if _LAST_ROI2_RECT is not None:
                self._sf_anchor = _LAST_FOCUS_ANCHOR
                self._sf_roi2_rect = _LAST_ROI2_RECT
                self._sf_roi3_rect = _LAST_ROI3_RECT
                used_last = True
                self._sf_dbg(f"focus not found; reuse last: anchor={self._sf_anchor}, roi2_rect={self._sf_roi2_rect}, roi3_rect={self._sf_roi3_rect}")
                return True
            self._sf_dbg("focus not found; no last cached focus to reuse")
            return False

        # Compute ROI2/ROI3 based on anchor and extension params (ROI1 is the captured before image).
        roi2_rect = None
        roi3_rect = None
        try:
            roi2_rect = simplefem_focus.compute_roi2_region((w, h), anchor, self._sf_roi2_ext)
            roi3_rect = simplefem_focus.compute_roi2_region((w, h), anchor, self._sf_roi3_ext)
        except Exception as e:
            self._sf_dbg(f"compute_roi2_region failed: {e}")
            roi2_rect = None
            roi3_rect = None

        if roi2_rect is None:
            if _LAST_ROI2_RECT is not None:
                self._sf_anchor = _LAST_FOCUS_ANCHOR
                self._sf_roi2_rect = _LAST_ROI2_RECT
                self._sf_roi3_rect = _LAST_ROI3_RECT
                used_last = True
                self._sf_dbg(f"roi2_rect invalid; reuse last: anchor={self._sf_anchor}, roi2_rect={self._sf_roi2_rect}, roi3_rect={self._sf_roi3_rect}")
                return True
            self._sf_dbg(f"roi2_rect invalid and no last cached rects; anchor={anchor}")
            return False

        self._sf_anchor = anchor
        self._sf_roi2_rect = roi2_rect
        self._sf_roi3_rect = roi3_rect

        _LAST_FOCUS_ANCHOR = anchor
        _LAST_ROI2_RECT = roi2_rect
        _LAST_ROI3_RECT = roi3_rect

        self._sf_dbg(f"focus ok: anchor={anchor}, roi2_rect={roi2_rect}, roi3_rect={roi3_rect}, used_last={used_last}")
        return True

    def compute_grayscale_v2(self, ultra_img):
        # The legacy algorithm assumes a full-height capture (1080p) and computes mean grayscale
        # in a fixed vertical band. If capture ROI is enabled, remap that band into ROI-local coords.
        h, w = ultra_img.shape[:2]

        base_row_start = 256
        base_row_end = 808

        oy = self._roi_offset[1]
        row_start = base_row_start - oy
        row_end = base_row_end - oy

        row_start = max(0, min(int(row_start), h))
        row_end = max(0, min(int(row_end), h))

        if row_end <= row_start or w <= 0:
            # ROI doesn't cover the legacy band; fall back to whole-frame mean to avoid OpenCV size asserts.
            self._dbg(
                f"compute_grayscale_v2: ROI band out of bounds (h={h}, w={w}, oy={oy}, band=({row_start},{row_end})); "
                "fallback to whole-frame mean"
            )
            return float(cv2.mean(ultra_img)[0])

        roi = ultra_img[row_start:row_end, :]
        mask = np.ones(roi.shape[:2], dtype=np.uint8)
        return float(cv2.mean(roi, mask)[0])

    def inser_info_database(self, db_dir, id, before_path, after_path):
        dbpath = db_dir + "/ccwssm"
        backup_dbpath = db_dir + "/zccwssm"

        db = sqlite3.connect(dbpath, check_same_thread=False, timeout=30)
        db_backup = sqlite3.connect(backup_dbpath, check_same_thread=False, timeout=30)

        modifytime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")[:-3]

        sql_sentence = """
            UPDATE SegmentImagesInfo
            SET ImagePath = ?, ModifyTime = ?
            WHERE ID = ?
            """

        if self.logger:
            self.logger.info(f"{before_path};{after_path};{modifytime};{id}")

        image_path = before_path + ";" + after_path + ";" + after_path.replace("_after", "_diff")

        db.cursor().execute(sql_sentence, (image_path, modifytime, id))
        db_backup.cursor().execute(sql_sentence, (image_path, modifytime, id))

        db.commit()
        db_backup.commit()

        db.cursor().close()
        db_backup.cursor().close()

    def convert_timestamp2str(self, timestamp):
        return timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]

    def _draw_fem_result_on_diff(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Draw SimpleFEM (roi2_color) result text onto the diff image.
        - green -> '绿色成功'
        - red   -> '红色失败'

        Uses PIL to render Chinese when available; falls back to ASCII text if font is missing.
        """
        if not self._sf_enabled:
            return bgr_img
        # When metrics are missing (e.g. focus/ROI not found), still render the 4-line overlay
        # to make debugging easier. Treat unknown as "red" for line1.
        color = (self._sf_color or "").lower()
        if color not in ("green", "red"):
            color = "red"

        # Ensure we have a 3-channel image for consistent drawing (PIL and OpenCV).
        try:
            if bgr_img is not None and getattr(bgr_img, "ndim", 0) == 2:
                bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)
        except Exception:
            # If conversion fails, keep original and rely on outer try/except fallbacks.
            pass

        # Build four per-condition lines (each line has its own color).
        # 1) Final success/fail
        # 2) ROI2 diff check
        # 3) ROI3 G1/G2 override check
        # 4) ROI3 column_diff override check
        roi2_diff = self._sf_roi2_diff
        thr = float(self._sf_difference_threshold) if self._sf_difference_threshold is not None else None
        roi2_pass = False
        if roi2_diff is not None and thr is not None:
            try:
                roi2_pass = float(roi2_diff) >= float(thr)
            except Exception:
                roi2_pass = False

        g1 = self._sf_roi3_g1
        g2 = self._sf_roi3_g2
        col = self._sf_roi3_column_diff

        g1g2_conf = dict(self._sf_roi3_g1_g2_override or {})
        g1g2_enabled = bool(g1g2_conf.get("enabled", True))
        g1_thr = float(g1g2_conf.get("g1_threshold", 98.0))
        g2_thr = float(g1g2_conf.get("g2_threshold", 20.0))
        g1g2_pass = False
        if g1g2_enabled and g1 is not None and g2 is not None:
            try:
                g1g2_pass = float(g1) > float(g1_thr) and float(g2) > float(g2_thr)
            except Exception:
                g1g2_pass = False

        col_conf = dict(self._sf_roi3_column_diff_override or {})
        col_enabled = bool(col_conf.get("enabled", True))
        col_g1_thr = float(col_conf.get("g1_threshold", 99.0))
        col_thr = float(col_conf.get("threshold", 15.0))
        col_pass = False
        if col_enabled and g1 is not None and col is not None:
            try:
                col_pass = float(g1) > float(col_g1_thr) and float(col) > float(col_thr)
            except Exception:
                col_pass = False

        line1_cn = f"1. {'成功' if color == 'green' else '失败'}"
        line1_en = f"1. {'OK' if color == 'green' else 'FAIL'}"

        if roi2_diff is None or thr is None:
            line2_cn = "2. ROI2: diff/threshold=N/A"
            line2_en = "2. ROI2: diff/threshold=N/A"
        else:
            line2_cn = f"2. ROI2: (after-before)={float(roi2_diff):.3f} / {float(thr):.3f}"
            line2_en = f"2. ROI2: d={float(roi2_diff):.3f} / thr={float(thr):.3f}"

        if not g1g2_enabled:
            line3_cn = f"3. ROI3(G1/G2): disabled"
            line3_en = f"3. ROI3(G1/G2): disabled"
        elif g1 is None or g2 is None:
            line3_cn = "3. ROI3(G1/G2): N/A"
            line3_en = "3. ROI3(G1/G2): N/A"
        else:
            line3_cn = f"3. ROI3: G1={float(g1):.2f}/{float(g1_thr):.2f}  G2={float(g2):.2f}/{float(g2_thr):.2f}"
            line3_en = f"3. ROI3: G1={float(g1):.2f}/{float(g1_thr):.2f}  G2={float(g2):.2f}/{float(g2_thr):.2f}"

        if not col_enabled:
            line4_cn = "4. ROI3(colDiff): disabled"
            line4_en = "4. ROI3(colDiff): disabled"
        elif g1 is None or col is None:
            line4_cn = "4. ROI3(colDiff): N/A"
            line4_en = "4. ROI3(colDiff): N/A"
        else:
            line4_cn = f"4. ROI3: G1={float(g1):.2f}/{float(col_g1_thr):.2f}  colDiff={float(col):.2f}/{float(col_thr):.2f}"
            line4_en = f"4. ROI3: G1={float(g1):.2f}/{float(col_g1_thr):.2f}  colDiff={float(col):.2f}/{float(col_thr):.2f}"

        lines_cn = [line1_cn, line2_cn, line3_cn, line4_cn]
        lines_en = [line1_en, line2_en, line3_en, line4_en]
        line_ok = [color == "green", roi2_pass, g1g2_pass, col_pass]

        def ok_bgr(ok: bool):
            return (0, 200, 0) if ok else (0, 0, 255)
        def ok_rgb(ok: bool):
            return (0, 200, 0) if ok else (255, 0, 0)

        # Font sizes for labels on the differ/diff image.
        # Keep them small by default to avoid obscuring details; can be overridden via:
        # setting["diff_label"] = {"pil_font_size": 18, "pil_spacing": 2, "cv_scale": 0.34, "cv_thickness": 1, "cv_line_step": 16}
        label_cfg = {}
        try:
            if isinstance(getattr(self, "_raw_setting", None), dict):
                label_cfg = self._raw_setting.get("diff_label") or {}
        except Exception:
            label_cfg = {}
        try:
            pil_font_size = int(label_cfg.get("pil_font_size", 18))
        except Exception:
            pil_font_size = 18
        try:
            pil_spacing = int(label_cfg.get("pil_spacing", 2))
        except Exception:
            pil_spacing = 2
        try:
            cv_scale = float(label_cfg.get("cv_scale", 0.34))
        except Exception:
            cv_scale = 0.34
        try:
            cv_thickness = int(label_cfg.get("cv_thickness", 1))
        except Exception:
            cv_thickness = 1
        try:
            cv_line_step = int(label_cfg.get("cv_line_step", 16))
        except Exception:
            cv_line_step = 16

        if Image is None or ImageDraw is None or ImageFont is None:
            y = 40
            for i, line in enumerate(lines_en):
                cv2.putText(bgr_img, str(line), (20, y), cv2.FONT_HERSHEY_SIMPLEX, cv_scale, ok_bgr(bool(line_ok[i])), cv_thickness)
                y += cv_line_step
            return bgr_img

        # Try common Windows fonts that can render Chinese.
        font_candidates = [
            r"C:\Windows\Fonts\msyh.ttc",   # Microsoft YaHei
            r"C:\Windows\Fonts\msyh.ttf",
            r"C:\Windows\Fonts\simhei.ttf",  # SimHei
            r"C:\Windows\Fonts\simsun.ttc",  # SimSun
        ]
        font = None
        for fp in font_candidates:
            try:
                if os.path.exists(fp):
                    font = ImageFont.truetype(fp, pil_font_size)
                    break
            except Exception:
                font = None

        # Convert to PIL (RGB) for text rendering
        try:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil_img)
            use_cn = font is not None
            lines = lines_cn if use_cn else lines_en
            x = 20
            y = 20
            try:
                bbox = font.getbbox("测试Ag") if font is not None else None
                line_h = int((bbox[3] - bbox[1]) if bbox else (pil_font_size + 2))
            except Exception:
                line_h = int(pil_font_size + 2)
            step = max(1, int(line_h + pil_spacing))
            for i, line in enumerate(lines):
                draw.text((x, y), str(line), font=font, fill=ok_rgb(bool(line_ok[i])))
                y += step
            out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return out
        except Exception:
            y = 40
            for i, line in enumerate(lines_en):
                cv2.putText(bgr_img, str(line), (20, y), cv2.FONT_HERSHEY_SIMPLEX, cv_scale, ok_bgr(bool(line_ok[i])), cv_thickness)
                y += cv_line_step
            return bgr_img

    def write_img(self):
        if self.logger:
            self.logger.info("write_img...")
        try:
            self._stage = "write_img"
            self._stage_detail = "start"
        except Exception:
            pass

        img_dir = "D:/software_data/imgs"

        before_path = f"{img_dir}/{self.before_name}_before.png"
        after_path = f"{img_dir}/{self.after_name}_after.png"

        if not self.is_save:
            before_path = img_dir + "/energy_before.png"
            after_path = img_dir + "/energy_after.png"
            if os.path.exists(before_path):
                os.remove(before_path)
            if os.path.exists(after_path):
                os.remove(after_path)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        if self.compare_before is not None:
            try:
                self._stage_detail = "save_before"
            except Exception:
                pass
            before_bgr = cv2.cvtColor(self.compare_before, cv2.COLOR_RGB2BGR)
            cv2.imwrite(before_path, before_bgr)
        else:
            if self.logger:
                self.logger.info("no compare before")
            before_bgr = None

        if self.compare_after is not None:
            try:
                self._stage_detail = "process_after"
            except Exception:
                pass
            try:
                compare_after = image_difference.process_two_images(
                    cv2.cvtColor(self.compare_before, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(self.compare_after, cv2.COLOR_RGB2BGR),
                    if_align=self.setting["if_align"],
                    binary_threshold=self.setting["binary_threshold"],
                    width_x=self.setting["width_x"],
                    height_y=self.setting["height_y"],
                    drawcontour=self.setting["drawcontour"],
                )
            except Exception as e:
                if self.logger:
                    self.logger.error(f"after processing failed: {e}; fallback to raw after")
                compare_after = None

            if compare_after is None:
                compare_after = cv2.cvtColor(self.compare_after, cv2.COLOR_RGB2BGR)

            cv2.imwrite(after_path, compare_after)
        else:
            if self.logger:
                self.logger.info("no compare after")
            compare_after = None

        if self.compare_before is not None and self.compare_after is not None:
            try:
                self._stage_detail = "save_diff"
            except Exception:
                pass
            direct_diff = np.array(self.compare_after).astype(np.float32) - np.array(self.compare_before).astype(np.float32)
            direct_diff[np.where(direct_diff < 0)] = 0
            direct_diff = direct_diff.astype(np.uint8)
            # Draw FEM result label on the diff image (if enabled and available)
            try:
                if direct_diff.ndim == 3:
                    diff_bgr = cv2.cvtColor(direct_diff, cv2.COLOR_RGB2BGR)
                else:
                    diff_bgr = cv2.cvtColor(direct_diff, cv2.COLOR_GRAY2BGR)
                diff_bgr = self._draw_fem_result_on_diff(diff_bgr)
                diff_path = after_path.replace("_after", "_diff")
                cv2.imwrite(diff_path, diff_bgr)

                # Also save diff/differ image into tmp session folder (if enabled) for easier offline inspection.
                if self._tmp_frames_enabled and self._tmp_session_dir:
                    try:
                        ts = self.after_name or self.before_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
                        name = f"diff_{ts}.png".replace(":", "-")
                        out_path = str(Path(self._tmp_session_dir) / name)
                        cv2.imwrite(out_path, diff_bgr)
                        # Event line (different schema is allowed)
                        self._tmp_append_meta(
                            {
                                "event": "differ_saved",
                                "filename": os.path.basename(out_path),
                                "point_id": self.save_point_id,
                                "session": self._tmp_session_id,
                                "ts": ts,
                            }
                        )
                    except Exception as e:
                        self._tmp_dbg(f"save differ to tmp failed: {e}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"write diff with label failed: {e}; fallback to raw diff")
                diff_path = after_path.replace("_after", "_diff")
                cv2.imwrite(diff_path, direct_diff)
                if self._tmp_frames_enabled and self._tmp_session_dir:
                    try:
                        ts = self.after_name or self.before_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
                        name = f"diff_{ts}.png".replace(":", "-")
                        out_path = str(Path(self._tmp_session_dir) / name)
                        cv2.imwrite(out_path, direct_diff)
                        self._tmp_append_meta(
                            {
                                "event": "differ_saved",
                                "filename": os.path.basename(out_path),
                                "point_id": self.save_point_id,
                                "session": self._tmp_session_id,
                                "ts": ts,
                            }
                        )
                    except Exception as e2:
                        self._tmp_dbg(f"save differ to tmp failed: {e2}")
                try:
                    if direct_diff.ndim == 3:
                        diff_bgr = cv2.cvtColor(direct_diff, cv2.COLOR_RGB2BGR)
                    else:
                        diff_bgr = cv2.cvtColor(direct_diff, cv2.COLOR_GRAY2BGR)
                except Exception:
                    diff_bgr = None
        else:
            if self.logger:
                self.logger.info("no compare diff")
            diff_bgr = None

        # Save canonical BEFORE/AFTER/DIFFER images into the same tmp session folder (if enabled).
        # This makes it easy to locate the final trio without guessing frame indices.
        if self._tmp_frames_enabled and self._tmp_session_dir:
            try:
                ts = self.after_name or self.before_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]

                if before_bgr is not None:
                    p = str(Path(self._tmp_session_dir) / "final_before.png")
                    cv2.imwrite(p, before_bgr)
                    self._tmp_append_meta(
                        {
                            "event": "final_before_saved",
                            "filename": os.path.basename(p),
                            "point_id": self.save_point_id,
                            "session": self._tmp_session_id,
                            "ts": ts,
                        }
                    )

                if compare_after is not None:
                    p = str(Path(self._tmp_session_dir) / "final_after.png")
                    cv2.imwrite(p, compare_after)
                    self._tmp_append_meta(
                        {
                            "event": "final_after_saved",
                            "filename": os.path.basename(p),
                            "point_id": self.save_point_id,
                            "session": self._tmp_session_id,
                            "ts": ts,
                        }
                    )

                if diff_bgr is not None:
                    p = str(Path(self._tmp_session_dir) / "final_differ.png")
                    cv2.imwrite(p, diff_bgr)
                    self._tmp_append_meta(
                        {
                            "event": "final_differ_saved",
                            "filename": os.path.basename(p),
                            "point_id": self.save_point_id,
                            "session": self._tmp_session_id,
                            "ts": ts,
                        }
                    )
            except Exception as e:
                self._tmp_dbg(f"save final trio to tmp failed: {e}")

        if self.is_save:
            try:
                try:
                    self._stage_detail = "db_insert"
                except Exception:
                    pass
                if self.logger:
                    self.logger.info("insert database---")
                self.inser_info_database(self.db_dir, self.save_point_id, before_path, after_path)
                if self.logger:
                    self.logger.info("insert database---success")
            except OSError as e:
                if self.logger:
                    self.logger.error(f"path error: {e}, {self.db_dir}, {self.save_point_id}, {before_path}, {after_path}")
            except sqlite3.OperationalError as e:
                if self.logger:
                    self.logger.error(f"db error: {e}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"other error: {e}")

    def get_screen_shot(self):
        img_time = datetime.now()
        if self._capture_roi is None:
            ultra_col_start = 1269
            ultra_col_end = 1920
            # This message is intentionally duplicated for both debug channels:
            # - [offline] is controlled by offline_debug_log
            # - [peakdbg] is controlled by peak_debug_log
            self._dbg(f"screenshot region=({ultra_col_start},0,{ultra_col_end-ultra_col_start},1080) (legacy default)")
            self._pdbg(f"screenshot region=({ultra_col_start},0,{ultra_col_end-ultra_col_start},1080) (legacy default)")
            with SCREENSHOT_LOCK:
                img = pyautogui.screenshot(
                    allScreens=False,
                    region=(ultra_col_start, 0, ultra_col_end - ultra_col_start, 1080),
                )
        else:
            x1, y1, x2, y2 = self._capture_roi
            self._dbg(f"screenshot region=({x1},{y1},{x2-x1},{y2-y1}) (roi1_capture)")
            self._pdbg(f"screenshot region=({x1},{y1},{x2-x1},{y2-y1}) (roi1_capture)")
            with SCREENSHOT_LOCK:
                img = pyautogui.screenshot(allScreens=False, region=(x1, y1, x2 - x1, y2 - y1))
        return img, img_time

    def detect(self, point_id=None, duration=3, is_save=None, stop_event=None):
        if self.logger:
            self.logger.info("detect: " + str(point_id) + str(self.point_id) + " " + str(duration))

        self._dbg(f"detect start: point_id={point_id}, duration={duration}, is_save={is_save}, stop_event_provided={stop_event is not None}")
        self._dbg(f"roi1_capture_effective={self._capture_roi is not None}, roi={self._capture_roi}")
        self._dbg(f"offline_debug_log enabled: every_n_frames={self._debug_every_n_frames}")

        self.point_id = point_id
        self.save_point_id = point_id
        self.is_save = is_save
        # In normal server flow, stop_event is always provided.
        # Keep a safe fallback for direct/manual calls.
        stop_event_provided = stop_event is not None
        self._stop_event = stop_event or threading.Event()
        # Reset lifecycle events per session.
        try:
            self._capture_done_event = threading.Event()
            self._finished_event = threading.Event()
        except Exception:
            pass
        try:
            self._session_start_ts = time.time()
            self._stop_requested_ts = None
            self._stage = "capture_loop"
            self._stage_detail = None
        except Exception:
            pass
        self._pdbg(
            f"offline session start: point_id={point_id}, duration={duration}, is_save={is_save}, stop_event_provided={stop_event_provided}"
        )
        # Hard timeout: even when stop_event is provided (two OFFLINE signals), stop after duration seconds.
        deadline_ts = None
        try:
            deadline_ts = time.time() + float(duration)
        except Exception:
            deadline_ts = None
        timed_out = False

        self.compare_before = None
        self.compare_after = None
        self.before_name = ""
        self.after_name = ""

        # Reset tmp session
        self._tmp_session_dir = None
        self._tmp_session_id = None
        self._tmp_day_dir = None
        self._tmp_meta_path = None
        self._tmp_frame_buffer = []

        # Reset session-scoped SimpleFEM state (but keep module-level last cache for reuse-on-failure)
        self._sf_anchor = None
        self._sf_roi2_rect = None
        self._sf_roi3_rect = None
        self._sf_roi2_before_mean = None
        self._sf_roi2_after_mean = None
        self._sf_roi2_diff = None
        self._sf_color = None
        self._sf_base_color = None
        self._sf_green_case = None

        peak_cfg = (self._raw_setting or {}).get("offline_peak") or {}
        peak_enabled = False
        # threshold is interpreted as an offset added to BEFORE ROI1 mean (dynamic threshold)
        peak_threshold_offset = None
        peak_threshold = None
        after_delay_frames = 2
        end_diff_threshold = 7.0
        try:
            if isinstance(peak_cfg, dict):
                peak_enabled = bool(peak_cfg.get("enabled", False))
                if peak_cfg.get("threshold") is not None:
                    peak_threshold_offset = float(peak_cfg.get("threshold"))
                after_delay_frames = int(peak_cfg.get("after_delay_frames", 2))
                if peak_cfg.get("end_diff_threshold") is not None:
                    end_diff_threshold = float(peak_cfg.get("end_diff_threshold"))
        except Exception:
            if self.logger:
                self.logger.exception("Failed to parse offline_peak config; disabling peak mode")
            peak_enabled = False

        if peak_enabled and peak_threshold_offset is None:
            if self.logger:
                self.logger.warning("offline_peak.enabled is true but threshold is missing; disabling peak mode")
            peak_enabled = False

        self._dbg(
            f"offline_peak parsed: enabled={peak_enabled}, threshold_offset={peak_threshold_offset}, "
            f"after_delay_frames={after_delay_frames}, end_diff_threshold={end_diff_threshold}"
        )
        if peak_enabled and self.logger:
            # Version marker for verifying the new dynamic-threshold behavior is running.
            self.logger.info("offline_peak threshold_mode=before_mean_plus_offset (dynamic) [v2026-01-29]")

        peak_in_high = False
        peak_in_descent = False
        peak_found = False
        after_target_frame = None
        before_gray_mean = None
        prev_is_high = None

        before_default = None
        before_name_default = None
        after_time = None

        frame_counter = 0

        try:
            while not self._stop_event.is_set():
                if deadline_ts is not None and time.time() >= deadline_ts:
                    timed_out = True
                    self._dbg(f"deadline reached (duration={duration}s), stopping loop: frame={frame_counter}")
                    self._pdbg(f"deadline reached: point_id={point_id}, duration={duration}, frame={frame_counter}")
                    try:
                        self._stop_event.set()
                    except Exception:
                        pass
                    break
                if self.compare_after is not None:
                    time.sleep(0.01)
                    continue

                img, img_time = self.get_screen_shot()
                frame_counter += 1

                if frame_counter % 60 == 0 and self.logger:
                    self.logger.info("fps:" + str(frame_counter))

                frame = np.array(img)

                if before_default is None:
                    before_default = frame
                    before_name_default = img_time
                    self._dbg(f"before_default captured at frame=1, ts={self.convert_timestamp2str(img_time)}")

                if frame_counter == 1 and self.compare_before is None:
                    self.compare_before = frame
                    self.before_name = self.convert_timestamp2str(img_time)
                    if self.logger:
                        self.logger.info(self.before_name + ", before img founded (first frame)")
                    self._dbg(f"compare_before captured at frame=1, ts={self.before_name}")

                    # Init tmp session on BEFORE
                    self._tmp_init_session(point_id, self.before_name)

                    # Focus detect + ROI2/ROI3 locate only once on BEFORE.
                    if self._sf_enabled:
                        ok = self._sf_locate_roi2_roi3_on_before(self.compare_before)
                        if ok and self._sf_roi2_rect is not None:
                            self._sf_roi2_before_mean = self._sf_gray_mean(self.compare_before, self._sf_roi2_rect)
                            self._sf_dbg(f"roi2(before) mean={self._sf_roi2_before_mean} rect={self._sf_roi2_rect}")
                        else:
                            self._sf_dbg("peak_detect enabled but focus/ROI2 locate failed on before (may reuse last later)")

                gray = self.compute_grayscale_v2(frame)
                if before_gray_mean is None and frame_counter == 1:
                    before_gray_mean = float(gray)
                    # Dynamic peak threshold: before_gray_mean + offset
                    if peak_enabled and peak_threshold_offset is not None:
                        peak_threshold = float(before_gray_mean) + float(peak_threshold_offset)
                        if self.logger:
                            self.logger.info(
                                f"offline_peak threshold=before_mean({float(before_gray_mean):.3f})+offset({float(peak_threshold_offset):.3f})=>{float(peak_threshold):.3f}"
                            )
                        self._dbg(
                            f"before_gray_mean set: {before_gray_mean:.3f}; "
                            f"peak_threshold(before+offset)={peak_threshold:.3f} (offset={float(peak_threshold_offset):.3f})"
                        )
                    else:
                        self._dbg(f"before_gray_mean set: {before_gray_mean:.3f}")
                    # First OFFLINE signal: record BEFORE frame with ROI1 mean.
                    if self._tmp_frames_enabled and self._tmp_session_dir and self.compare_before is not None:
                        self._tmp_record_frame(self.compare_before, img_time, frame_counter, "before", roi1_gray=gray)

                # Buffer every captured frame while session is active (between before->after capture)
                if self._tmp_frames_enabled and self._tmp_session_dir:
                    self._tmp_record_frame(frame, img_time, frame_counter, "frame", roi1_gray=gray)

                if not peak_enabled:
                    if self._debug_log_enabled and self._debug_every_n_frames > 0 and frame_counter % self._debug_every_n_frames == 0:
                        self._dbg(f"loop: frame={frame_counter}, gray={float(gray):.3f}, peak_enabled=false")
                    continue

                # Peak threshold is dynamic and computed after BEFORE is available
                if peak_threshold is None:
                    # Should not happen (BEFORE is always frame 1), but keep safe fallback
                    if peak_threshold_offset is not None and before_gray_mean is not None:
                        peak_threshold = float(before_gray_mean) + float(peak_threshold_offset)
                    else:
                        peak_threshold = float(peak_threshold_offset or 0.0)

                is_high = gray >= peak_threshold
                if prev_is_high is None:
                    prev_is_high = is_high
                    self._dbg(f"peak init: frame={frame_counter}, gray={float(gray):.3f}, is_high={is_high}")
                    continue

                if self._debug_log_enabled and self._debug_every_n_frames > 0 and frame_counter % self._debug_every_n_frames == 0:
                    self._dbg(
                        f"loop: frame={frame_counter}, gray={float(gray):.3f}, is_high={is_high}, "
                        f"state={{in_high:{peak_in_high}, in_descent:{peak_in_descent}, found:{peak_found}, after_target:{after_target_frame}}}"
                    )

                if (not peak_found) and (not peak_in_high) and (not peak_in_descent) and (not prev_is_high) and is_high:
                    peak_in_high = True
                    if self.logger:
                        self.logger.info(f"peak start: frame={frame_counter}")
                    self._dbg(f"offline_peak took effect: entered high zone at frame={frame_counter}, gray={float(gray):.3f}, threshold={float(peak_threshold):.3f}")

                if (not peak_found) and peak_in_high and prev_is_high and (not is_high):
                    peak_in_high = False
                    peak_in_descent = True
                    if self.logger:
                        self.logger.info(f"peak drop detected (start descent): frame={frame_counter}")
                    self._dbg(f"offline_peak: start descent at frame={frame_counter}, gray={float(gray):.3f}")

                if (not peak_found) and peak_in_descent and before_gray_mean is not None:
                    diff_to_before = abs(float(gray) - float(before_gray_mean))
                    if diff_to_before <= float(end_diff_threshold):
                        peak_in_descent = False
                        peak_found = True
                        after_target_frame = frame_counter + max(0, after_delay_frames)
                        if self.logger:
                            self.logger.info(
                                f"peak end detected: frame={frame_counter}, diff_to_before={diff_to_before:.3f}, "
                                f"end_diff_threshold={float(end_diff_threshold):.3f}, after_target={after_target_frame}"
                            )
                        self._dbg(
                            f"offline_peak: end detected at frame={frame_counter}, diff_to_before={diff_to_before:.3f}, "
                            f"after_target_frame={after_target_frame}"
                        )

                if peak_found and self.compare_after is None and after_target_frame is not None:
                    if frame_counter == after_target_frame:
                        self.compare_after = frame
                        try:
                            self._stage_detail = "after_by_peak"
                        except Exception:
                            pass
                        after_time = img_time
                        self.after_name = self.convert_timestamp2str(after_time)
                        if self.logger:
                            self.logger.info(self.after_name + f", after img founded (peak+{after_delay_frames})")
                        self._dbg(f"compare_after captured via peak: frame={frame_counter}, ts={self.after_name}")
                        self._tmp_record_frame(self.compare_after, after_time, frame_counter, "after_peak", roi1_gray=gray)

                prev_is_high = is_high

        except Exception as e:
            if self.logger:
                self.logger.error(f"in detect_compare_points, some error occurred:\t{e}")
            self.response = {"success": False, "info": "error_in_detect", "detail": str(e)}
            try:
                if self._capture_done_event is not None:
                    self._capture_done_event.set()
                if self._finished_event is not None:
                    self._finished_event.set()
            except Exception:
                pass
            return

        # Screenshot loop ended (by stop_event or timeout). Signal server that capture is done.
        try:
            if self._capture_done_event is not None:
                self._capture_done_event.set()
        except Exception:
            pass
        try:
            self._stage = "post_capture"
            self._stage_detail = None
        except Exception:
            pass

        self._dbg(
            f"loop finished: frames={frame_counter}, peak_enabled={peak_enabled}, peak_found={peak_found}, "
            f"compare_before={'set' if self.compare_before is not None else 'none'}, compare_after={'set' if self.compare_after is not None else 'none'}"
        )

        if self.compare_before is None:
            if self.logger:
                self.logger.warning("can not find before, use default")
            self.compare_before = before_default
            if before_name_default is not None:
                self.before_name = self.convert_timestamp2str(before_name_default)[:-1]
            self._dbg(f"compare_before fallback to default: ts={self.before_name}")

        stop_fallback_failed = False

        # stop fallback: if after not captured, use stop-time screenshot as after
        if self.compare_after is None and self._stop_event is not None and self._stop_event.is_set():
            try:
                stop_img, stop_time = self.get_screen_shot()
                self.compare_after = np.array(stop_img)
                self.after_name = self.convert_timestamp2str(stop_time)
                try:
                    self._stage_detail = "after_by_stop_fallback"
                except Exception:
                    pass
                if self.logger:
                    if timed_out:
                        self.logger.warning(self.after_name + ", after img fallback to timeout screenshot (post-loop)")
                    else:
                        self.logger.info(self.after_name + ", after img fallback to stop-time screenshot (post-loop)")
                self._dbg(f"compare_after fallback to stop-time screenshot: ts={self.after_name}")
                roi1_gray = None
                try:
                    roi1_gray = self.compute_grayscale_v2(self.compare_after)
                except Exception:
                    roi1_gray = None
                self._tmp_record_frame(self.compare_after, stop_time, frame_counter + 1, "after_timeout" if timed_out else "after_stop", roi1_gray=roi1_gray)
            except Exception as e:
                stop_fallback_failed = True
                if self.logger:
                    self.logger.error(f"fallback stop-time screenshot failed (post-loop): {e}")
                self._dbg(f"compare_after stop-time fallback failed: {e}")

        # final fallback:
        # - only when stop fallback screenshot failed, OR stop_event was not provided
        if self.compare_after is None and ((not stop_event_provided) or stop_fallback_failed):
            if self.logger:
                self.logger.warning("can not find after, fallback to a final screenshot")
            after_img, after_t = self.get_screen_shot()
            self.compare_after = np.array(after_img)
            self.after_name = self.convert_timestamp2str(after_t)[:-1]
            try:
                self._stage_detail = "after_by_final_fallback"
            except Exception:
                pass
            self._dbg(f"compare_after final fallback screenshot: ts={self.after_name}")
            roi1_gray = None
            try:
                roi1_gray = self.compute_grayscale_v2(self.compare_after)
            except Exception:
                roi1_gray = None
            self._tmp_record_frame(self.compare_after, after_t, frame_counter + 1, "after_final", roi1_gray=roi1_gray)

        # First-layer判定：ROI2(before) vs ROI2(after) 灰度均值差
        if self._sf_enabled:
            # Ensure we have ROI2 rect; if not, try to locate using BEFORE again (may reuse last cache).
            if self._sf_roi2_rect is None and self.compare_before is not None:
                self._sf_locate_roi2_roi3_on_before(self.compare_before)
                if self._sf_roi2_rect is not None and self._sf_roi2_before_mean is None:
                    self._sf_roi2_before_mean = self._sf_gray_mean(self.compare_before, self._sf_roi2_rect)

            if self._sf_roi2_rect is not None and self.compare_before is not None and self.compare_after is not None:
                if self._sf_roi2_before_mean is None:
                    self._sf_roi2_before_mean = self._sf_gray_mean(self.compare_before, self._sf_roi2_rect)
                self._sf_roi2_after_mean = self._sf_gray_mean(self.compare_after, self._sf_roi2_rect)
                if self._sf_roi2_before_mean is not None and self._sf_roi2_after_mean is not None:
                    self._sf_roi2_diff = float(self._sf_roi2_after_mean) - float(self._sf_roi2_before_mean)
                    self._sf_color = "green" if self._sf_roi2_diff >= float(self._sf_difference_threshold) else "red"
                    self._sf_base_color = self._sf_color
                    if self._sf_color == "green":
                        self._sf_green_case = 1
                    # ROI3 override rules may flip the base ROI2 result from red -> green.
                    if self._sf_color == "red":
                        self._sf_apply_roi3_overrides()
                    self._sf_dbg(
                        f"roi2 diff 판단: before_mean={self._sf_roi2_before_mean:.3f}, after_mean={self._sf_roi2_after_mean:.3f}, "
                        f"diff={self._sf_roi2_diff:.3f}, threshold={float(self._sf_difference_threshold):.3f}, color={self._sf_color}, "
                        f"using_cached_roi2=true"
                    )
                    # Expose in response for caller
                    try:
                        self.response = dict(self.response or {})
                        self.response.update(
                            {
                                "peak_detect_enabled": True,
                                "roi2_rect": self._sf_roi2_rect,
                                "roi3_rect": self._sf_roi3_rect,
                                "roi2_before_mean": self._sf_roi2_before_mean,
                                "roi2_after_mean": self._sf_roi2_after_mean,
                                "roi2_diff": self._sf_roi2_diff,
                                "roi2_color": self._sf_color,
                                "focus_anchor": self._sf_anchor,
                                "roi3_g1": self._sf_roi3_g1,
                                "roi3_g2": self._sf_roi3_g2,
                                "roi3_column_diff": self._sf_roi3_column_diff,
                                "roi3_override_applied": self._sf_roi3_override_applied,
                                "roi3_override_method": self._sf_roi3_override_method,
                                "roi3_override_frame_index": self._sf_roi3_override_frame_index,
                                "roi3_override_tag": self._sf_roi3_override_tag,
                            }
                        )
                    except Exception:
                        pass
                else:
                    self._sf_dbg("roi2 mean compute failed; missing before_mean or after_mean")
            else:
                self._sf_dbg("roi2 diff 판단跳过：roi2_rect/before/after 不齐全")

        # Second OFFLINE signal triggers stop_event; only then do we flush buffered tmp frames to disk.
        if self._tmp_frames_enabled and stop_event_provided and self._stop_event is not None and self._stop_event.is_set():
            self._tmp_flush_on_stop(point_id)

        try:
            try:
                self._stage = "write_img"
                self._stage_detail = "call"
            except Exception:
                pass
            self._dbg(f"write_img start: before_name={self.before_name}, after_name={self.after_name}, is_save={self.is_save}")
            self.write_img()
            self._dbg("write_img done")
        except Exception as e:
            if self.logger:
                self.logger.error(e)
            self.response = {"success": False, "info": "error_in_write img", "detail": str(e)}
            try:
                if self._finished_event is not None:
                    self._finished_event.set()
            except Exception:
                pass
            return

        try:
            stop_reason = "timeout" if timed_out else ("stop_signal" if (self._stop_event is not None and self._stop_event.is_set()) else "unknown")
            after_method = None
            try:
                d = (self._stage_detail or "")
                if "peak" in d:
                    after_method = "peak"
                elif "stop_fallback" in d:
                    after_method = "stop_fallback"
                elif "final_fallback" in d:
                    after_method = "final_fallback"
            except Exception:
                after_method = None
            self._pdbg(
                f"offline session end: point_id={point_id}, stop_reason={stop_reason}, after_method={after_method}, "
                f"before_ts={self.before_name}, after_ts={self.after_name}, peak_enabled={peak_enabled}, peak_found={peak_found}, "
                f"frames={frame_counter}"
            )
        except Exception:
            pass

        try:
            self._stage = "finished"
            self._stage_detail = None
            if self._finished_event is not None:
                self._finished_event.set()
        except Exception:
            pass


if __name__ == "__main__":
    # simple smoke run (manual)
    compare = ComparePoints()
    print("ComparePoints ready:", compare is not None)
