# encoding=utf-8
import os
import sqlite3
import time
from datetime import datetime

import cv2
import image_difference
import numpy as np
import pyautogui
import threading


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

        self.compare_before = None
        self.compare_after = None
        self.before_name = ""
        self.after_name = ""

    def _dbg(self, msg):
        if self._debug_log_enabled and self.logger:
            self.logger.info(f"[offline] {msg}")

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

    def write_img(self):
        if self.logger:
            self.logger.info("write_img...")

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
            cv2.imwrite(before_path, cv2.cvtColor(self.compare_before, cv2.COLOR_RGB2BGR))
        else:
            if self.logger:
                self.logger.info("no compare before")

        if self.compare_after is not None:
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

        if self.compare_before is not None and self.compare_after is not None:
            direct_diff = np.array(self.compare_after).astype(np.float32) - np.array(self.compare_before).astype(np.float32)
            direct_diff[np.where(direct_diff < 0)] = 0
            direct_diff = direct_diff.astype(np.uint8)
            cv2.imwrite(after_path.replace("_after", "_diff"), direct_diff)
        else:
            if self.logger:
                self.logger.info("no compare diff")

        if self.is_save:
            try:
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
            self._dbg(f"screenshot region=({ultra_col_start},0,{ultra_col_end-ultra_col_start},1080) (legacy default)")
            img = pyautogui.screenshot(
                allScreens=False,
                region=(ultra_col_start, 0, ultra_col_end - ultra_col_start, 1080),
            )
        else:
            x1, y1, x2, y2 = self._capture_roi
            self._dbg(f"screenshot region=({x1},{y1},{x2-x1},{y2-y1}) (roi1_capture)")
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
        deadline_ts = None
        if stop_event is None:
            try:
                deadline_ts = time.time() + float(duration)
            except Exception:
                deadline_ts = None

        self.compare_before = None
        self.compare_after = None
        self.before_name = ""
        self.after_name = ""

        peak_cfg = (self._raw_setting or {}).get("offline_peak") or {}
        peak_enabled = False
        peak_threshold = None
        after_delay_frames = 2
        end_diff_threshold = 7.0
        try:
            if isinstance(peak_cfg, dict):
                peak_enabled = bool(peak_cfg.get("enabled", False))
                if peak_cfg.get("threshold") is not None:
                    peak_threshold = float(peak_cfg.get("threshold"))
                after_delay_frames = int(peak_cfg.get("after_delay_frames", 2))
                if peak_cfg.get("end_diff_threshold") is not None:
                    end_diff_threshold = float(peak_cfg.get("end_diff_threshold"))
        except Exception:
            if self.logger:
                self.logger.exception("Failed to parse offline_peak config; disabling peak mode")
            peak_enabled = False

        if peak_enabled and peak_threshold is None:
            if self.logger:
                self.logger.warning("offline_peak.enabled is true but threshold is missing; disabling peak mode")
            peak_enabled = False

        self._dbg(
            f"offline_peak parsed: enabled={peak_enabled}, threshold={peak_threshold}, "
            f"after_delay_frames={after_delay_frames}, end_diff_threshold={end_diff_threshold}"
        )

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
                    self._dbg(f"deadline reached, stopping loop: frame={frame_counter}")
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

                gray = self.compute_grayscale_v2(frame)
                if before_gray_mean is None and frame_counter == 1:
                    before_gray_mean = float(gray)
                    self._dbg(f"before_gray_mean set: {before_gray_mean:.3f}")

                if not peak_enabled:
                    if self._debug_log_enabled and self._debug_every_n_frames > 0 and frame_counter % self._debug_every_n_frames == 0:
                        self._dbg(f"loop: frame={frame_counter}, gray={float(gray):.3f}, peak_enabled=false")
                    continue

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
                        after_time = img_time
                        self.after_name = self.convert_timestamp2str(after_time)
                        if self.logger:
                            self.logger.info(self.after_name + f", after img founded (peak+{after_delay_frames})")
                        self._dbg(f"compare_after captured via peak: frame={frame_counter}, ts={self.after_name}")

                prev_is_high = is_high

        except Exception as e:
            if self.logger:
                self.logger.error(f"in detect_compare_points, some error occurred:\t{e}")
            self.response = {"success": False, "info": "error_in_detect", "detail": str(e)}
            return

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
                if self.logger:
                    self.logger.info(self.after_name + ", after img fallback to stop-time screenshot (post-loop)")
                self._dbg(f"compare_after fallback to stop-time screenshot: ts={self.after_name}")
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
            self._dbg(f"compare_after final fallback screenshot: ts={self.after_name}")

        try:
            self._dbg(f"write_img start: before_name={self.before_name}, after_name={self.after_name}, is_save={self.is_save}")
            self.write_img()
            self._dbg("write_img done")
        except Exception as e:
            if self.logger:
                self.logger.error(e)
            self.response = {"success": False, "info": "error_in_write img", "detail": str(e)}
            return


if __name__ == "__main__":
    # simple smoke run (manual)
    compare = ComparePoints()
    print("ComparePoints ready:", compare is not None)
