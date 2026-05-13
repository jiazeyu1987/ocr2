# -*- coding: utf-8 -*-
import colorsys
import json
import os
import socket
import sqlite3
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox
from typing import Optional

from PIL import ImageGrab

HOST = "127.0.0.1"
PORT = 30415
PASSWORD = "31415"
CONNECT_TIMEOUT_S = 2.0
RECV_TIMEOUT_START_S = 2.0
RECV_TIMEOUT_STOP_S = 25.0

OFFLINE_TIME_OUT = 100
OFFLINE_IS_SAVE = True
OFFLINE_MAIN_DB_PATH = "D:/software_data/ccwssm"
OFFLINE_BACKUP_DB_PATH = "D:/software_data/zccwssm"

# Auto-monitor config (screen absolute coordinates)
# Red-box center: when this point turns yellow -> trigger START(发射)
# Yellow-box center: when this point changes from green to non-green -> trigger STOP(结束)
MONITOR_ENABLED = True
MONITOR_INTERVAL_MS = 120
START_TRIGGER_POINTS = [
    (1146, 791),
    (735, 961),
    (792, 830),
]
START_REGION_RADIUS = 3
START_ACTIVE_RATIO = 0.25
END_TRIGGER_POINTS = [
    (1124, 829),
    (1237, 788),
    (926, 965),
]

# HSV ranges with宽松阈值，适配颜色抖动
YELLOW_H_MIN = 0.10
YELLOW_H_MAX = 0.20
YELLOW_S_MIN = 0.20
YELLOW_V_MIN = 0.15

ORANGE_H_MIN = 0.03
ORANGE_H_MAX = 0.12
ORANGE_S_MIN = 0.20
ORANGE_V_MIN = 0.15

GREEN_H_MIN = 0.25
GREEN_H_MAX = 0.45
GREEN_S_MIN = 0.20
GREEN_V_MIN = 0.15

# Stop trigger stability: per-point green edge detection (green -> non-green)
END_REGION_RADIUS = 3  # 7x7 area around each point in END_TRIGGER_POINTS
END_GREEN_ACTIVE_RATIO = 0.22

# Optional debug logs for color monitor tuning
MONITOR_DEBUG_LOG = False

# Fullscreen overlay visualization
OVERLAY_TRANSPARENT_COLOR = "#010203"
START_BOX_COLOR = "red"
END_BOX_COLOR = "yellow"
BOX_BORDER_WIDTH = 2
BOX_HALF_SIZE = 10  # visualization size; independent from detection radius
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 50
BUTTON_X = 20
BUTTON_Y = 20


def build_offline_request_text(point_id: int) -> str:
    payload = {
        "point_id": point_id,
        "time_out": OFFLINE_TIME_OUT,
        "is_save": OFFLINE_IS_SAVE,
    }
    return f"OFFLINE;{PASSWORD};{json.dumps(payload, ensure_ascii=False)}\n"


def recv_one(sock: socket.socket, timeout_s: float) -> str:
    sock.settimeout(timeout_s)
    chunks: list[bytes] = []
    while True:
        try:
            data = sock.recv(65536)
        except socket.timeout:
            break

        if not data:
            break

        chunks.append(data)
        joined = b"".join(chunks)

        if b"\n" in joined:
            joined = joined.split(b"\n", 1)[0]
            return joined.decode("utf-8", errors="replace").strip()

        joined2 = joined.strip()
        if joined2.startswith(b"{") and joined2.endswith(b"}"):
            break

    return b"".join(chunks).decode("utf-8", errors="replace").strip()


def send_offline_once(point_id: int, action_is_start: bool = True) -> str:
    request_text = build_offline_request_text(point_id)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(CONNECT_TIMEOUT_S)
        try:
            sock.connect((HOST, PORT))
        except TimeoutError as exc:
            raise TimeoutError(
                f"连接 {HOST}:{PORT} 超时，请确认 OCR 服务(main.py)已启动并在该端口监听"
            ) from exc
        except OSError as exc:
            raise ConnectionError(
                f"无法连接到 {HOST}:{PORT}，请确认 OCR 服务(main.py)已启动。原始错误: {exc}"
            ) from exc
        sock.sendall(request_text.encode("utf-8"))
        recv_timeout = RECV_TIMEOUT_START_S if action_is_start else RECV_TIMEOUT_STOP_S
        return recv_one(sock, recv_timeout)


def _table_exists(db: sqlite3.Connection, table_name: str) -> bool:
    row = db.cursor().execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _get_segment_table_create_sql(main_db_path: str) -> str:
    with sqlite3.connect(main_db_path, timeout=30) as db:
        row = db.cursor().execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='SegmentImagesInfo' LIMIT 1"
        ).fetchone()
    if row is None or not row[0]:
        raise RuntimeError("主库缺少 SegmentImagesInfo 表定义")
    return str(row[0])


def _ensure_segment_table(db_path: str, create_sql: str) -> None:
    with sqlite3.connect(db_path, timeout=30) as db:
        if _table_exists(db, "SegmentImagesInfo"):
            return
        db.cursor().execute(create_sql)
        db.commit()


def _ensure_segment_row_in_db(db_path: str, point_id: int, now_text: str) -> None:
    with sqlite3.connect(db_path, timeout=30) as db:
        cur = db.cursor()
        row = cur.execute(
            "SELECT 1 FROM SegmentImagesInfo WHERE ID = ? LIMIT 1",
            (point_id,),
        ).fetchone()
        if row is not None:
            return

        cur.execute(
            "INSERT INTO SegmentImagesInfo (ID, PointID, CreateTime, ModifyTime) VALUES (?, ?, ?, ?)",
            (point_id, point_id, now_text, now_text),
        )
        db.commit()


def ensure_segment_images_row(point_id: int) -> None:
    main_db_path = OFFLINE_MAIN_DB_PATH
    backup_db_path = OFFLINE_BACKUP_DB_PATH

    if not os.path.exists(main_db_path):
        raise FileNotFoundError(f"主数据库不存在: {main_db_path}")

    backup_dir = os.path.dirname(backup_db_path)
    if backup_dir and (not os.path.isdir(backup_dir)):
        raise FileNotFoundError(f"备份数据库目录不存在: {backup_dir}")

    create_sql = _get_segment_table_create_sql(main_db_path)
    _ensure_segment_table(main_db_path, create_sql)
    _ensure_segment_table(backup_db_path, create_sql)

    now_text = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")[:-3]
    _ensure_segment_row_in_db(main_db_path, point_id, now_text)
    _ensure_segment_row_in_db(backup_db_path, point_id, now_text)


def _query_max_segment_id(db_path: str) -> int:
    if not os.path.exists(db_path):
        return -1

    with sqlite3.connect(db_path, timeout=30) as db:
        cur = db.cursor()
        row = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='SegmentImagesInfo' LIMIT 1"
        ).fetchone()
        if row is None:
            return -1
        row = cur.execute("SELECT MAX(ID) FROM SegmentImagesInfo").fetchone()
        if row is None or row[0] is None:
            return -1
        return int(row[0])


def resolve_initial_point_id(default_start: int = 100) -> int:
    main_db_path = OFFLINE_MAIN_DB_PATH
    backup_db_path = OFFLINE_BACKUP_DB_PATH
    if not os.path.exists(main_db_path):
        raise FileNotFoundError(f"main database not found: {main_db_path}")

    max_ids = [
        _query_max_segment_id(main_db_path),
        _query_max_segment_id(backup_db_path),
    ]
    max_used = max(max_ids)
    return max(int(default_start), int(max_used) + 1)


class OfflineToggleApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.point_id = resolve_initial_point_id(default_start=100)
        self.in_session = False
        self.sending = False
        self.monitor_enabled = MONITOR_ENABLED
        self.monitor_job = None
        self.prev_start_active_list: list[bool] = []
        self.prev_end_green_list: list[bool] = []

        self.button = tk.Button(root, text="发射", command=self.on_click)
        self.button.pack(fill=tk.BOTH, expand=True)
        self._refresh_button_appearance()
        self._start_monitor_loop()

    def _rgb_from_screen_point(self, point: tuple[int, int]) -> tuple[int, int, int]:
        x, y = point
        shot = ImageGrab.grab(bbox=(x, y, x + 1, y + 1))
        pixel = shot.getpixel((0, 0))
        if isinstance(pixel, int):
            return pixel, pixel, pixel
        if isinstance(pixel, tuple) and len(pixel) >= 3:
            return int(pixel[0]), int(pixel[1]), int(pixel[2])
        raise ValueError(f"无法读取像素颜色: point={point}, pixel={pixel!r}")

    def _rgb_region_from_screen_point(self, point: tuple[int, int], radius: int) -> list[tuple[int, int, int]]:
        x, y = point
        r = max(0, int(radius))
        left = max(0, x - r)
        top = max(0, y - r)
        right = x + r + 1
        bottom = y + r + 1
        shot = ImageGrab.grab(bbox=(left, top, right, bottom))

        pixels: list[tuple[int, int, int]] = []
        for pixel in shot.getdata():
            if isinstance(pixel, int):
                pixels.append((pixel, pixel, pixel))
                continue
            if isinstance(pixel, tuple) and len(pixel) >= 3:
                pixels.append((int(pixel[0]), int(pixel[1]), int(pixel[2])))

        if not pixels:
            raise ValueError(f"empty sampled region: point={point}, radius={radius}")
        return pixels

    @staticmethod
    def _color_ratio(pixels: list[tuple[int, int, int]], predicate) -> float:
        if not pixels:
            return 0.0
        matched = 0
        for rgb in pixels:
            if predicate(rgb):
                matched += 1
        return float(matched) / float(len(pixels))

    @staticmethod
    def _is_hsv_in_range(
        rgb: tuple[int, int, int],
        h_min: float,
        h_max: float,
        s_min: float,
        v_min: float,
    ) -> bool:
        r, g, b = rgb
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        return h_min <= h <= h_max and s >= s_min and v >= v_min

    def _is_start_color_yellow(self, rgb: tuple[int, int, int]) -> bool:
        return self._is_hsv_in_range(
            rgb,
            YELLOW_H_MIN,
            YELLOW_H_MAX,
            YELLOW_S_MIN,
            YELLOW_V_MIN,
        )

    def _is_start_color_orange(self, rgb: tuple[int, int, int]) -> bool:
        return self._is_hsv_in_range(
            rgb,
            ORANGE_H_MIN,
            ORANGE_H_MAX,
            ORANGE_S_MIN,
            ORANGE_V_MIN,
        )

    def _is_start_signal_color(self, rgb: tuple[int, int, int]) -> bool:
        # Start signal may appear as yellow/orange indicator or green treatment bar.
        return (
            self._is_start_color_yellow(rgb)
            or self._is_start_color_orange(rgb)
            or self._is_end_color_green(rgb)
        )

    def _sample_start_point_active(self, point: tuple[int, int]) -> tuple[float, bool]:
        pixels = self._rgb_region_from_screen_point(point, START_REGION_RADIUS)
        active_ratio = self._color_ratio(pixels, self._is_start_signal_color)
        return active_ratio, active_ratio >= START_ACTIVE_RATIO

    def _is_end_color_green(self, rgb: tuple[int, int, int]) -> bool:
        return self._is_hsv_in_range(
            rgb,
            GREEN_H_MIN,
            GREEN_H_MAX,
            GREEN_S_MIN,
            GREEN_V_MIN,
        )

    def _debug_monitor(self, message: str) -> None:
        if MONITOR_DEBUG_LOG:
            print(f"[offline_monitor] {message}")

    def _on_root_configure(self, _event=None) -> None:
        self._draw_detection_boxes()

    def _draw_detection_boxes(self) -> None:
        if not hasattr(self, "overlay"):
            return

        self.overlay.delete("detect_boxes")

        for x, y in START_TRIGGER_POINTS:
            self.overlay.create_rectangle(
                x - BOX_HALF_SIZE,
                y - BOX_HALF_SIZE,
                x + BOX_HALF_SIZE,
                y + BOX_HALF_SIZE,
                outline=START_BOX_COLOR,
                width=BOX_BORDER_WIDTH,
                tags="detect_boxes",
            )

        for ex, ey in END_TRIGGER_POINTS:
            self.overlay.create_rectangle(
                ex - BOX_HALF_SIZE,
                ey - BOX_HALF_SIZE,
                ex + BOX_HALF_SIZE,
                ey + BOX_HALF_SIZE,
                outline=END_BOX_COLOR,
                width=BOX_BORDER_WIDTH,
                tags="detect_boxes",
            )

        self.button.lift()

    def _reset_stop_state(self) -> None:
        self.prev_end_green_list = [False] * len(END_TRIGGER_POINTS)

    def _start_monitor_loop(self) -> None:
        if not self.monitor_enabled:
            return
        try:
            self.prev_start_active_list = []
            for pt in START_TRIGGER_POINTS:
                ratio, active = self._sample_start_point_active(pt)
                self.prev_start_active_list.append(active)
                self._debug_monitor(f"init-start: point={pt}, active_ratio={ratio:.3f}, active={active}")
            end_green_ratios: list[float] = []
            for pt in END_TRIGGER_POINTS:
                end_pixels = self._rgb_region_from_screen_point(pt, END_REGION_RADIUS)
                end_green_ratios.append(self._color_ratio(end_pixels, self._is_end_color_green))
            self.prev_end_green_list = [
                ratio >= END_GREEN_ACTIVE_RATIO for ratio in end_green_ratios
            ]
            self._debug_monitor(
                f"init: end_points={END_TRIGGER_POINTS}, "
                f"end_green_ratios={[round(v, 3) for v in end_green_ratios]}, "
                f"end_prev_green={self.prev_end_green_list}, "
                f"end_green_active_ratio={END_GREEN_ACTIVE_RATIO:.3f}"
            )
        except Exception as e:
            messagebox.showerror("监测失败", f"无法读取监测像素，请检查坐标与权限: {e}")
            self.root.destroy()
            return
        self.monitor_job = self.root.after(MONITOR_INTERVAL_MS, self._poll_monitor)

    def _poll_monitor(self) -> None:
        self.monitor_job = None
        if not self.monitor_enabled:
            return

        try:
            start_active_list: list[bool] = []
            start_ratio_list: list[float] = []
            for pt in START_TRIGGER_POINTS:
                ratio, active = self._sample_start_point_active(pt)
                start_ratio_list.append(ratio)
                start_active_list.append(active)
            end_green_ratios: list[float] = []
            for pt in END_TRIGGER_POINTS:
                end_pixels = self._rgb_region_from_screen_point(pt, END_REGION_RADIUS)
                end_green_ratios.append(self._color_ratio(end_pixels, self._is_end_color_green))
            curr_end_green_list = [ratio >= END_GREEN_ACTIVE_RATIO for ratio in end_green_ratios]
        except Exception as e:
            messagebox.showerror("监测失败", f"读取像素失败: {e}")
            self.root.destroy()
            return

        # 红框点: 非黄 -> 黄 => 发射
        should_start = False
        if len(self.prev_start_active_list) != len(start_active_list):
            self.prev_start_active_list = [False] * len(start_active_list)
        for idx, active in enumerate(start_active_list):
            prev = self.prev_start_active_list[idx]
            if (not prev) and active:
                should_start = True
                break
        should_start = (not self.in_session) and (not self.sending) and should_start
        # 黄框点: 绿 -> 非绿 => 结束
        should_stop = False
        stop_trigger_indexes: list[int] = []
        if self.in_session and (not self.sending):
            if len(self.prev_end_green_list) != len(curr_end_green_list):
                self.prev_end_green_list = [False] * len(curr_end_green_list)
            for idx, curr_green in enumerate(curr_end_green_list):
                prev_green = self.prev_end_green_list[idx]
                if prev_green and (not curr_green):
                    stop_trigger_indexes.append(idx)
            should_stop = len(stop_trigger_indexes) > 0
        else:
            self._reset_stop_state()

        self._debug_monitor(
            f"poll: in_session={self.in_session}, sending={self.sending}, "
            f"start_points={START_TRIGGER_POINTS}, start_ratios={[round(v, 3) for v in start_ratio_list]}, "
            f"start_prev={self.prev_start_active_list}, start_curr={start_active_list}, "
            f"end_points={END_TRIGGER_POINTS}, end_green_ratios={[round(v, 3) for v in end_green_ratios]}, "
            f"end_prev={self.prev_end_green_list}, end_curr={curr_end_green_list}, "
            f"stop_triggers={stop_trigger_indexes}, "
            f"should_start={should_start}, should_stop={should_stop}"
        )
        self.prev_start_active_list = start_active_list
        if self.in_session and (not self.sending):
            self.prev_end_green_list = curr_end_green_list

        if should_start:
            self.on_click()
        elif should_stop:
            self.on_click()

        self.monitor_job = self.root.after(MONITOR_INTERVAL_MS, self._poll_monitor)

    def _refresh_button_appearance(self) -> None:
        if not self.in_session:
            # 发射状态：按钮绿色
            self.button.configure(
                text="发射",
                bg="#2ecc71",
                activebackground="#27ae60",
                fg="white",
                activeforeground="white",
            )
            return

        # 结束状态：文字红色
        self.button.configure(
            text="结束",
            bg="SystemButtonFace",
            activebackground="SystemButtonFace",
            fg="red",
            activeforeground="red",
        )

    def on_click(self) -> None:
        if self.sending:
            return

        action_is_start = not self.in_session
        target_point_id = self.point_id

        self.sending = True
        self.button.configure(state=tk.DISABLED)

        worker = threading.Thread(
            target=self._send_worker,
            args=(target_point_id, action_is_start),
            daemon=True,
        )
        worker.start()

    def _send_worker(self, point_id: int, action_is_start: bool) -> None:
        err: Optional[Exception] = None
        try:
            if action_is_start:
                ensure_segment_images_row(point_id)
            raw = send_offline_once(point_id, action_is_start)
            if not raw:
                raise TimeoutError(
                    f"服务端未在超时时间内返回响应(start={RECV_TIMEOUT_START_S}s, stop={RECV_TIMEOUT_STOP_S}s)"
                )

            try:
                obj = json.loads(raw)
            except Exception as exc:
                raise RuntimeError(f"服务端返回非 JSON 响应: {raw!r}") from exc

            if not isinstance(obj, dict):
                raise RuntimeError(f"服务端返回格式错误: {obj!r}")
            if not bool(obj.get("success", False)):
                info = str(obj.get("info", "offline request failed"))
                if info == "offline_ignored_extra_request":
                    raise RuntimeError(
                        f"point_id={point_id} already reached server request limit (2). "
                        "Restart GUI to refresh point_id, or use a larger new point_id."
                    )
                raise RuntimeError(info)
        except Exception as exc:
            err = exc
        self.root.after(0, lambda: self._after_send(action_is_start, err))

    def _after_send(self, action_is_start: bool, err: Optional[Exception]) -> None:
        self.sending = False
        self.button.configure(state=tk.NORMAL)

        if err is not None:
            messagebox.showerror("发送失败", str(err))
            return

        if action_is_start:
            self.in_session = True
            self._reset_stop_state()
            self._refresh_button_appearance()
            return

        self.in_session = False
        self._reset_stop_state()
        self.point_id += 1
        self._refresh_button_appearance()


def main() -> None:
    root = tk.Tk()
    root.title("offline")
    root.geometry("100x50")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    try:
        OfflineToggleApp(root)
    except Exception as e:
        messagebox.showerror("启动失败", str(e))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
