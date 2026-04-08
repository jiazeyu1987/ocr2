# -*- coding: utf-8 -*-
import json
import os
import socket
import sqlite3
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox
from typing import Optional

HOST = "127.0.0.1"
PORT = 30415
PASSWORD = "31415"
SOCKET_TIMEOUT_S = 2.0

OFFLINE_TIME_OUT = 100
OFFLINE_IS_SAVE = True
OFFLINE_MAIN_DB_PATH = "D:/software_data/ccwssm"
OFFLINE_BACKUP_DB_PATH = "D:/software_data/zccwssm"


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


def send_offline_once(point_id: int) -> str:
    request_text = build_offline_request_text(point_id)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(SOCKET_TIMEOUT_S)
        sock.connect((HOST, PORT))
        sock.sendall(request_text.encode("utf-8"))
        return recv_one(sock, SOCKET_TIMEOUT_S)


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


class OfflineToggleApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.point_id = 100
        self.in_session = False
        self.sending = False

        self.button = tk.Button(root, text="发射", command=self.on_click)
        self.button.pack(fill=tk.BOTH, expand=True)
        self._refresh_button_appearance()

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
            send_offline_once(point_id)
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
            self._refresh_button_appearance()
            return

        self.in_session = False
        self.point_id += 1
        self._refresh_button_appearance()


def main() -> None:
    root = tk.Tk()
    root.title("offline")
    root.geometry("100x50")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    OfflineToggleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
