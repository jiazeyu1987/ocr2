# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple


def build_request_text(req_type: str, password: str, arg_json_text: str) -> str:
    req = req_type.strip().upper()
    if not req:
        raise ValueError("req_type is empty")

    pw = password.strip()
    if not pw:
        raise ValueError("password is empty")

    arg_text = (arg_json_text or "").strip()
    if not arg_text:
        arg_text = "{}"

    try:
        arg_obj = json.loads(arg_text)
    except Exception as e:
        raise ValueError(f"Invalid JSON args: {e}") from e

    # Protocol framing:
    # - Each request is one line, terminated by '\n' (TCP is a byte stream; without framing, messages can coalesce).
    return f"{req};{pw};{json.dumps(arg_obj, ensure_ascii=False)}\n"


def build_request(req_type: str, password: str, arg_json_text: str) -> bytes:
    return build_request_text(req_type, password, arg_json_text).encode("utf-8")


def recv_one(sock: socket.socket, bufsize: int = 65536, timeout_s: float = 2.0) -> str:
    sock.settimeout(timeout_s)
    chunks: list[bytes] = []
    while True:
        try:
            data = sock.recv(bufsize)
        except socket.timeout:
            break
        if not data:
            break
        chunks.append(data)

        # The server sends one JSON object per request, newline-delimited.
        joined = b"".join(chunks)
        if b"\n" in joined:
            joined = joined.split(b"\n", 1)[0]
            return joined.decode("utf-8", errors="replace").strip()

        # Backward-compat fallback: some servers might not newline-delimit.
        joined2 = joined.strip()
        if joined2.startswith(b"{") and joined2.endswith(b"}"):
            break

    return b"".join(chunks).decode("utf-8", errors="replace").strip()


def send_once(
    host: str,
    port: int,
    req_type: str,
    password: str,
    arg_json_text: str,
    timeout_s: float,
) -> Tuple[Optional[dict[str, Any]], str, str]:
    request_text_full = build_request_text(req_type, password, arg_json_text)
    request_text = request_text_full.rstrip("\r\n")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        # Send exactly what build_request_text() produced (includes trailing '\n' framing).
        s.sendall(request_text_full.encode("utf-8"))
        raw = recv_one(s, timeout_s=timeout_s)

    if not raw:
        return None, raw, request_text_full

    try:
        return json.loads(raw), raw, request_text_full
    except Exception:
        return None, raw, request_text_full


def run_cli() -> int:
    p = argparse.ArgumentParser(description="VeinOCRServer TCP test client (CLI/GUI).")
    p.add_argument("--cli", action="store_true", help="Run in CLI mode (default is GUI when no args).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30415)
    p.add_argument("--password", default="31415")
    p.add_argument("--type", default="ONLINE")
    p.add_argument("--arg", default="{}", help='JSON args string, e.g. \'{"point_id":1,"time_out":100,"is_save":true}\'')
    p.add_argument("--timeout", type=float, default=2.0)
    p.add_argument("--watch", action="store_true", help="Poll repeatedly and print results.")
    p.add_argument("--interval", type=float, default=0.5)
    args = p.parse_args()

    if not args.cli:
        # Caller asked for GUI (or ran without --cli); signal to main().
        return 2

    if not args.watch:
        obj, raw, sent = send_once(args.host, args.port, args.type, args.password, args.arg, args.timeout)
        # Show the full, framed request (including trailing '\n') for debugging.
        print(f">> {sent!r}")
        print(json.dumps(obj, ensure_ascii=False, indent=2) if obj is not None else raw)
        return 0

    while True:
        obj, raw, sent = send_once(args.host, args.port, args.type, args.password, args.arg, args.timeout)
        print(f">> {sent!r}")
        print(json.dumps(obj, ensure_ascii=False) if obj is not None else raw)
        time.sleep(args.interval)


@dataclass(frozen=True)
class LogEvent:
    ts: float
    level: str
    message: str


def run_gui() -> int:
    # Tkinter is in the Python stdlib (Windows). No extra dependencies required.
    import tkinter as tk
    from tkinter import messagebox
    from tkinter import ttk
    from tkinter.scrolledtext import ScrolledText

    log_q: "queue.Queue[LogEvent]" = queue.Queue()
    stop_watch = threading.Event()

    def log(level: str, message: str) -> None:
        log_q.put(LogEvent(ts=time.time(), level=level, message=message))

    def fmt_ts(ts: float) -> str:
        return time.strftime("%H:%M:%S", time.localtime(ts))

    root = tk.Tk()
    root.title("VeinOCRServer TCP Tester")
    root.geometry("980x720")

    topmost_var = tk.BooleanVar(master=root, value=False)

    def set_topmost(enabled: bool) -> None:
        topmost_var.set(bool(enabled))
        # On Windows, setting -topmost alone may not immediately raise the window;
        # lift/focus helps make the behavior obvious to the user.
        root.wm_attributes("-topmost", bool(enabled))
        if enabled:
            root.lift()  # 提升窗口
            root.attributes('-topmost', True)  # 再次确保
            root.after(10, root.lift)  # 延迟再次提升
            try:
                root.focus_force()
            except Exception:
                pass
            root.update_idletasks()
        log("INFO", f"Topmost: {'ON' if enabled else 'OFF'} (Ctrl+T)")

    def toggle_topmost(_event: Optional[object] = None) -> None:
        set_topmost(not topmost_var.get())

    root.bind("<Control-t>", toggle_topmost)
    root.bind("<Control-T>", toggle_topmost)

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill=tk.BOTH, expand=True)

    conn = ttk.LabelFrame(frm, text="Connection", padding=10)
    conn.pack(fill=tk.X)

    host_var = tk.StringVar(value="127.0.0.1")
    port_var = tk.StringVar(value="30415")
    password_var = tk.StringVar(value="31415")

    ttk.Label(conn, text="Host").grid(row=0, column=0, sticky="w")
    ttk.Entry(conn, textvariable=host_var, width=22).grid(row=0, column=1, sticky="w", padx=(6, 18))
    ttk.Label(conn, text="Port").grid(row=0, column=2, sticky="w")
    ttk.Entry(conn, textvariable=port_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 18))
    ttk.Label(conn, text="Password").grid(row=0, column=4, sticky="w")
    ttk.Entry(conn, textvariable=password_var, width=12, show="").grid(row=0, column=5, sticky="w", padx=(6, 0))

    req = ttk.LabelFrame(frm, text="Request", padding=10)
    req.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    req_type_var = tk.StringVar(value="ONLINE")
    timeout_var = tk.StringVar(value="2.0")
    interval_var = tk.StringVar(value="0.5")

    ttk.Label(req, text="Type").grid(row=0, column=0, sticky="w")
    type_combo = ttk.Combobox(
        req,
        textvariable=req_type_var,
        width=18,
        values=["ONLINE", "OFFLINE", "CLOSE", "OPENOCR", "CLOSEOCR", "OCR"],
    )
    type_combo.grid(row=0, column=1, sticky="w", padx=(6, 18))
    type_combo["state"] = "normal"  # allow custom typing

    ttk.Label(req, text="Timeout(s)").grid(row=0, column=2, sticky="w")
    ttk.Entry(req, textvariable=timeout_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 18))
    ttk.Label(req, text="Watch interval(s)").grid(row=0, column=4, sticky="w")
    ttk.Entry(req, textvariable=interval_var, width=10).grid(row=0, column=5, sticky="w")

    ttk.Label(req, text="Args (JSON)").grid(row=1, column=0, sticky="nw", pady=(12, 0))
    args_text = ScrolledText(req, height=10, width=80)
    args_text.grid(row=1, column=1, columnspan=5, sticky="nsew", pady=(12, 0))
    args_text.insert(tk.END, "{}")

    req.grid_rowconfigure(1, weight=1)
    req.grid_columnconfigure(5, weight=1)

    btns = ttk.Frame(req)
    btns.grid(row=2, column=0, columnspan=6, sticky="w", pady=(10, 0))

    watch_var = tk.BooleanVar(value=False)

    def parse_conn() -> Tuple[str, int, str, float]:
        host = host_var.get().strip()
        if not host:
            raise ValueError("Host is empty")
        try:
            port = int(port_var.get().strip())
        except Exception as e:
            raise ValueError(f"Invalid port: {e}") from e
        password = password_var.get().strip()
        if not password:
            raise ValueError("Password is empty")
        try:
            timeout_s = float(timeout_var.get().strip())
        except Exception as e:
            raise ValueError(f"Invalid timeout: {e}") from e
        return host, port, password, timeout_s

    def worker_send_once() -> None:
        try:
            host, port, password, timeout_s = parse_conn()
            t = req_type_var.get().strip()
            a = args_text.get("1.0", tk.END)
            obj, raw, sent = send_once(host, port, t, password, a, timeout_s)
            # Show the full, framed request (including trailing '\n') in an unambiguous way.
            log("INFO", f"Send: {host}:{port}  {sent!r}")

            if raw == "":
                # OFFLINE path in server.py intentionally sets response=None, so no response is expected.
                log("WARN", "No response received (timeout/empty). If you sent OFFLINE, server may intentionally not reply.")
            elif obj is not None:
                log("OK", json.dumps(obj, ensure_ascii=False))
            else:
                log("OK", raw)
        except Exception as e:
            log("ERR", str(e))

    def start_watch() -> None:
        if watch_var.get():
            return
        stop_watch.clear()
        watch_var.set(True)
        log("INFO", "Watch started")

        def loop() -> None:
            while not stop_watch.is_set():
                worker_send_once()
                try:
                    interval_s = float(interval_var.get().strip())
                except Exception:
                    interval_s = 0.5
                stop_watch.wait(max(0.05, interval_s))

        threading.Thread(target=loop, daemon=True).start()

    def stop_watch_now() -> None:
        if not watch_var.get():
            return
        watch_var.set(False)
        stop_watch.set()
        log("INFO", "Watch stopped")

    def on_send_click() -> None:
        threading.Thread(target=worker_send_once, daemon=True).start()

    def set_online() -> None:
        req_type_var.set("ONLINE")
        args_text.delete("1.0", tk.END)
        args_text.insert(tk.END, "{}")

    def set_offline() -> None:
        req_type_var.set("OFFLINE")
        args_text.delete("1.0", tk.END)
        args_text.insert(tk.END, json.dumps({"point_id": 123, "time_out": 100, "is_save": True}, ensure_ascii=False, indent=2))

    def set_close() -> None:
        req_type_var.set("CLOSE")
        args_text.delete("1.0", tk.END)
        args_text.insert(tk.END, "{}")

    ttk.Button(btns, text="Send Once", command=on_send_click).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(btns, text="Start Watch", command=start_watch).grid(row=0, column=1, padx=(0, 8))
    ttk.Button(btns, text="Stop Watch", command=stop_watch_now).grid(row=0, column=2, padx=(0, 18))

    ttk.Separator(btns, orient=tk.VERTICAL).grid(row=0, column=3, sticky="ns", padx=(0, 18))
    ttk.Button(btns, text="Preset: ONLINE", command=set_online).grid(row=0, column=4, padx=(0, 8))
    ttk.Button(btns, text="Preset: OFFLINE", command=set_offline).grid(row=0, column=5, padx=(0, 8))
    ttk.Button(btns, text="Preset: CLOSE", command=set_close).grid(row=0, column=6)

    out = ttk.LabelFrame(frm, text="Response / Log", padding=10)
    out.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    log_box = ScrolledText(out, height=18)
    log_box.pack(fill=tk.BOTH, expand=True)
    log_box.configure(state="disabled")

    def flush_logs() -> None:
        try:
            while True:
                ev = log_q.get_nowait()
                line = f"[{fmt_ts(ev.ts)}] {ev.level}: {ev.message}\n"
                log_box.configure(state="normal")
                log_box.insert(tk.END, line)
                log_box.see(tk.END)
                log_box.configure(state="disabled")
        except queue.Empty:
            pass
        root.after(100, flush_logs)

    def on_close() -> None:
        stop_watch.set()
        root.destroy()

    def on_help() -> None:
        messagebox.showinfo(
            "Help",
            "This tool connects to VeinOCRServer (main.py) over TCP.\n\n"
            "Known behaviors:\n"
            "- ONLINE usually replies with JSON (A/B/SkinDepth/etc).\n"
            "- OFFLINE may intentionally not reply (server.py sets response=None).\n"
            "- CLOSE is implemented; OCR/OPENOCR/CLOSEOCR are listed but may not be implemented by the server.\n",
        )

    menubar = tk.Menu(root)
    helpmenu = tk.Menu(menubar, tearoff=0)
    helpmenu.add_command(label="Help", command=on_help)
    menubar.add_cascade(label="Help", menu=helpmenu)
    viewmenu = tk.Menu(menubar, tearoff=0)
    viewmenu.add_checkbutton(label="Topmost (Ctrl+T)", variable=topmost_var, command=toggle_topmost)
    menubar.add_cascade(label="View", menu=viewmenu)
    root.config(menu=menubar)

    root.protocol("WM_DELETE_WINDOW", on_close)
    flush_logs()
    log("INFO", "Ready. Start your server with: python main.py")
    root.mainloop()
    return 0


def main() -> int:
    # Default behavior: GUI when run with no args; CLI when --cli is provided.
    try:
        rc = run_cli()
        if rc != 2:
            return rc
    except SystemExit as e:
        # argparse may call SystemExit; interpret missing args as "start GUI".
        if int(getattr(e, "code", 0) or 0) != 0:
            return int(e.code)

    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
