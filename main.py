import os
import sys
import ctypes
from ctypes import wintypes

# Work around OpenMP runtime conflicts on Windows (common with MKL + Paddle/OpenCV).
# Must be set before importing libraries that load OpenMP (e.g., paddlepaddle/paddleocr/numpy).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# PyInstaller windowed mode may set stdout/stderr to None; tqdm (used by PaddleOCR downloads) will crash.
try:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
except Exception:
    pass

# Ensure relative paths (e.g. ./whl, ./settings) resolve to the app folder when frozen.
try:
    app_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
except Exception:
    pass


def configure_process_priorities() -> None:
    if os.name != "nt":
        return

    # Windows priority classes.
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
    TH32CS_SNAPPROCESS = 0x00000002
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_SET_INFORMATION = 0x0200
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    def _set_priority(pid: int, priority_class: int) -> bool:
        h_proc = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_SET_INFORMATION, False, pid)
        if not h_proc:
            return False
        try:
            return bool(kernel32.SetPriorityClass(h_proc, priority_class))
        finally:
            kernel32.CloseHandle(h_proc)

    # Lower current backend process priority first.
    try:
        _set_priority(os.getpid(), BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

    # Raise Slicer UI process priority if it exists.
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == INVALID_HANDLE_VALUE:
        return
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            if entry.szExeFile.lower() == "slicerapp-real.exe":
                _set_priority(int(entry.th32ProcessID), ABOVE_NORMAL_PRIORITY_CLASS)
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)


configure_process_priorities()

import server

# server
import socket
import threading
import json
from ocr_detect import OCRDetect
import os
import logging

# ocr
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import pyautogui

import cv2
import time, os


if __name__ == '__main__':
    server.run()
