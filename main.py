import os
import sys

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
