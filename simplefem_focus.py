# encoding=utf-8
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def compute_roi2_region(
    roi1_size: Tuple[int, int],
    center: Tuple[int, int],
    extension_params: Dict[str, int],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute ROI2/ROI3 region inside ROI1 based on intersection center and extension params.
    Returns (x1,y1,x2,y2) in ROI1-local coordinates, or None.
    """
    roi_width, roi_height = int(roi1_size[0]), int(roi1_size[1])
    cx, cy = int(center[0]), int(center[1])

    if roi_width <= 0 or roi_height <= 0:
        return None

    cx = max(0, min(roi_width - 1, cx))
    cy = max(0, min(roi_height - 1, cy))

    left = int(extension_params.get("left", 0))
    right = int(extension_params.get("right", 0))
    top = int(extension_params.get("top", 0))
    bottom = int(extension_params.get("bottom", 0))

    x1 = max(0, cx - left)
    x2 = min(roi_width, cx + right)
    y1 = max(0, cy - top)
    y2 = min(roi_height, cy + bottom)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _detect_green_lines(image_bgr: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 80, 80], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    edges = cv2.Canny(mask_green, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=80,
        maxLineGap=20,
    )

    if lines is None or len(lines) < 2:
        return None

    def line_length_sq(l):
        x1, y1, x2, y2 = l
        return (x2 - x1) ** 2 + (y2 - y1) ** 2

    lines_list = [tuple(l[0]) for l in lines]
    lines_list.sort(key=line_length_sq, reverse=True)

    def line_angle(l):
        x1, y1, x2, y2 = l
        return float(np.arctan2(y2 - y1, x2 - x1))

    first_line = lines_list[0]
    first_angle = line_angle(first_line)

    chosen_second = None
    for candidate in lines_list[1:]:
        angle = line_angle(candidate)
        diff = abs(angle - first_angle)
        diff = min(diff, float(np.pi) - diff)
        if diff > float(np.deg2rad(10)):
            chosen_second = candidate
            break

    if chosen_second is None:
        return None

    return first_line, chosen_second


def _compute_intersection(
    line1: Tuple[int, int, int, int],
    line2: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float]]:
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    px_num = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    py_num = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)

    return float(px_num / denom), float(py_num / denom)


def detect_green_intersection(image_bgr: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Detect intersection of two green lines (SimpleFEM-style).
    Returns (x,y) in image coordinates, or None.
    """
    if image_bgr is None:
        return None
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None

    detected = _detect_green_lines(image_bgr)
    if detected is None:
        return None
    line1, line2 = detected

    intersection = _compute_intersection(line1, line2)
    if intersection is None:
        return None

    x, y = intersection
    cx = int(round(x))
    cy = int(round(y))
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    return cx, cy

