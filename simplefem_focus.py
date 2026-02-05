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


def _mask_green(image_bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary mask (0/255) for green overlay lines.

    Note: thresholds are tuned for the UI green overlay. If the green is too thick
    or anti-aliased, downstream code will prefer center pixels via distanceTransform.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 80, 80], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Clean small speckles while keeping long strokes.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _fit_line_from_points(points_xy: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Fit an infinite 2D line using cv2.fitLine.
    Returns (p0, d) where p0 is a point on the line (x,y) and d is a unit direction (dx,dy).
    """
    if points_xy is None or len(points_xy) < 2:
        return None
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    d = np.array([float(vx), float(vy)], dtype=np.float64)
    n = float(np.hypot(d[0], d[1]))
    if n <= 1e-9:
        return None
    d /= n
    p0 = np.array([float(x0), float(y0)], dtype=np.float64)
    return p0, d


def _point_line_distance(points_xy: np.ndarray, p0: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Perpendicular distance from each point to an infinite line (p0 + t*d).
    points_xy: (N,2)
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    v = pts - p0.reshape(1, 2)
    # distance = |cross(v, d)| since d is unit length
    return np.abs(v[:, 0] * d[1] - v[:, 1] * d[0])


def _ransac_two_lines_intersection(
    points_xy: np.ndarray,
    *,
    dist_thresh: float,
    min_angle_deg: float = 10.0,
    iterations: int = 200,
    min_inliers: int = 200,
) -> Optional[dict]:
    """
    Robustly fit two dominant lines from a point cloud and return their intersection.
    Works well for thick lines because we fit on center pixels (when provided).
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2 * min_inliers:
        return None

    rng = np.random.default_rng(0)

    def fit_one_line_ransac(candidates: np.ndarray, *, require_angle_to: Optional[np.ndarray] = None):
        if len(candidates) < min_inliers:
            return None
        best_inliers = None
        best_p0 = None
        best_d = None

        # Evaluate on a subset for speed when extremely dense.
        if len(candidates) > 120_000:
            idx_eval = rng.choice(len(candidates), size=120_000, replace=False)
            eval_pts = candidates[idx_eval]
        else:
            eval_pts = candidates

        for _ in range(int(iterations)):
            i1, i2 = rng.integers(0, len(candidates), size=2)
            if i1 == i2:
                continue
            p1 = candidates[i1]
            p2 = candidates[i2]
            d = p2 - p1
            n = float(np.hypot(d[0], d[1]))
            if n <= 1e-6:
                continue
            d = d / n
            if require_angle_to is not None:
                dot = float(abs(d[0] * require_angle_to[0] + d[1] * require_angle_to[1]))
                dot = min(1.0, max(-1.0, dot))
                angle = float(np.degrees(np.arccos(dot)))
                if angle < float(min_angle_deg):
                    continue

            dists = _point_line_distance(eval_pts, p1, d)
            inlier_mask = dists <= float(dist_thresh)
            count = int(np.count_nonzero(inlier_mask))
            if best_inliers is None or count > int(np.count_nonzero(best_inliers)):
                best_inliers = inlier_mask
                best_p0 = p1
                best_d = d

        if best_inliers is None or best_p0 is None or best_d is None:
            return None

        # Recompute inliers on all points and refine with fitLine.
        dists_all = _point_line_distance(candidates, best_p0, best_d)
        inliers_idx = np.where(dists_all <= float(dist_thresh))[0]
        if inliers_idx.size < min_inliers:
            return None
        model = _fit_line_from_points(candidates[inliers_idx].astype(np.float32))
        if model is None:
            return None
        p0_ref, d_ref = model
        return {"p0": p0_ref, "d": d_ref, "inliers_idx": inliers_idx}

    first = fit_one_line_ransac(pts)
    if first is None:
        return None

    remaining_mask = np.ones(len(pts), dtype=bool)
    remaining_mask[first["inliers_idx"]] = False
    remaining = pts[remaining_mask]

    second = fit_one_line_ransac(remaining, require_angle_to=first["d"])
    if second is None:
        return None

    p0a, da = first["p0"], first["d"]
    p0b, db = second["p0"], second["d"]

    denom = float(da[0] * db[1] - da[1] * db[0])
    if abs(denom) < 1e-9:
        return None
    dp = p0b - p0a
    t = float((dp[0] * db[1] - dp[1] * db[0]) / denom)
    inter = p0a + t * da
    return {
        "intersection": (float(inter[0]), float(inter[1])),
        "line1": (p0a, da),
        "line2": (p0b, db),
    }


def _refine_intersection_local(
    fg_mask: np.ndarray,
    dist: np.ndarray,
    coarse_xy: Tuple[float, float],
    line1: Tuple[np.ndarray, np.ndarray],
    line2: Tuple[np.ndarray, np.ndarray],
    *,
    window_radius: int = 30,
) -> Tuple[float, float]:
    """
    Refine a coarse intersection using a small local window:
    - candidates: green pixels in the window
    - prefer center pixels (high distanceTransform)
    - constrain to be near both fitted lines
    """
    h, w = fg_mask.shape[:2]
    if h <= 0 or w <= 0:
        return coarse_xy

    cx, cy = int(round(float(coarse_xy[0]))), int(round(float(coarse_xy[1])))
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    r = int(max(5, window_radius))
    x1 = max(0, cx - r)
    x2 = min(w - 1, cx + r)
    y1 = max(0, cy - r)
    y2 = min(h - 1, cy + r)

    win = fg_mask[y1 : y2 + 1, x1 : x2 + 1]
    if win is None or win.size == 0:
        return coarse_xy

    ys, xs = np.where(win > 0)
    if ys.size == 0:
        return coarse_xy

    pts = np.stack([xs + x1, ys + y1], axis=1).astype(np.float64, copy=False)
    p0a, da = line1
    p0b, db = line2
    d1 = _point_line_distance(pts, p0a, da)
    d2 = _point_line_distance(pts, p0b, db)
    dv = dist[pts[:, 1].astype(int), pts[:, 0].astype(int)].astype(np.float64, copy=False)

    # Adaptive gate: allow candidates close to both lines; scale with local thickness.
    local_thick = float(np.percentile(dv, 90)) if dv.size else 0.0
    max_line_dist = max(2.0, local_thick * 0.7)

    gate = (d1 <= max_line_dist) & (d2 <= max_line_dist) & (dv >= 1.0)
    if int(np.count_nonzero(gate)) == 0:
        # Soft score fallback when strict gating yields nothing.
        score = dv - 0.35 * (d1 + d2)
        i = int(np.argmax(score))
        return float(pts[i, 0]), float(pts[i, 1])

    pts_g = pts[gate]
    d1_g = d1[gate]
    d2_g = d2[gate]
    dv_g = dv[gate]

    # Primary: maximize center-ness (dv); Secondary: minimize distance to both lines.
    best_dv = float(np.max(dv_g))
    cand = np.where(dv_g >= (best_dv - 1e-9))[0]
    if cand.size == 1:
        i = int(cand[0])
        return float(pts_g[i, 0]), float(pts_g[i, 1])
    sums = (d1_g + d2_g)[cand]
    j = int(cand[int(np.argmin(sums))])
    return float(pts_g[j, 0]), float(pts_g[j, 1])


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

    # High-accuracy path:
    # - threshold green in HSV
    # - prefer center pixels using distanceTransform (robust to thick lines)
    # - RANSAC-fit two lines and compute their intersection
    try:
        mask = _mask_green(image_bgr)
        fg = (mask > 0).astype(np.uint8) * 255
        pts = None
        dist = None
        if int(np.count_nonzero(fg)) >= 800:
            dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
            vals = dist[fg > 0]
            if vals.size >= 200:
                cut = float(np.percentile(vals, 70))
                center = (dist >= max(1.0, cut)) & (fg > 0)
                if int(np.count_nonzero(center)) >= 200:
                    ys, xs = np.where(center)
                    pts = np.stack([xs, ys], axis=1)
            if pts is None:
                ys, xs = np.where(fg > 0)
                pts = np.stack([xs, ys], axis=1)

        if pts is not None and len(pts) >= 800:
            # Adapt distance threshold to line thickness (center pixels have distance >= ~thickness/2).
            try:
                # If we used center pixels, dist at those points is >= cut; use a small tolerance around it.
                # Otherwise fall back to 3px.
                dist_thresh = 3.0
                if "dist" in locals() and "vals" in locals() and vals.size:
                    dist_thresh = max(2.0, float(np.percentile(vals, 80)) * 0.8)
            except Exception:
                dist_thresh = 3.0

            model = _ransac_two_lines_intersection(
                pts,
                dist_thresh=float(dist_thresh),
                min_angle_deg=10.0,
                iterations=220,
                min_inliers=250,
            )
            if model is not None:
                intersection = model.get("intersection")
                # Local refinement using the fitted lines + distanceTransform.
                try:
                    if intersection is not None and dist is not None:
                        intersection = _refine_intersection_local(
                            fg,
                            dist,
                            intersection,
                            model["line1"],
                            model["line2"],
                            window_radius=34,
                        )
                except Exception:
                    pass
            else:
                intersection = None
        else:
            intersection = None
    except Exception:
        intersection = None

    # Fallback: edge-based Hough on mask (legacy behavior).
    if intersection is None:
        detected = _detect_green_lines(image_bgr)
        if detected is None:
            return None
        line1, line2 = detected
        intersection = _compute_intersection(line1, line2)
        if intersection is None:
            return None
        # Try the same local refinement on the green mask for the fallback path.
        try:
            mask = _mask_green(image_bgr)
            fg = (mask > 0).astype(np.uint8) * 255
            if int(np.count_nonzero(fg)) >= 200:
                dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
                p0a = np.array([float(line1[0]), float(line1[1])], dtype=np.float64)
                da = np.array([float(line1[2] - line1[0]), float(line1[3] - line1[1])], dtype=np.float64)
                na = float(np.hypot(da[0], da[1]))
                if na > 1e-9:
                    da /= na
                p0b = np.array([float(line2[0]), float(line2[1])], dtype=np.float64)
                db = np.array([float(line2[2] - line2[0]), float(line2[3] - line2[1])], dtype=np.float64)
                nb = float(np.hypot(db[0], db[1]))
                if nb > 1e-9:
                    db /= nb
                intersection = _refine_intersection_local(
                    fg, dist, intersection, (p0a, da), (p0b, db), window_radius=34
                )
        except Exception:
            pass

    x, y = intersection
    cx = int(round(x))
    cy = int(round(y))
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    return cx, cy
