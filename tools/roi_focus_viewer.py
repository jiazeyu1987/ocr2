import json
import os
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from typing import Optional, List

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover
    Image = None
    ImageTk = None


@dataclass
class OverlayConfig:
    roi2_ext: dict
    roi3_ext: dict


def _load_settings(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_settings_candidates() -> list[Path]:
    candidates = [Path(r"D:\software_data\settings")]
    try:
        here = Path(__file__).resolve().parents[1]
        candidates.append(here / "settings")
    except Exception:
        pass
    return candidates


def _find_settings_file() -> Optional[Path]:
    for p in _default_settings_candidates():
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def _extract_overlay_cfg(settings: dict) -> OverlayConfig:
    peak = (settings or {}).get("peak_detect") or {}
    roi2 = peak.get("roi2_extension_params") or {"left": 40, "right": 40, "top": 50, "bottom": 30}
    roi3 = peak.get("roi3_extension_params") or {"left": 30, "right": 30, "top": 50, "bottom": 100}
    return OverlayConfig(roi2_ext=dict(roi2), roi3_ext=dict(roi3))


def _imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def _draw_overlays(bgr: np.ndarray, cfg: OverlayConfig):
    # Ensure repo root is importable (when running as python tools/roi_focus_viewer.py)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        import simplefem_focus  # type: ignore
    except Exception as e:
        raise RuntimeError(f"导入 simplefem_focus 失败: {e}")

    h, w = bgr.shape[:2]
    anchor = simplefem_focus.detect_green_intersection(bgr)

    out = bgr.copy()

    if anchor is not None:
        ax, ay = int(anchor[0]), int(anchor[1])
        # Focus: red circle (radius=2, thickness=1)
        cv2.circle(out, (ax, ay), 2, (0, 0, 255), 1)
        cv2.putText(out, f"anchor=({ax},{ay})", (ax + 8, max(18, ay - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        roi2 = simplefem_focus.compute_roi2_region((w, h), (ax, ay), cfg.roi2_ext)
        roi3 = simplefem_focus.compute_roi2_region((w, h), (ax, ay), cfg.roi3_ext)

        if roi2 is not None:
            x1, y1, x2, y2 = map(int, roi2)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue
            cv2.putText(out, "ROI2", (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        if roi3 is not None:
            x1, y1, x2, y2 = map(int, roi3)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 215, 255), 2)  # yellow-ish
            cv2.putText(out, "ROI3", (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
    else:
        cv2.putText(out, "anchor: NOT FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return out, anchor


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ROI2/ROI3 & Focus Viewer")
        self.geometry("1100x800")

        if Image is None or ImageTk is None:
            messagebox.showerror("Missing dependency", "需要安装 Pillow：pip install pillow")
            raise SystemExit(2)

        self._settings_path: Optional[Path] = _find_settings_file()
        self._settings: Optional[dict] = None
        self._cfg: Optional[OverlayConfig] = None
        self._image_path: Optional[Path] = None
        self._bgr: Optional[np.ndarray] = None
        self._photo = None
        self._scale = 1.0

        self._build_ui()
        self._load_settings_initial()

    def _build_ui(self):
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Button(top, text="打开图片", command=self.on_open_image).pack(side=tk.LEFT)
        tk.Button(top, text="选择settings", command=self.on_choose_settings).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(top, text="重载/刷新", command=self.on_refresh).pack(side=tk.LEFT, padx=(8, 0))

        self.lbl_settings = tk.Label(top, text="settings: (未加载)")
        self.lbl_settings.pack(side=tk.LEFT, padx=(12, 0))

        self.lbl_info = tk.Label(top, text="")
        self.lbl_info.pack(side=tk.RIGHT)

        body = tk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg="#1e1e1e")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.vbar = tk.Scrollbar(body, orient=tk.VERTICAL, command=self.canvas.yview)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.hbar = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)

        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Control-plus>", lambda _e: self._zoom(1.15))
        self.bind("<Control-minus>", lambda _e: self._zoom(1 / 1.15))
        self.bind("<Control-0>", lambda _e: self._set_zoom(1.0))

    def _load_settings_initial(self):
        if self._settings_path is None:
            self.lbl_settings.configure(text="settings: (未找到，点“选择settings”)")
            return
        try:
            self._settings = _load_settings(self._settings_path)
            self._cfg = _extract_overlay_cfg(self._settings)
            self.lbl_settings.configure(text=f"settings: {self._settings_path}")
        except Exception as e:
            self._settings = None
            self._cfg = None
            self.lbl_settings.configure(text=f"settings: 加载失败: {e}")

    def on_choose_settings(self):
        p = filedialog.askopenfilename(
            title="选择 settings(JSON)",
            initialdir=str(Path(r"D:\software_data")) if Path(r"D:\software_data").exists() else str(Path.cwd()),
            filetypes=[("settings", "settings"), ("json", "*.json"), ("all", "*.*")],
        )
        if not p:
            return
        self._settings_path = Path(p)
        self._load_settings_initial()
        self._redraw()

    def on_open_image(self):
        p = filedialog.askopenfilename(
            title="打开图片",
            initialdir=str(Path(r"D:\software_data\imgs")) if Path(r"D:\software_data\imgs").exists() else str(Path.cwd()),
            filetypes=[("images", "*.png;*.jpg;*.jpeg;*.bmp"), ("all", "*.*")],
        )
        if not p:
            return
        self._image_path = Path(p)
        self._bgr = _imread(self._image_path)
        if self._bgr is None:
            messagebox.showerror("读取失败", f"无法读取图片: {self._image_path}")
            return
        self._scale = 1.0
        self._redraw()

    def on_refresh(self):
        self._load_settings_initial()
        self._redraw()

    def _zoom(self, factor: float):
        self._set_zoom(self._scale * factor)

    def _set_zoom(self, scale: float):
        self._scale = float(max(0.1, min(6.0, scale)))
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        if self._bgr is None:
            return
        if self._cfg is None:
            self.lbl_info.configure(text="settings未加载")
            img = self._bgr
        else:
            try:
                img, anchor = _draw_overlays(self._bgr, self._cfg)
                if anchor is None:
                    self.lbl_info.configure(text="anchor: NOT FOUND")
                else:
                    self.lbl_info.configure(text=f"anchor: {anchor}")
            except Exception as e:
                self.lbl_info.configure(text=f"绘制失败: {e}")
                img = self._bgr

        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        if abs(self._scale - 1.0) > 1e-6:
            w, h = pil.size
            pil = pil.resize((int(w * self._scale), int(h * self._scale)), Image.Resampling.LANCZOS)

        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, image=self._photo, anchor=tk.NW)
        self.canvas.configure(scrollregion=(0, 0, pil.size[0], pil.size[1]))


def main():
    # Make sure tkinter doesn't pop up behind other windows
    os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")
    app = App()
    app.mainloop()


if __name__ == "__main__":
    raise SystemExit(main())
