import argparse
from pathlib import Path

import cv2
import numpy as np
import sys
import importlib.util


def _imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite(path: Path, img) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Mark SimpleFEM focus(anchor) on an ultrasound screenshot.")
    p.add_argument("--in", dest="in_path", required=True, help="Input image path (BEFORE screenshot).")
    p.add_argument("--out", dest="out_path", default=r"D:\software_data\imgs\tmp.png", help="Output image path.")
    p.add_argument("--radius", type=int, default=18, help="Circle radius (pixels).")
    p.add_argument("--thickness", type=int, default=3, help="Circle thickness (pixels).")
    p.add_argument("--force-py", action="store_true", help="Force-load simplefem_focus.py (ignore any .pyd).")
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    img = _imread(in_path)
    if img is None:
        print(f"ERROR: failed to read image: {in_path}")
        return 2

    # When running as `python tools/mark_focus.py`, sys.path[0] points to `tools/`,
    # so repo-root modules (e.g. simplefem_focus.py) are not importable by default.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    def load_simplefem():
        if args.force_py:
            raise ImportError("force-py")
        import simplefem_focus  # type: ignore
        return simplefem_focus

    try:
        simplefem_focus = load_simplefem()  # type: ignore
    except Exception as e:  # pragma: no cover
        # Fallback: explicitly load simplefem_focus.py from repo root.
        try:
            sf_py = repo_root / "simplefem_focus.py"
            if sf_py.exists():
                spec = importlib.util.spec_from_file_location("simplefem_focus", str(sf_py))
                if spec is not None and spec.loader is not None:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                    simplefem_focus = mod  # type: ignore
                else:
                    raise ImportError("spec_from_file_location returned None")
            else:
                raise ImportError(f"missing {sf_py}")
        except Exception as e2:
            print(f"ERROR: failed to import simplefem_focus: {e}; fallback load failed: {e2}")
            return 3

    anchor = None
    try:
        anchor = simplefem_focus.detect_green_intersection(img)
    except Exception as e:
        print(f"ERROR: detect_green_intersection failed: {e}")
        return 4

    if anchor is None:
        print("ERROR: focus(anchor) not found (detect_green_intersection returned None).")
        return 5

    x, y = int(anchor[0]), int(anchor[1])
    cv2.circle(img, (x, y), int(args.radius), (0, 0, 255), int(args.thickness))
    cv2.putText(img, f"anchor=({x},{y})", (x + 10, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if not _imwrite(out_path, img):
        print(f"ERROR: failed to write image: {out_path}")
        return 6

    print(f"OK: anchor=({x},{y}) saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
