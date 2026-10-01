"""
main.py
-------
Điểm vào cho bản đóng gói (.exe). Bọc toàn bộ app trong try/except để nếu có
lỗi khi khởi động (kể cả lỗi import), nó được ghi ra error_log.txt cạnh file exe
thay vì tắt im lặng.
"""

import os
import sys
import traceback


def _logdir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _selftest(img_path: str) -> None:
    """Tự kiểm tra cả 2 engine trong bản đóng gói (dùng khi test exe)."""
    from PIL import Image
    from core import frame_cutter as fc
    from core.processor import BackgroundRemover

    img = Image.open(img_path).convert("RGB")

    # 1) Khoét khung (OpenCV)
    out, _ = fc.cut_auto(img, feather=1.5)
    print("SELFTEST frame_cutter OK:", out.size, out.mode)

    # 2) Tách nền AI (rembg + onnxruntime + numba/pymatting)
    rem = BackgroundRemover()
    res = rem.remove(img, "u2net")
    print("SELFTEST AI OK:", res.size, res.mode)
    print("SELFTEST PASSED")


def main():
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        _selftest(sys.argv[i + 1])
        return
    try:
        from app import App
        App().mainloop()
    except Exception:
        err = traceback.format_exc()
        try:
            with open(os.path.join(_logdir(), "error_log.txt"), "w", encoding="utf-8") as f:
                f.write(err)
        except Exception:
            pass
        # Hiện hộp thoại lỗi nếu có thể
        try:
            import tkinter.messagebox as mb
            mb.showerror("Lỗi khởi động", err[-1500:])
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
