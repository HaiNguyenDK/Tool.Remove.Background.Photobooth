"""
cli.py
------
Công cụ dòng lệnh: khoét khung photobooth (mặc định) hoặc tách chủ thể AI.

── Khoét khung photobooth (tự động dò vùng trắng ở giữa) ──
    py -3.11 cli.py khung.png
    py -3.11 cli.py khung.png -o ketqua.png --bright 230 --sat 30 --feather 2
    py -3.11 cli.py ./thumuc_khung --batch -o ./ketqua

── Tách chủ thể bằng AI (rembg) ──
    py -3.11 cli.py anh.jpg --ai
    py -3.11 cli.py anh.jpg --ai -m isnet-general-use --alpha --bg "#FFFFFF"
    py -3.11 cli.py ./anh --ai --batch -o ./ketqua
"""

from __future__ import annotations

import argparse
import os
import sys

# In được tiếng Việt trên console Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from PIL import Image

from core.processor import BackgroundRemover, MODELS, DEFAULT_MODEL
from core import frame_cutter as fc

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def parse_bg(bg):
    if not bg:
        return None
    return bg if bg.startswith("#") else Image.open(bg).convert("RGB")


def main():
    p = argparse.ArgumentParser(description="Tool khoét khung photobooth / tách nền AI")
    p.add_argument("input", help="Ảnh đầu vào, hoặc thư mục nếu --batch")
    p.add_argument("-o", "--output", help="File/thư mục đầu ra")
    p.add_argument("--batch", action="store_true", help="Xử lý cả thư mục")

    # Chế độ
    p.add_argument("--ai", action="store_true",
                   help="Dùng tách chủ thể AI thay vì khoét khung")

    # Khoét khung
    p.add_argument("--bright", type=int, default=225, help="[khung] Ngưỡng độ sáng (150-255)")
    p.add_argument("--sat", type=int, default=35, help="[khung] Độ bão hòa tối đa (0-90)")
    p.add_argument("--min-area", type=float, default=1.0, help="[khung] Diện tích ô tối thiểu (%%)")
    p.add_argument("--expand", type=int, default=0, help="[khung] Nở/co viền (px)")
    p.add_argument("--feather", type=float, default=1.5, help="[khung] Làm mịn viền")

    # AI
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()),
                   help="[AI] Model")
    p.add_argument("--alpha", action="store_true", help="[AI] Bật alpha matting")
    p.add_argument("--bg", help='[AI] Nền mới: "#RRGGBB" hoặc ảnh. Mặc định trong suốt')
    args = p.parse_args()

    remover = BackgroundRemover() if args.ai else None
    bg = parse_bg(args.bg) if args.ai else None

    def process_one(src, dst):
        img = Image.open(src).convert("RGB")
        if args.ai:
            res = remover.remove(img, args.model, alpha_matting=args.alpha)
            out = remover.apply_background(res, bg)
        else:
            out, _ = fc.cut_auto(
                img, brightness=args.bright, saturation_max=args.sat,
                min_area_ratio=args.min_area / 100.0,
                expand=args.expand, feather=args.feather)
        if dst.lower().endswith((".jpg", ".jpeg")):
            flat = Image.new("RGB", out.size, (255, 255, 255))
            flat.paste(out.convert("RGBA"), (0, 0), out.convert("RGBA"))
            flat.save(dst, quality=95)
        else:
            out.save(dst)
        print(f"  ✓ {os.path.basename(src)} -> {dst}")

    mode = "tách chủ thể AI" if args.ai else "khoét khung photobooth"
    suffix = "_tachnen" if args.ai else "_khoet"

    if args.batch:
        in_dir = args.input
        out_dir = args.output or os.path.join(in_dir, "ketqua")
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(in_dir) if f.lower().endswith(EXTS)]
        if not files:
            print("Không tìm thấy ảnh.", file=sys.stderr); sys.exit(1)
        print(f"[{mode}] Xử lý {len(files)} ảnh…")
        for f in files:
            dst = os.path.join(out_dir, os.path.splitext(f)[0] + suffix + ".png")
            try:
                process_one(os.path.join(in_dir, f), dst)
            except Exception as e:
                print(f"  ✗ {f}: {e}", file=sys.stderr)
        print(f"Xong. Kết quả tại: {out_dir}")
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        dst = args.output or f"{base}{suffix}.png"
        print(f"[{mode}] Xử lý '{args.input}'…")
        process_one(args.input, dst)
        print("Hoàn tất.")


if __name__ == "__main__":
    main()
