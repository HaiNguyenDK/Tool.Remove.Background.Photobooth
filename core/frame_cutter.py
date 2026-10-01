"""
frame_cutter.py
---------------
Khoét trong suốt vùng trống của khung ảnh PHOTOBOOTH.

Khác với tách nền AI (giữ chủ thể, xóa xung quanh), ở đây ta làm NGƯỢC LẠI:
giữ nguyên phần viền/khung trang trí và biến các "ô ảnh" (vùng trống, thường
màu trắng/sáng ở giữa) thành TRONG SUỐT — để sau này ghép ảnh chụp vào phía sau.

Hai cơ chế:
  • cut_auto()      : tự động dò mọi vùng sáng/đồng nhất lớn (xử lý nhiều ô).
  • cut_by_seeds()  : flood-fill từ các điểm người dùng click (chỉnh tay).

Dùng OpenCV cho flood-fill, connected-components và làm mịn viền.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Tiện ích mask -> ảnh RGBA
# --------------------------------------------------------------------------- #
def _refine_mask(hole: np.ndarray, expand: int, feather: float) -> np.ndarray:
    """Tinh chỉnh mask lỗ (255 = khoét). Trả về kênh alpha (255 = giữ, 0 = trong suốt)."""
    hole = hole.copy()

    # Nở (expand>0) hoặc co (expand<0) vùng khoét để ăn sát/lùi khỏi viền
    if expand != 0:
        ksize = abs(expand) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        hole = cv2.dilate(hole, kernel) if expand > 0 else cv2.erode(hole, kernel)

    alpha = 255 - hole  # giữ phần KHÔNG phải lỗ

    # Feather: làm mềm biên để tránh răng cưa / viền trắng
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)

    return alpha


def _compose_rgba(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    out = np.dstack([rgb, alpha]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


# --------------------------------------------------------------------------- #
# 1) Flood-fill từ các điểm click (chỉnh tay)
# --------------------------------------------------------------------------- #
def build_hole_from_seeds(
    rgb: np.ndarray,
    seeds: Iterable[Tuple[int, int]],
    tolerance: int = 20,
) -> np.ndarray:
    """
    Trả về mask lỗ (uint8, 255 = vùng được flood-fill từ các seed).

    Mỗi seed (x, y) là tọa độ pixel trên ảnh GỐC. Flood-fill lan ra các pixel
    có màu gần giống seed (trong ngưỡng tolerance) và dừng lại ở viền khung.
    """
    h, w = rgb.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)  # floodFill yêu cầu mask lớn hơn 2px
    lo = (tolerance,) * 3
    up = (tolerance,) * 3
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for (x, y) in seeds:
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        cv2.floodFill(bgr, mask, (x, y), 0, lo, up, flags)

    return mask[1:-1, 1:-1] * 255 if mask.max() == 1 else mask[1:-1, 1:-1]


def cut_by_seeds(
    image: Image.Image,
    seeds: Iterable[Tuple[int, int]],
    *,
    tolerance: int = 20,
    expand: int = 0,
    feather: float = 1.5,
) -> Tuple[Image.Image, np.ndarray]:
    """Khoét ảnh theo các điểm click. Trả về (ảnh RGBA, mask lỗ)."""
    rgb = _to_rgb_array(image)
    hole = build_hole_from_seeds(rgb, seeds, tolerance)
    alpha = _refine_mask(hole, expand, feather)
    return _compose_rgba(rgb, alpha), hole


# --------------------------------------------------------------------------- #
# 2) Tự động dò vùng sáng/trắng lớn (nhiều ô)
# --------------------------------------------------------------------------- #
def detect_light_holes(
    rgb: np.ndarray,
    *,
    brightness: int = 225,
    saturation_max: int = 35,
    min_area_ratio: float = 0.01,
    rel_area_ratio: float = 0.30,
    ignore_border: bool = True,
) -> np.ndarray:
    """
    Dò các vùng SÁNG & ÍT MÀU (gần trắng) đủ lớn — ứng viên cho "ô ảnh".

    brightness      : ngưỡng độ sáng V (HSV) tối thiểu để coi là sáng.
    saturation_max  : độ bão hòa tối đa (gần trắng/xám -> bão hòa thấp).
    min_area_ratio  : diện tích tối thiểu tuyệt đối (theo % ảnh) để xét.
    rel_area_ratio  : chỉ giữ vùng có diện tích >= tỉ lệ này so với vùng LỚN NHẤT.
                      Giúp loại các dải trắng trang trí nhỏ, chỉ giữ ô ảnh thật
                      (các ô ảnh thật thường lớn và xấp xỉ nhau).
    ignore_border   : bỏ qua vùng chạm sát mép ảnh (thường là nền ngoài, không
                      phải ô ảnh bên trong khung).
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    candidate = ((v >= brightness) & (s <= saturation_max)).astype(np.uint8)

    # Làm sạch nhiễu nhỏ và trám lỗ kim
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    min_area = min_area_ratio * h * w

    # Lượt 1: lọc theo diện tích tuyệt đối + vị trí (không chạm mép)
    kept = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if ignore_border:
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            touches = x <= 1 or y <= 1 or (x + cw) >= w - 1 or (y + ch) >= h - 1
            # Ô ảnh của khung photobooth luôn nằm BÊN TRONG, không chạm mép.
            # Vùng sáng chạm mép (mây, nền ngoài...) bị loại để tránh khoét nhầm.
            if touches:
                continue
        kept.append((i, area))

    hole = np.zeros((h, w), np.uint8)
    if not kept:
        return hole

    # Lượt 2: chỉ giữ vùng đủ lớn so với vùng lớn nhất (loại dải trang trí nhỏ)
    max_area = max(a for _, a in kept)
    for i, area in kept:
        if area < rel_area_ratio * max_area:
            continue
        # Trám kín từng vùng (lấp các "đảo" chữ/logo nằm trong ô nếu có)
        comp = (labels == i).astype(np.uint8)
        comp = _fill_holes(comp)
        hole[comp > 0] = 255

    return hole


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Trám các lỗ bên trong một vùng nhị phân (giữ vùng đặc)."""
    ff = binary.copy()
    h, w = binary.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)       # tô nền ngoài
    holes = (ff == 0).astype(np.uint8)        # phần còn 0 là lỗ bên trong
    return ((binary > 0) | (holes > 0)).astype(np.uint8)


def cut_auto(
    image: Image.Image,
    *,
    brightness: int = 225,
    saturation_max: int = 35,
    min_area_ratio: float = 0.01,
    fill_interior_islands: bool = True,
    expand: int = 0,
    feather: float = 1.5,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Tự động khoét mọi vùng sáng/trắng lớn trong khung. Trả về (ảnh RGBA, mask lỗ).

    fill_interior_islands: nếu True, lấp các chi tiết nhỏ (chữ/logo) nằm lọt
    trong ô ảnh để toàn ô trong suốt liền mạch (đã xử lý trong detect_light_holes).
    """
    rgb = _to_rgb_array(image)
    hole = detect_light_holes(
        rgb,
        brightness=brightness,
        saturation_max=saturation_max,
        min_area_ratio=min_area_ratio,
    )
    alpha = _refine_mask(hole, expand, feather)
    return _compose_rgba(rgb, alpha), hole


# --------------------------------------------------------------------------- #
# Kết hợp: dùng mask có sẵn + thêm seed (cho GUI cộng dồn)
# --------------------------------------------------------------------------- #
def apply_hole_mask(
    image: Image.Image,
    hole: np.ndarray,
    *,
    expand: int = 0,
    feather: float = 1.5,
) -> Image.Image:
    """Áp một mask lỗ đã dựng sẵn lên ảnh (dùng khi GUI gộp auto + click)."""
    rgb = _to_rgb_array(image)
    alpha = _refine_mask(hole, expand, feather)
    return _compose_rgba(rgb, alpha)
