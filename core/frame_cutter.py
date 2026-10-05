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
def _refine_mask(
    rgb: np.ndarray,
    hole: np.ndarray,
    expand: int = 0,
    feather: float = 0.0,
    despill_band: int = 2,
    rim_white: int = 205,
) -> np.ndarray:
    """Tinh chỉnh mask lỗ -> alpha. (255 = giữ, 0 = trong suốt).

    Chống "hào quang" (quầng sáng ở mép) + soft edge tự nhiên:
      1) expand: nở(>0)/co(<0) lõi lỗ.
      2) GỘP RÌA GẦN-TRẮNG vào lỗ: các pixel sát biên có min(R,G,B) >= rim_white
         được coi là phần cần khoét -> cắt sát, loại hẳn rìa sáng gây hào quang.
      3) SOFT EDGE MỘT PHÍA: chỉ làm mềm về PHÍA TRONG lỗ (nơi ảnh ghép hiện ra),
         giữ mép khung sắc nét -> mềm mượt mà KHÔNG làm sáng mép khung.

    feather      : bán kính làm mềm vào trong lỗ (px). 0 = mép sắc.
    despill_band : bề rộng dải quét rìa gần-trắng để gộp (px).
    rim_white    : ngưỡng min(R,G,B) để coi là rìa trắng cần cắt (thấp = cắt mạnh).
    """
    hole = hole.copy()

    if expand != 0:
        ksize = abs(expand) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        hole = cv2.dilate(hole, kernel) if expand > 0 else cv2.erode(hole, kernel)

    # (2) Gộp rìa gần-trắng quanh biên vào lỗ (cắt sát, bỏ hào quang)
    if despill_band > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * despill_band + 1, 2 * despill_band + 1))
        roi = cv2.dilate(hole, kernel)
        minc = rgb.min(axis=2)
        rim = (roi > 0) & (hole == 0) & (minc >= rim_white)
        hole[rim] = 255

    # (3) Soft edge một phía: mép khung giữ 255, chỉ ramp xuống 0 đi vào trong lỗ
    if feather > 0:
        base = np.where(hole > 0, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(base, (0, 0), feather)
        alpha = np.where(hole > 0, blurred, 255).astype(np.uint8)
    else:
        alpha = np.where(hole > 0, 0, 255).astype(np.uint8)

    return alpha


def _compose_rgba(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    out = np.dstack([rgb, alpha]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


# --------------------------------------------------------------------------- #
# 1) Flood-fill từ các điểm click (chỉnh tay)
# --------------------------------------------------------------------------- #
def edge_barrier(
    rgb: np.ndarray,
    canny_lo: int = 25,
    canny_hi: int = 80,
    wall: int = 1,
) -> np.ndarray:
    """
    Bản đồ CẠNH dùng làm "tường chặn" cho flood-fill.

    Dù ô ảnh trắng có dính liền MÀU với mảng trắng của trang trí (mái nhà, tường
    cao ốc), giữa chúng hầu như luôn có một đường viền/nét mảnh. Phát hiện cạnh
    (Canny) rồi làm tường chặn giúp flood DỪNG tại viền đó, không ăn sang trang trí.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    if wall > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * wall + 1, 2 * wall + 1))
        edges = cv2.dilate(edges, k)
    return edges


def build_hole_from_seeds(
    rgb: np.ndarray,
    seeds: Iterable[Tuple[int, int]],
    tolerance: int = 16,
    fixed_range: bool = True,
    use_edges: bool = True,
) -> np.ndarray:
    """
    Trả về mask lỗ (uint8, 255 = vùng được flood-fill từ các seed).

    Mỗi seed (x, y) là tọa độ pixel trên ảnh GỐC.

    fixed_range=True (mặc định): "loang theo đúng màu seed" — một pixel được lấy
    nếu nó nằm trong ±tolerance so với CHÍNH MÀU SEED (thường là #ffffff). Lan
    sang các pixel gần-trắng liền kề và DỪNG NGAY khi gặp pixel khác màu. Không
    bị "creep" trôi dần qua vùng chuyển màu như chế độ dải động.

    use_edges=True: thêm tường chặn bằng bản đồ cạnh (xem edge_barrier) để flood
    không tràn qua viền sang mảng trắng của trang trí (chống "ăn góc tòa nhà").
    """
    h, w = rgb.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)  # floodFill yêu cầu mask lớn hơn 2px
    if use_edges:
        mask[1:-1, 1:-1][edge_barrier(rgb) > 0] = 1  # tường chặn (giá trị != 0)

    lo = (tolerance,) * 3
    up = (tolerance,) * 3
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    if fixed_range:
        flags |= cv2.FLOODFILL_FIXED_RANGE

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for (x, y) in seeds:
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        if mask[y + 1, x + 1] != 0:  # seed rơi trúng tường -> bỏ qua
            continue
        cv2.floodFill(bgr, mask, (x, y), 0, lo, up, flags)

    # Chỉ lấy phần được flood (giá trị 255), BỎ các pixel tường chặn (giá trị 1)
    hole = (mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
    # Nối lại các nét cạnh mảnh NẰM TRONG ô (kernel nhỏ, không đủ để bắc cầu sang
    # trang trí vì tường dày ~3px).
    hole = cv2.morphologyEx(
        hole, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return hole


def cut_by_seeds(
    image: Image.Image,
    seeds: Iterable[Tuple[int, int]],
    *,
    tolerance: int = 20,
    expand: int = 0,
    feather: float = 0.0,
) -> Tuple[Image.Image, np.ndarray]:
    """Khoét ảnh theo các điểm click. Trả về (ảnh RGBA, mask lỗ)."""
    rgb = _to_rgb_array(image)
    hole = build_hole_from_seeds(rgb, seeds, tolerance)
    alpha = _refine_mask(rgb, hole, expand, feather)
    return _compose_rgba(rgb, alpha), hole


# --------------------------------------------------------------------------- #
# 2) Tự động dò vùng sáng/trắng lớn (nhiều ô)
# --------------------------------------------------------------------------- #
def detect_light_holes(
    rgb: np.ndarray,
    *,
    brightness: int = 230,
    saturation_max: int = 15,
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

    # Làm sạch nhiễu nhỏ và trám lỗ kim. Kernel nhỏ (3x3) để biên bám chi tiết
    # (khe trắng mảnh giữa lá/tòa nhà được giữ), chỉ phá cầu nối 1px giả.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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


def detect_holes_floodfill(
    rgb: np.ndarray,
    *,
    brightness: int = 230,
    saturation_max: int = 15,
    min_area_ratio: float = 0.01,
    rel_area_ratio: float = 0.30,
    tolerance: int = 16,
) -> np.ndarray:
    """
    Dò vị trí ô ảnh (bằng ngưỡng màu) rồi FLOOD-FILL từ bên trong mỗi ô để lấy
    đúng vùng ô (bị chặn bởi ranh giới khung). Trả về mask lỗ (255 = khoét).

    Ưu điểm so với ngưỡng toàn cục: KHÔNG ăn nhầm các mảng trắng của trang trí
    (mái nhà, tường cao ốc cũng màu trắng 254) vì flood-fill không lan qua ranh
    giới ô ảnh. Giống hệt khi người dùng click tay vào giữa ô.
    """
    # 1) Tìm ứng viên ô (đã lọc diện tích/vị trí/tỉ lệ)
    seed_regions = detect_light_holes(
        rgb,
        brightness=brightness,
        saturation_max=saturation_max,
        min_area_ratio=min_area_ratio,
        rel_area_ratio=rel_area_ratio,
    )
    if seed_regions.max() == 0:
        return seed_regions

    # 2) Mỗi ô -> chọn điểm sâu nhất bên trong (distance transform) làm seed
    num, labels = cv2.connectedComponents((seed_regions > 0).astype(np.uint8))
    seeds = []
    for i in range(1, num):
        comp = (labels == i).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        _, _, _, maxloc = cv2.minMaxLoc(dist)   # (x, y) nằm sâu trong ô
        seeds.append(maxloc)

    # 3) Flood-fill từ các seed (bị chặn ở ranh giới ô) + trám đảo bên trong
    hole = build_hole_from_seeds(rgb, seeds, tolerance=tolerance)
    num2, labels2 = cv2.connectedComponents((hole > 0).astype(np.uint8))
    filled = np.zeros_like(hole)
    for i in range(1, num2):
        comp = _fill_holes((labels2 == i).astype(np.uint8))
        filled[comp > 0] = 255
    return filled


def cut_auto(
    image: Image.Image,
    *,
    brightness: int = 230,
    saturation_max: int = 15,
    min_area_ratio: float = 0.01,
    tolerance: int = 16,
    fill_interior_islands: bool = True,
    expand: int = 0,
    feather: float = 0.0,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Tự động khoét các ô ảnh trong khung bằng dò-vị-trí + flood-fill (bám chi tiết,
    không ăn trang trí trắng). Trả về (ảnh RGBA, mask lỗ).
    """
    rgb = _to_rgb_array(image)
    hole = detect_holes_floodfill(
        rgb,
        brightness=brightness,
        saturation_max=saturation_max,
        min_area_ratio=min_area_ratio,
        tolerance=tolerance,
    )
    alpha = _refine_mask(rgb, hole, expand, feather)
    return _compose_rgba(rgb, alpha), hole


# --------------------------------------------------------------------------- #
# Kết hợp: dùng mask có sẵn + thêm seed (cho GUI cộng dồn)
# --------------------------------------------------------------------------- #
def apply_hole_mask(
    image: Image.Image,
    hole: np.ndarray,
    *,
    expand: int = 0,
    feather: float = 0.0,
) -> Image.Image:
    """Áp một mask lỗ đã dựng sẵn lên ảnh (dùng khi GUI gộp auto + click)."""
    rgb = _to_rgb_array(image)
    alpha = _refine_mask(rgb, hole, expand, feather)
    return _compose_rgba(rgb, alpha)
