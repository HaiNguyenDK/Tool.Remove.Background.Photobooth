"""
processor.py
------------
Lớp lõi xử lý tách nền bằng AI (bọc thư viện rembg).

rembg sử dụng các mô hình học sâu (U^2-Net, ISNet...) chạy trên ONNX Runtime
để dự đoán mặt nạ (mask) cho vật thể tiền cảnh (foreground). Module này cung
cấp một API gọn gàng, lazy-load model theo tên và cache lại session để tái dùng.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional, Tuple

from PIL import Image, ImageColor


# Danh sách model AI hỗ trợ -> (tên hiển thị, mô tả ngắn)
# Tên key phải khớp model_name của rembg.
MODELS: dict[str, Tuple[str, str]] = {
    "u2net":            ("U²-Net (Mặc định)", "Cân bằng tốt giữa tốc độ và chất lượng. Phù hợp đa số ảnh."),
    "isnet-general-use":("ISNet General",      "Chất lượng cao, viền sắc nét cho ảnh tổng quát."),
    "u2netp":           ("U²-Net Lite",        "Bản nhẹ, nhanh hơn, độ chính xác thấp hơn một chút."),
    "u2net_human_seg":  ("Human Seg",          "Chuyên tách người (chân dung, toàn thân)."),
    "silueta":          ("Silueta",            "Model nhỏ (~43MB), nhanh, tiết kiệm bộ nhớ."),
    "isnet-anime":      ("Anime",              "Tối ưu cho ảnh anime / nhân vật 2D."),
}

DEFAULT_MODEL = "u2net"


class BackgroundRemover:
    """Bọc rembg: quản lý session theo model và thực thi tách nền."""

    def __init__(self) -> None:
        # cache: model_name -> rembg session
        self._sessions: dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # Quản lý model
    # ------------------------------------------------------------------ #
    def _get_session(self, model_name: str):
        """Lazy-load và cache session cho model. Import rembg trong hàm để
        khởi động GUI không bị chậm vì nạp onnxruntime ngay lập tức."""
        if model_name not in self._sessions:
            from rembg import new_session  # import trễ
            self._sessions[model_name] = new_session(model_name)
        return self._sessions[model_name]

    def preload(self, model_name: str = DEFAULT_MODEL) -> None:
        """Tải trước model (tải weights về nếu chưa có) để lần xử lý đầu nhanh hơn."""
        self._get_session(model_name)

    # ------------------------------------------------------------------ #
    # Xử lý chính
    # ------------------------------------------------------------------ #
    def remove(
        self,
        image: Image.Image,
        model_name: str = DEFAULT_MODEL,
        *,
        alpha_matting: bool = False,
        alpha_fg_threshold: int = 240,
        alpha_bg_threshold: int = 10,
        alpha_erode: int = 10,
        post_process_mask: bool = True,
    ) -> Image.Image:
        """
        Tách nền một ảnh PIL, trả về ảnh RGBA (nền trong suốt).

        alpha_matting: bật kỹ thuật alpha matting giúp viền tóc/lông mượt hơn
        (chậm hơn). Các ngưỡng fg/bg/erode điều chỉnh độ mịn của viền.
        """
        from rembg import remove as rembg_remove  # import trễ

        session = self._get_session(model_name)
        result = rembg_remove(
            image,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_fg_threshold,
            alpha_matting_background_threshold=alpha_bg_threshold,
            alpha_matting_erode_size=alpha_erode,
            post_process_mask=post_process_mask,
        )
        if not isinstance(result, Image.Image):
            result = Image.open(BytesIO(result))
        return result.convert("RGBA")

    # ------------------------------------------------------------------ #
    # Ghép nền mới
    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_background(
        cutout: Image.Image,
        bg: Optional[str | Image.Image] = None,
    ) -> Image.Image:
        """
        Ghép ảnh đã tách (RGBA) lên một nền.

        bg=None           -> giữ nền trong suốt (trả về RGBA).
        bg="#RRGGBB"      -> nền màu đặc.
        bg=PIL.Image      -> nền là ảnh khác (được resize/crop cho khớp).
        """
        cutout = cutout.convert("RGBA")
        if bg is None:
            return cutout

        if isinstance(bg, str):
            color = ImageColor.getrgb(bg)
            background = Image.new("RGBA", cutout.size, color + (255,))
        else:
            background = BackgroundRemover._fit_cover(bg.convert("RGBA"), cutout.size)

        background.alpha_composite(cutout)
        return background.convert("RGB")

    @staticmethod
    def _fit_cover(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Resize + center-crop ảnh nền để phủ kín khung (giống object-fit: cover)."""
        tw, th = size
        iw, ih = img.size
        scale = max(tw / iw, th / ih)
        nw, nh = int(iw * scale + 0.5), int(ih * scale + 0.5)
        img = img.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - tw) // 2, (nh - th) // 2
        return img.crop((left, top, left + tw, top + th))
