"""
app.py
------
Tool Tách Nền AI + Khoét Khung Photobooth - ứng dụng desktop.

Hai chế độ:
  • 🖼️  Khung Photobooth : khoét trong suốt vùng trống (ô ảnh) của khung —
         tự động dò vùng sáng/trắng + click thủ công để chỉnh. (Mặc định)
  • 🪄  Tách chủ thể AI  : xóa nền quanh chủ thể bằng rembg (U²-Net/ISNet).

Chạy:  py -3.11 app.py   (hoặc nhấp đúp run.bat)
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from core.processor import BackgroundRemover, MODELS, DEFAULT_MODEL
from core import frame_cutter as fc

APP_TITLE = "Tool Tách Nền • Khung Photobooth + AI"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# --------------------------------------------------------------------------- #
# Tiện ích ảnh
# --------------------------------------------------------------------------- #
def checkerboard(w: int, h: int, box: int = 12) -> Image.Image:
    """Nền bàn cờ (hiển thị vùng trong suốt), dựng nhanh bằng numpy."""
    xs = (np.arange(w) // box)[None, :]
    ys = (np.arange(h) // box)[:, None]
    pattern = ((xs + ys) % 2).astype(np.uint8)
    light = np.array([235, 235, 235], np.uint8)
    dark = np.array([200, 200, 200], np.uint8)
    img = np.where(pattern[..., None] == 0, light, dark)
    return Image.fromarray(img, "RGB")


def flatten_on(img: Image.Image, bg: Image.Image | None) -> Image.Image:
    """Ghép ảnh RGBA lên nền (ảnh) để hiển thị; nếu bg None -> bàn cờ."""
    img = img.convert("RGBA")
    if bg is None:
        base = checkerboard(img.width, img.height).convert("RGBA")
    else:
        base = bg.convert("RGBA").resize(img.size)
    base.alpha_composite(img)
    return base.convert("RGB")


# --------------------------------------------------------------------------- #
# Khung hiển thị ảnh trên Canvas (hỗ trợ click + marker)
# --------------------------------------------------------------------------- #
class CanvasView:
    """Hiển thị 1 ảnh PIL vừa khung, trong suốt -> bàn cờ, cho phép click."""

    def __init__(self, parent, on_click=None):
        self.canvas = tk.Canvas(parent, bg="#242424", highlightthickness=0, bd=0)
        self.on_click = on_click
        self.pil: Image.Image | None = None
        self.behind: Image.Image | None = None   # ảnh nền hiển thị sau vùng trong suốt
        self.tkimg = None
        self.scale = 1.0
        self.ox = self.oy = 0
        self.markers: list[tuple[int, int]] = []  # tọa độ GỐC
        self.placeholder = "—"
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<Button-1>", self._click)

    def grid(self, **kw):
        self.canvas.grid(**kw)

    def set_image(self, pil: Image.Image | None, behind: Image.Image | None = None):
        self.pil = pil
        self.behind = behind
        self.render()

    def set_markers(self, pts):
        self.markers = list(pts)
        self.render()

    def _click(self, e):
        if self.pil is None or self.on_click is None:
            return
        x = int((e.x - self.ox) / self.scale)
        y = int((e.y - self.oy) / self.scale)
        if 0 <= x < self.pil.width and 0 <= y < self.pil.height:
            self.on_click(x, y)

    def render(self):
        c = self.canvas
        c.delete("all")
        cw, ch = c.winfo_width(), c.winfo_height()
        if cw < 10 or ch < 10:
            return
        if self.pil is None:
            c.create_text(cw // 2, ch // 2, text=self.placeholder,
                          fill="gray55", font=("Segoe UI", 13))
            return

        iw, ih = self.pil.size
        self.scale = min(cw / iw, ch / ih)
        dw, dh = max(1, int(iw * self.scale)), max(1, int(ih * self.scale))
        self.ox, self.oy = (cw - dw) // 2, (ch - dh) // 2

        disp = flatten_on(self.pil, self.behind).resize((dw, dh), Image.LANCZOS)
        self.tkimg = ImageTk.PhotoImage(disp)
        c.create_image(self.ox, self.oy, anchor="nw", image=self.tkimg)

        # vẽ các điểm click
        for (mx, my) in self.markers:
            px, py = self.ox + mx * self.scale, self.oy + my * self.scale
            r = 7
            c.create_oval(px - r, py - r, px + r, py + r,
                          outline="#00e5ff", width=2)
            c.create_line(px - r, py, px + r, py, fill="#00e5ff", width=2)
            c.create_line(px, py - r, px, py + r, fill="#00e5ff", width=2)


# --------------------------------------------------------------------------- #
# Ứng dụng
# --------------------------------------------------------------------------- #
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1240x760")
        self.minsize(1040, 640)

        self.remover = BackgroundRemover()

        # Trạng thái chung
        self.mode = "photobooth"
        self.source_img: Image.Image | None = None
        self.source_path: str | None = None
        self.result_img: Image.Image | None = None
        self._busy = False
        self._recompute_after = None

        # Photobooth
        self.auto_hole: np.ndarray | None = None
        self.seeds: list[tuple[int, int]] = []
        self.behind_img: Image.Image | None = None

        # AI
        self.bg_image: Image.Image | None = None

        self._build_ui()
        self._switch_mode("photobooth")

    # ====================================================================== #
    # Dựng giao diện
    # ====================================================================== #
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------- Sidebar ----------
        self.side = ctk.CTkScrollableFrame(self, width=320, corner_radius=0)
        self.side.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.side, text="TÁCH NỀN PHOTOBOOTH",
                     font=ctk.CTkFont(size=19, weight="bold")).pack(pady=(12, 2), padx=14)

        # Bộ chọn chế độ
        self.mode_seg = ctk.CTkSegmentedButton(
            self.side, values=["🖼️ Khung Photobooth", "🪄 Tách chủ thể AI"],
            command=self._on_mode_seg)
        self.mode_seg.set("🖼️ Khung Photobooth")
        self.mode_seg.pack(fill="x", padx=14, pady=(6, 10))

        # Mở ảnh (chung)
        ctk.CTkButton(self.side, text="📂  Mở ảnh khung…", height=40,
                      command=self.open_image).pack(fill="x", padx=14, pady=(0, 10))

        # ---- Khung điều khiển PHOTOBOOTH ----
        self.pb_frame = ctk.CTkFrame(self.side, fg_color="transparent")
        self._build_photobooth_controls(self.pb_frame)

        # ---- Khung điều khiển AI ----
        self.ai_frame = ctk.CTkFrame(self.side, fg_color="transparent")
        self._build_ai_controls(self.ai_frame)

        # Lưu / batch (chung)
        self.save_btn = ctk.CTkButton(self.side, text="💾  Lưu PNG trong suốt…",
                                      height=42, font=ctk.CTkFont(size=14, weight="bold"),
                                      fg_color="#2e7d32", hover_color="#1b5e20",
                                      command=self.save_result)
        self.save_btn.pack(fill="x", padx=14, pady=(14, 4), side="bottom")

        self.progress = ctk.CTkProgressBar(self.side, mode="indeterminate")
        self.progress.set(0)

        # ---------- Khu hiển thị ----------
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        main.grid_columnconfigure((0, 1), weight=1)
        main.grid_rowconfigure(1, weight=1)

        self.lbl_src = ctk.CTkLabel(main, text="Ảnh gốc  (bấm vào vùng cần khoét)",
                                    font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_src.grid(row=0, column=0, pady=3)
        ctk.CTkLabel(main, text="Kết quả (trong suốt = bàn cờ)",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=1, pady=3)

        src_wrap = ctk.CTkFrame(main, corner_radius=10)
        src_wrap.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        src_wrap.grid_rowconfigure(0, weight=1); src_wrap.grid_columnconfigure(0, weight=1)
        self.src_view = CanvasView(src_wrap, on_click=self._on_canvas_click)
        self.src_view.placeholder = "Chưa có ảnh — nhấn “📂 Mở ảnh khung…”"
        self.src_view.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        res_wrap = ctk.CTkFrame(main, corner_radius=10)
        res_wrap.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        res_wrap.grid_rowconfigure(0, weight=1); res_wrap.grid_columnconfigure(0, weight=1)
        self.res_view = CanvasView(res_wrap)
        self.res_view.placeholder = "Kết quả sẽ hiện ở đây"
        self.res_view.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # ---------- Thanh trạng thái ----------
        self.status = ctk.CTkLabel(self, text="Sẵn sàng.", anchor="w",
                                   height=24, text_color="gray70")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12)

    # --------------------- Controls: Photobooth --------------------------- #
    def _build_photobooth_controls(self, f):
        ctk.CTkButton(f, text="🪄  Tự động khoét vùng trắng", height=40,
                      command=self.pb_auto).pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(f, text="Hoặc bấm trực tiếp vào từng ô ảnh ở khung bên phải.",
                     text_color="gray60", wraplength=280, justify="left",
                     font=ctk.CTkFont(size=11)).pack(fill="x", pady=(0, 6))

        row = ctk.CTkFrame(f, fg_color="transparent"); row.pack(fill="x", pady=2)
        ctk.CTkButton(row, text="↩️ Bỏ điểm cuối", width=135,
                      fg_color="gray30", hover_color="gray25",
                      command=self.pb_undo).pack(side="left")
        ctk.CTkButton(row, text="🗑️ Xóa hết", width=135,
                      fg_color="gray30", hover_color="gray25",
                      command=self.pb_clear).pack(side="right")

        self._slider(f, "Độ nhạy màu (click)", 5, 90, 18, "tol")
        ctk.CTkLabel(f, text="— Tinh chỉnh tự động —", text_color="gray55",
                     font=ctk.CTkFont(size=11)).pack(pady=(8, 0))
        self._slider(f, "Ngưỡng độ sáng", 150, 255, 225, "bright", self._auto_param_changed)
        self._slider(f, "Độ bão hòa tối đa", 0, 90, 35, "sat", self._auto_param_changed)
        self._slider(f, "Diện tích ô tối thiểu (%)", 0, 10, 1, "area", self._auto_param_changed)
        ctk.CTkLabel(f, text="— Viền —", text_color="gray55",
                     font=ctk.CTkFont(size=11)).pack(pady=(8, 0))
        self._slider(f, "Nở/Co viền (px)", -15, 15, 0, "expand")
        self._slider(f, "Làm mịn viền", 0, 6, 1, "feather", step=0.5)

        ctk.CTkButton(f, text="🖼️  Xem thử ảnh sau khung…", height=36,
                      fg_color="gray30", hover_color="gray25",
                      command=self.pb_pick_behind).pack(fill="x", pady=(10, 2))
        ctk.CTkButton(f, text="🗂️  Khoét hàng loạt thư mục…", height=36,
                      fg_color="gray30", hover_color="gray25",
                      command=self.pb_batch).pack(fill="x", pady=2)

    # --------------------- Controls: AI ----------------------------------- #
    def _build_ai_controls(self, f):
        ctk.CTkLabel(f, text="Mô hình AI", anchor="w",
                     font=ctk.CTkFont(weight="bold")).pack(fill="x", pady=(0, 2))
        self.model_names = {v[0]: k for k, v in MODELS.items()}
        self.model_var = ctk.StringVar(value=MODELS[DEFAULT_MODEL][0])
        ctk.CTkOptionMenu(f, values=list(self.model_names.keys()),
                          variable=self.model_var,
                          command=self._on_model_change).pack(fill="x")
        self.model_desc = ctk.CTkLabel(f, text=MODELS[DEFAULT_MODEL][1],
                                       text_color="gray70", wraplength=280,
                                       justify="left", font=ctk.CTkFont(size=11))
        self.model_desc.pack(fill="x", pady=(4, 8))

        self.alpha_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(f, text="Làm mịn viền (Alpha Matting)",
                      variable=self.alpha_var).pack(fill="x", pady=4)

        ctk.CTkButton(f, text="✨  TÁCH NỀN AI", height=44,
                      font=ctk.CTkFont(size=14, weight="bold"),
                      command=self.ai_run).pack(fill="x", pady=(8, 8))

        ctk.CTkLabel(f, text="Nền sau khi tách", anchor="w",
                     font=ctk.CTkFont(weight="bold")).pack(fill="x", pady=(4, 2))
        self.bg_mode = ctk.StringVar(value="transparent")
        for val, txt in [("transparent", "Trong suốt (PNG)"),
                         ("color", "Màu đơn sắc"),
                         ("image", "Ảnh nền khác")]:
            ctk.CTkRadioButton(f, text=txt, variable=self.bg_mode, value=val,
                               command=self._ai_update_preview).pack(fill="x", padx=8, pady=2)
        row = ctk.CTkFrame(f, fg_color="transparent"); row.pack(fill="x", pady=(4, 0))
        self.color_var = ctk.StringVar(value="#FFFFFF")
        ctk.CTkEntry(row, textvariable=self.color_var, width=90).pack(side="left")
        ctk.CTkButton(row, text="Chọn ảnh nền…", width=150,
                      command=self._ai_pick_bg).pack(side="right")

    # --------------------- Helper: slider với nhãn giá trị ---------------- #
    def _slider(self, parent, label, lo, hi, init, key, cmd=None, step=1):
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(fill="x", pady=(4, 0))
        top = ctk.CTkFrame(wrap, fg_color="transparent"); top.pack(fill="x")
        ctk.CTkLabel(top, text=label, anchor="w",
                     font=ctk.CTkFont(size=12)).pack(side="left")
        vallbl = ctk.CTkLabel(top, text=str(init), width=40, anchor="e",
                              text_color="gray70", font=ctk.CTkFont(size=12))
        vallbl.pack(side="right")
        var = ctk.DoubleVar(value=init)
        steps = int((hi - lo) / step)

        def on_change(v):
            v = float(v)
            vallbl.configure(text=(f"{v:.1f}" if step < 1 else f"{int(v)}"))
            if cmd:
                cmd()
            else:
                self._schedule_recompute()

        ctk.CTkSlider(wrap, from_=lo, to=hi, number_of_steps=steps,
                      variable=var, command=on_change).pack(fill="x", pady=(2, 0))
        setattr(self, f"var_{key}", var)

    def _sv(self, key):
        return getattr(self, f"var_{key}").get()

    # ====================================================================== #
    # Chuyển chế độ
    # ====================================================================== #
    def _on_mode_seg(self, value):
        self._switch_mode("photobooth" if "Photobooth" in value else "ai")

    def _switch_mode(self, mode):
        self.mode = mode
        self.pb_frame.pack_forget()
        self.ai_frame.pack_forget()
        if mode == "photobooth":
            self.pb_frame.pack(fill="x", padx=14, after=self.side.winfo_children()[2])
            self.lbl_src.configure(text="Ảnh gốc  (bấm vào vùng cần khoét)")
            self.save_btn.configure(text="💾  Lưu PNG trong suốt…")
        else:
            self.ai_frame.pack(fill="x", padx=14, after=self.side.winfo_children()[2])
            self.lbl_src.configure(text="Ảnh gốc")
            self.save_btn.configure(text="💾  Lưu kết quả…")
        self._refresh_result_view()

    # ====================================================================== #
    # Mở ảnh
    # ====================================================================== #
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.webp *.bmp"), ("Tất cả", "*.*")])
        if not path:
            return
        try:
            self.source_img = Image.open(path).convert("RGB")
            self.source_path = path
            self.result_img = None
            self.auto_hole = None
            self.seeds = []
            self.src_view.set_image(self.source_img)
            self.src_view.set_markers([])
            self.res_view.set_image(None)
            self._set_status(f"Đã mở: {os.path.basename(path)}  "
                             f"({self.source_img.width}×{self.source_img.height})")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không mở được ảnh:\n{e}")

    # ====================================================================== #
    # PHOTOBOOTH
    # ====================================================================== #
    def _on_canvas_click(self, x, y):
        if self.mode != "photobooth" or self.source_img is None:
            return
        self.seeds.append((x, y))
        self.src_view.set_markers(self.seeds)
        self._recompute_photobooth()

    def pb_auto(self):
        if self.source_img is None:
            messagebox.showinfo("Chưa có ảnh", "Hãy mở ảnh khung trước.")
            return
        try:
            rgb = np.array(self.source_img.convert("RGB"))
            self.auto_hole = fc.detect_light_holes(
                rgb,
                brightness=int(self._sv("bright")),
                saturation_max=int(self._sv("sat")),
                min_area_ratio=self._sv("area") / 100.0,
            )
            n = self._count_regions(self.auto_hole)
            self._recompute_photobooth()
            self._set_status(f"Tự động: tìm thấy {n} vùng trống để khoét. "
                             f"Có thể bấm thêm vào ô bị sót.")
        except Exception:
            messagebox.showerror("Lỗi", traceback.format_exc())

    def _auto_param_changed(self):
        # Chỉ chạy lại dò tự động nếu đã bật auto trước đó
        if self.auto_hole is not None:
            self._schedule_recompute(rerun_auto=True)

    def pb_undo(self):
        if self.seeds:
            self.seeds.pop()
            self.src_view.set_markers(self.seeds)
            self._recompute_photobooth()

    def pb_clear(self):
        self.seeds = []
        self.auto_hole = None
        self.result_img = None
        self.src_view.set_markers([])
        self.res_view.set_image(None)
        self._set_status("Đã xóa toàn bộ vùng khoét.")

    def _schedule_recompute(self, rerun_auto=False):
        if self._recompute_after:
            self.after_cancel(self._recompute_after)
        self._recompute_after = self.after(
            120, lambda: self._recompute_photobooth(rerun_auto))

    def _recompute_photobooth(self, rerun_auto=False):
        self._recompute_after = None
        if self.mode != "photobooth" or self.source_img is None:
            return
        rgb = np.array(self.source_img.convert("RGB"))
        h, w = rgb.shape[:2]

        if rerun_auto and self.auto_hole is not None:
            self.auto_hole = fc.detect_light_holes(
                rgb,
                brightness=int(self._sv("bright")),
                saturation_max=int(self._sv("sat")),
                min_area_ratio=self._sv("area") / 100.0,
            )

        hole = np.zeros((h, w), np.uint8)
        if self.auto_hole is not None:
            hole = np.maximum(hole, self.auto_hole)
        if self.seeds:
            seed_hole = fc.build_hole_from_seeds(rgb, self.seeds,
                                                 tolerance=int(self._sv("tol")))
            hole = np.maximum(hole, seed_hole)

        if hole.max() == 0:
            self.result_img = None
            self.res_view.set_image(None)
            return

        self.result_img = fc.apply_hole_mask(
            self.source_img, hole,
            expand=int(self._sv("expand")),
            feather=float(self._sv("feather")))
        self._refresh_result_view()

    @staticmethod
    def _count_regions(hole):
        import cv2
        num, _ = cv2.connectedComponents((hole > 0).astype(np.uint8))
        return max(0, num - 1)

    def pb_pick_behind(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh chụp để xem thử sau khung",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.webp *.bmp"), ("Tất cả", "*.*")])
        if not path:
            return
        self.behind_img = Image.open(path).convert("RGB")
        self._refresh_result_view()
        self._set_status("Đang xem thử với ảnh sau khung. (Ảnh lưu ra vẫn trong suốt.)")

    # ====================================================================== #
    # AI MODE
    # ====================================================================== #
    def _on_model_change(self, display_name):
        key = self.model_names[display_name]
        self.model_desc.configure(text=MODELS[key][1])

    def _ai_pick_bg(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh làm nền",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.webp *.bmp"), ("Tất cả", "*.*")])
        if path:
            self.bg_image = Image.open(path).convert("RGB")
            self.bg_mode.set("image")
            self._ai_update_preview()

    def _ai_current_bg(self):
        mode = self.bg_mode.get()
        if mode == "color":
            return self.color_var.get().strip() or "#FFFFFF"
        if mode == "image" and self.bg_image is not None:
            return self.bg_image
        return None

    def ai_run(self):
        if self._busy:
            return
        if self.source_img is None:
            messagebox.showinfo("Chưa có ảnh", "Hãy mở ảnh trước.")
            return
        self._busy = True
        self.progress.pack(fill="x", padx=14, pady=(4, 0), side="bottom")
        self.progress.start()
        model_key = self.model_names[self.model_var.get()]
        alpha = self.alpha_var.get()
        self._set_status(f"Đang tải model “{self.model_var.get()}” và xử lý…")
        threading.Thread(target=self._ai_worker, args=(model_key, alpha),
                         daemon=True).start()

    def _ai_worker(self, model_key, alpha):
        try:
            result = self.remover.remove(self.source_img, model_key, alpha_matting=alpha)
            self.after(0, self._ai_done, result)
        except Exception:
            self.after(0, self._ai_error, traceback.format_exc())

    def _ai_done(self, result):
        self.ai_cutout = result
        self.result_img = result
        self.progress.stop(); self.progress.pack_forget()
        self._busy = False
        self._ai_update_preview()
        self._set_status("Hoàn tất! Chọn kiểu nền rồi “💾 Lưu”.")

    def _ai_error(self, err):
        self.progress.stop(); self.progress.pack_forget()
        self._busy = False
        messagebox.showerror("Lỗi xử lý", err)

    def _ai_update_preview(self):
        if getattr(self, "ai_cutout", None) is None:
            return
        try:
            composed = self.remover.apply_background(self.ai_cutout, self._ai_current_bg())
        except Exception as e:
            self._set_status(f"Nền không hợp lệ: {e}")
            composed = self.ai_cutout
        self.result_img = composed.convert("RGBA")
        self.res_view.set_image(self.result_img)

    # ====================================================================== #
    # Hiển thị kết quả
    # ====================================================================== #
    def _refresh_result_view(self):
        if self.result_img is None:
            self.res_view.set_image(None)
            return
        behind = self.behind_img if self.mode == "photobooth" else None
        self.res_view.set_image(self.result_img, behind=behind)

    # ====================================================================== #
    # Lưu
    # ====================================================================== #
    def save_result(self):
        if self.result_img is None:
            messagebox.showinfo("Chưa có kết quả",
                                "Hãy khoét/tách nền trước khi lưu.")
            return
        base = os.path.splitext(os.path.basename(self.source_path or "ketqua"))[0]
        suffix = "_khoet" if self.mode == "photobooth" else "_tachnen"
        path = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả", defaultextension=".png",
            initialfile=f"{base}{suffix}.png",
            filetypes=[("PNG (trong suốt)", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        try:
            img = self.result_img
            if path.lower().endswith((".jpg", ".jpeg")):
                flat = Image.new("RGB", img.size, (255, 255, 255))
                flat.paste(img.convert("RGBA"), (0, 0), img.convert("RGBA"))
                flat.save(path, quality=95)
            else:
                img.convert("RGBA").save(path)
            self._set_status(f"Đã lưu: {path}")
            messagebox.showinfo("Thành công", f"Đã lưu:\n{path}")
        except Exception as e:
            messagebox.showerror("Lỗi lưu file", str(e))

    # ====================================================================== #
    # Khoét hàng loạt (photobooth, tự động)
    # ====================================================================== #
    def pb_batch(self):
        if self._busy:
            return
        in_dir = filedialog.askdirectory(title="Thư mục khung đầu vào")
        if not in_dir:
            return
        out_dir = filedialog.askdirectory(title="Thư mục lưu kết quả")
        if not out_dir:
            return
        files = [f for f in os.listdir(in_dir) if f.lower().endswith(IMG_EXTS)]
        if not files:
            messagebox.showinfo("Trống", "Không có ảnh trong thư mục.")
            return
        params = dict(brightness=int(self._sv("bright")),
                      saturation_max=int(self._sv("sat")),
                      min_area_ratio=self._sv("area") / 100.0,
                      expand=int(self._sv("expand")),
                      feather=float(self._sv("feather")))
        self._busy = True
        self.progress.configure(mode="determinate")
        self.progress.pack(fill="x", padx=14, pady=(4, 0), side="bottom")
        self.progress.set(0)
        threading.Thread(target=self._pb_batch_worker,
                         args=(files, in_dir, out_dir, params), daemon=True).start()

    def _pb_batch_worker(self, files, in_dir, out_dir, params):
        total, ok = len(files), 0
        for i, name in enumerate(files, 1):
            try:
                img = Image.open(os.path.join(in_dir, name)).convert("RGB")
                out, _ = fc.cut_auto(img, **params)
                out.save(os.path.join(out_dir, os.path.splitext(name)[0] + "_khoet.png"))
                ok += 1
            except Exception:
                pass
            self.after(0, lambda i=i: (self.progress.set(i / total),
                                       self._set_status(f"Hàng loạt: {i}/{total}…")))
        self.after(0, self._pb_batch_done, ok, total, out_dir)

    def _pb_batch_done(self, ok, total, out_dir):
        self.progress.pack_forget()
        self.progress.configure(mode="indeterminate")
        self._busy = False
        self._set_status(f"Xong: {ok}/{total} khung → {out_dir}")
        messagebox.showinfo("Hoàn tất",
                            f"Đã khoét {ok}/{total} khung.\nLưu tại:\n{out_dir}")

    # ------------------------------------------------------------------ #
    def _set_status(self, msg):
        self.status.configure(text=msg)
        self.update_idletasks()


if __name__ == "__main__":
    App().mainloop()
