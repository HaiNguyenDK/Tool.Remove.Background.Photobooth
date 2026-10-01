# 🖼️ Tool Tách Nền Photobooth + AI

Ứng dụng desktop (Python) phục vụ **vận hành photobooth**: tự động **khoét trong suốt vùng trống (ô ảnh) của khung** để ghép ảnh chụp vào phía sau. Kèm chế độ **tách chủ thể bằng AI**.

Hai chế độ:

| Chế độ | Mục đích | Công nghệ |
|--------|----------|-----------|
| 🖼️ **Khung Photobooth** *(chính)* | Khoét trong suốt ô ảnh ở giữa khung, giữ nguyên viền trang trí | OpenCV: flood-fill + connected-components |
| 🪄 **Tách chủ thể AI** | Xóa nền quanh người/vật thể | `rembg` (U²-Net / ISNet, ONNX) |

Toàn bộ chạy **offline** trên máy — an toàn cho ảnh riêng tư.

---

## ✨ Tính năng chế độ Photobooth

- 🪄 **Tự động khoét** mọi vùng trắng/sáng lớn (hỗ trợ **nhiều ô ảnh** trong một khung).
- 🖱️ **Click thủ công** vào từng ô cần khoét — flood-fill lan đến sát viền khung, chính xác tuyệt đối cho khung khó.
- 🎚️ Slider tinh chỉnh: độ nhạy màu, ngưỡng sáng, bão hòa, diện tích ô, **nở/co viền**, **làm mịn viền**.
- 🧠 Thuật toán thông minh: tự loại dải trắng trang trí (giữ vùng ≥30% ô lớn nhất), bỏ vùng chạm mép, trám chữ/logo lọt trong ô.
- 👁️ **Xem thử ảnh sau khung**: nạp một ảnh chụp mẫu để kiểm tra căn chỉnh trước khi vận hành.
- 🗂️ **Khoét hàng loạt** cả thư mục khung (tự động).
- 💾 Xuất **PNG trong suốt** sẵn sàng cho phần mềm photobooth.

---

## 📦 Cài đặt

Yêu cầu **Python 3.11** (khuyến nghị).

```bash
py -3.11 -m pip install -r requirements.txt
```

> Chế độ AI lần đầu sẽ tự tải weights (~176MB). Chế độ Photobooth **không cần tải gì thêm**.

---

## 🚀 Sử dụng

### Giao diện (khuyến nghị cho vận hành)

```bash
py -3.11 app.py
```
Hoặc nhấp đúp **`run.bat`**.

**Quy trình photobooth:**
1. 📂 **Mở ảnh khung**.
2. Nhấn **🪄 Tự động khoét** — hoặc **bấm trực tiếp** vào từng ô ảnh ở khung bên phải.
3. (Tùy chọn) chỉnh slider **làm mịn / nở-co viền**; bấm **🖼️ Xem thử ảnh sau khung** để kiểm tra.
4. 💾 **Lưu PNG trong suốt**.

> Mẹo: nếu auto bắt nhầm → bấm **🗑️ Xóa hết** rồi **click tay** từng ô (độ nhạy màu ~15–20 cho kết quả sạch nhất). **↩️ Bỏ điểm cuối** để hoàn tác.

### Dòng lệnh (CLI)

```bash
# Khoét 1 khung (tự động)
py -3.11 cli.py khung.png -o ketqua.png --feather 2

# Khoét cả thư mục khung
py -3.11 cli.py ./khung_vao --batch -o ./khung_ra

# Tinh chỉnh dò vùng trắng
py -3.11 cli.py khung.png --bright 230 --sat 30 --min-area 2

# Chế độ tách chủ thể AI
py -3.11 cli.py anh.jpg --ai -m isnet-general-use --alpha --bg "#FFFFFF"
```

---

## 📦 Đóng gói file .exe (Windows)

Đã có sẵn cấu hình. Chỉ cần:

```bash
py -3.11 -m pip install pyinstaller
py -3.11 -m PyInstaller build_exe.spec --noconfirm
```

Kết quả: thư mục **`dist/TachNenPhotobooth/`** chứa **`TachNenPhotobooth.exe`**.
Copy **cả thư mục** này sang máy khác để chạy (không cần cài Python).

> Bản exe gồm cả 2 chế độ. Chế độ AI lần đầu vẫn tải model (~176MB) về `~/.rembg/`.
> Nếu exe không mở được, xem file `error_log.txt` sinh ra cạnh exe.

---

## 📁 Cấu trúc dự án

```
tach-nen-tool/
├── app.py                  # GUI (customtkinter) — 2 chế độ
├── cli.py                  # Công cụ dòng lệnh
├── core/
│   ├── processor.py        # Tách chủ thể AI (rembg)
│   └── frame_cutter.py     # Khoét khung photobooth (OpenCV)
├── requirements.txt
├── run.bat                 # Launcher Windows
└── README.md
```

---

## ⚙️ Cơ chế khoét khung (Computer Vision)

1. **Dò vùng ứng viên**: chuyển HSV, lấy pixel **sáng** (V cao) & **ít màu** (S thấp) → gần trắng.
2. **Phân vùng**: `connectedComponents` tách thành các vùng rời; lọc theo diện tích, loại vùng chạm mép, chỉ giữ vùng đủ lớn so với ô lớn nhất → đúng các "ô ảnh".
3. **Trám đặc** mỗi ô (lấp chữ/logo lọt trong).
4. **Click thủ công** dùng `floodFill` có ngưỡng màu, lan từ điểm bấm đến sát viền.
5. **Hậu xử lý viền**: nở/co (morphology) + làm mịn (Gaussian) → đưa vào kênh **alpha** → PNG trong suốt.
```
