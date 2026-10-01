# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — đóng gói Tool Tách Nền Photobooth + AI (onedir).

Build:
    py -3.11 -m PyInstaller build_exe.spec --noconfirm
Kết quả: dist/TachNenPhotobooth/TachNenPhotobooth.exe
"""

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# Các gói cần gom trọn vẹn (data + binary + submodule) để chạy được khi đóng gói
for pkg in [
    "customtkinter",   # theme/asset JSON
    "onnxruntime",     # DLL provider cho AI
    "rembg",           # engine tách nền AI
    "pymatting",       # alpha matting
    "numba",           # JIT cho pymatting
    "llvmlite",        # backend của numba (DLL)
    "scipy",           # phụ thuộc của pymatting
    "PIL",
]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += ["cv2", "numpy"]


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "pytest", "tkinter.test", "IPython", "notebook"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TachNenPhotobooth",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # ứng dụng cửa sổ, không hiện console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TachNenPhotobooth",
)
