# -*- mode: python ; coding: utf-8 -*-
# Build with:  pyinstaller PassportSheet.spec
# Run download_models.py first so ./models contains both ONNX files.

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = [
    ("requirements.json", "."),
    ("assets/icon.ico", "assets"),
    ("models/face_detection_yunet_2023mar.onnx", "models"),
    ("models/u2net.onnx", "models"),
]
binaries = []
hiddenimports = []

for pkg in ("onnxruntime",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PassportSheet",
    debug=False,
    strip=False,
    upx=False,
    console=False,           # windowed app — stdout guard in main.py handles this
    icon="assets/icon.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="PassportSheet",
)
