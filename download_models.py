"""Download the ONNX models used by PassportSheet into ./models.

Run once before first launch and before building with PyInstaller:
    python download_models.py
"""
import os
import urllib.request

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "u2net.onnx": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
    ),
}


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(here, "models")
    os.makedirs(models_dir, exist_ok=True)
    for name, url in MODELS.items():
        dest = os.path.join(models_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name} …")
        urllib.request.urlretrieve(url, dest)
        print(f"[ ok ] {name} ({os.path.getsize(dest) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
