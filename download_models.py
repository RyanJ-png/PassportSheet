"""Download the ONNX models used by PassportSheet into ./models.

Run once before first launch and before building with PyInstaller:
    python download_models.py
"""
import hashlib
import os
import urllib.request

# name -> (url, sha256)
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "u2net.onnx": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
    ),
}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(here, "models")
    os.makedirs(models_dir, exist_ok=True)
    for name, (url, sha) in MODELS.items():
        dest = os.path.join(models_dir, name)
        if os.path.exists(dest):
            if _sha256(dest) == sha:
                print(f"[skip] {name} already present and verified")
                continue
            print(f"[warn] {name} exists but fails checksum — re-downloading")
        print(f"[get ] {name} …")
        # Download to a temp name and rename only after the checksum passes,
        # so an interrupted or corrupted download never masquerades as a
        # valid model.
        part = dest + ".part"
        urllib.request.urlretrieve(url, part)
        actual = _sha256(part)
        if actual != sha:
            os.remove(part)
            raise SystemExit(
                f"[fail] {name}: checksum mismatch\n"
                f"       expected {sha}\n"
                f"       got      {actual}\n"
                "       The download was corrupted or the upstream file changed."
            )
        os.replace(part, dest)
        print(f"[ ok ] {name} ({os.path.getsize(dest) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
