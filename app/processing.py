"""Image processing pipeline.

Steps:
1. Load photo (EXIF-aware).
2. Detect face with OpenCV YuNet, level the image using the eye line.
3. Segment the person with rembg (u2net) -> RGBA cutout.
4. Measure crown (top of hair, from the alpha mask) and chin (from the
   face box) so the UI can auto-fit the crop to a country spec.

Background compositing is cheap and done separately (per country color),
so switching countries never re-runs the expensive segmentation.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass
class AutoFit:
    """Measurements in *cutout* pixel coordinates."""
    face_cx: float      # horizontal center of the face
    crown_y: float      # top of the head, hair included
    chin_y: float       # bottom of the chin

    @property
    def head_px(self) -> float:
        return self.chin_y - self.crown_y


@dataclass
class ProcessedPhoto:
    cutout: Image.Image          # RGBA, person on transparent bg, leveled
    autofit: AutoFit | None      # None if no face was detected


# ---------------------------------------------------------------------------
# Face detection (YuNet)
# ---------------------------------------------------------------------------

_DETECT_MAX_SIDE = 1280


def _yunet_model_path() -> str:
    from .specs import resource_path
    return resource_path(os.path.join("models", "face_detection_yunet_2023mar.onnx"))


def _detect_face(pil_img: Image.Image):
    """Return (box, right_eye, left_eye) in image px, or None."""
    import cv2

    rgb = np.asarray(pil_img.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = min(1.0, _DETECT_MAX_SIDE / max(w, h))
    if scale < 1.0:
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    sh, sw = small.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        _yunet_model_path(), "", (sw, sh), score_threshold=0.6
    )
    detector.setInputSize((sw, sh))
    _, faces = detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if faces is None or len(faces) == 0:
        return None

    # Largest face wins.
    face = max(faces, key=lambda f: f[2] * f[3])
    inv = 1.0 / scale
    x, y, bw, bh = (face[0] * inv, face[1] * inv, face[2] * inv, face[3] * inv)
    right_eye = (face[4] * inv, face[5] * inv)
    left_eye = (face[6] * inv, face[7] * inv)
    return (x, y, bw, bh), right_eye, left_eye


# ---------------------------------------------------------------------------
# Segmentation (rembg / u2net)
# ---------------------------------------------------------------------------

_session = None


def _rembg_session():
    global _session
    if _session is None:
        from rembg import new_session
        _session = new_session("u2net")
    return _session


def _segment(pil_img: Image.Image) -> Image.Image:
    from rembg import remove
    cutout = remove(pil_img.convert("RGB"), session=_rembg_session())
    # Slightly feather the mask so composited edges look natural.
    r, g, b, a = cutout.split()
    a = a.filter(ImageFilter.GaussianBlur(1.2))
    return Image.merge("RGBA", (r, g, b, a))


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def _crown_from_mask(cutout: Image.Image, face_x0: float, face_x1: float) -> float | None:
    """Topmost opaque row within the face's horizontal span (captures hair)."""
    alpha = np.asarray(cutout.getchannel("A"))
    x0 = max(0, int(face_x0))
    x1 = min(cutout.width, int(math.ceil(face_x1)))
    if x1 <= x0:
        return None
    band = alpha[:, x0:x1]
    rows = np.where((band > 128).any(axis=1))[0]
    if rows.size == 0:
        return None
    return float(rows[0])


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def process_photo(path: str) -> ProcessedPhoto:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    # Level the image using the eye line, then re-detect on the leveled image.
    det = _detect_face(img)
    if det is not None:
        _, right_eye, left_eye = det
        angle = math.degrees(math.atan2(left_eye[1] - right_eye[1],
                                        left_eye[0] - right_eye[0]))
        if abs(angle) > 1.0:
            img = img.rotate(angle, resample=Image.BICUBIC,
                             expand=False, fillcolor=(255, 255, 255))
            det = _detect_face(img)

    cutout = _segment(img)

    autofit = None
    if det is not None:
        (x, y, bw, bh), _, _ = det
        crown = _crown_from_mask(cutout, x, x + bw)
        if crown is None or crown >= y + bh:
            crown = y  # fall back to the face box top
        chin = y + bh
        autofit = AutoFit(face_cx=x + bw / 2.0, crown_y=crown, chin_y=chin)

    return ProcessedPhoto(cutout=cutout, autofit=autofit)


def composite_on_background(cutout: Image.Image,
                            rgb: tuple[int, int, int]) -> Image.Image:
    bg = Image.new("RGB", cutout.size, rgb)
    bg.paste(cutout, mask=cutout.getchannel("A"))
    return bg
