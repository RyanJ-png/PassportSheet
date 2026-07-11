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


@dataclass(frozen=True)
class EdgeContact:
    """Which image edges the person's opaque mask touches. If the output
    frame extends past a touched edge, the person is visibly cut off; past
    an untouched edge only seamless background is exposed."""
    top: bool = False
    right: bool = False
    bottom: bool = False
    left: bool = False


@dataclass
class ProcessedPhoto:
    cutout: Image.Image          # RGBA, person on transparent bg, leveled
    autofit: AutoFit | None      # None if no face was detected
    faces_found: int = 0         # everyone detected — passports want exactly 1
    person_edges: EdgeContact = EdgeContact()


# ---------------------------------------------------------------------------
# Face detection (YuNet)
# ---------------------------------------------------------------------------

_DETECT_MAX_SIDE = 1280


def _yunet_model_path() -> str:
    from .specs import resource_path
    return resource_path(os.path.join("models", "face_detection_yunet_2023mar.onnx"))


def _detect_face(pil_img: Image.Image):
    """Return (box, right_eye, left_eye, n_faces) in image px, or None."""
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
    return (x, y, bw, bh), right_eye, left_eye, len(faces)


# ---------------------------------------------------------------------------
# Segmentation (u2net via onnxruntime)
# ---------------------------------------------------------------------------

_session = None


def _u2net_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        from .specs import resource_path
        _session = ort.InferenceSession(
            resource_path(os.path.join("models", "u2net.onnx")),
            providers=["CPUExecutionProvider"])
    return _session


def _segment(pil_img: Image.Image) -> Image.Image:
    """Person cutout with u2net, run directly through onnxruntime.

    Pre/post-processing mirrors rembg's u2net session (verified pixel-
    identical), without rembg's heavy dependency tree (numba, scipy,
    scikit-image, ...) that roughly doubled the frozen bundle.
    """
    img = pil_img.convert("RGB")
    session = _u2net_session()

    small = img.resize((320, 320), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.float32)
    arr = arr / max(float(arr.max()), 1e-6)
    arr = (arr - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    arr = arr.transpose((2, 0, 1))[np.newaxis].astype(np.float32)

    pred = session.run(None, {session.get_inputs()[0].name: arr})[0][0, 0]
    mi, ma = float(pred.min()), float(pred.max())
    pred = (pred - mi) / max(ma - mi, 1e-6)
    mask = Image.fromarray((pred.clip(0, 1) * 255).astype(np.uint8), "L")
    mask = mask.resize(img.size, Image.Resampling.LANCZOS)

    # Slightly feather the mask so composited edges look natural.
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    cutout = img.convert("RGBA")
    cutout.putalpha(mask)
    return cutout


# ---------------------------------------------------------------------------
# Canvas extension
# ---------------------------------------------------------------------------

# Tall/wide photo frames (Canada 50x70, USA 51x51) often extend past the
# edges of a tightly cropped source photo once the head is placed per spec.
# Pad the cutout by replicating its edge pixels so clothing continues to the
# frame edge instead of ending in a hard line with background below
# ("floating portrait"). Transparent edges replicate as transparent, so the
# padding is invisible unless the frame actually reaches it. The top is never
# padded — heads must not be smeared upward.
_PAD_BOTTOM_FRAC = 0.75
_PAD_SIDE_FRAC = 0.35


def _extend_canvas(cutout: Image.Image) -> tuple[Image.Image, int]:
    """Return (padded cutout, left padding px)."""
    pad_b = int(cutout.height * _PAD_BOTTOM_FRAC)
    pad_s = int(cutout.width * _PAD_SIDE_FRAC)
    arr = np.pad(np.asarray(cutout), ((0, pad_b), (pad_s, pad_s), (0, 0)),
                 mode="edge")
    padded = Image.fromarray(arr, "RGBA")
    # Blur the replicated regions so they read as out-of-focus continuation
    # rather than streaks, then restore the original pixels on top.
    soft = padded.filter(ImageFilter.GaussianBlur(12))
    soft.paste(cutout, (pad_s, 0))
    return soft, pad_s


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
    return process_image(img)


def process_image(img: Image.Image) -> ProcessedPhoto:
    img = img.convert("RGB")

    # Level the image using the eye line, then re-detect on the leveled image.
    det = _detect_face(img)
    if det is not None:
        _, right_eye, left_eye, _ = det
        angle = math.degrees(math.atan2(left_eye[1] - right_eye[1],
                                        left_eye[0] - right_eye[0]))
        if abs(angle) > 1.0:
            leveled = img.rotate(angle, resample=Image.BICUBIC,
                                 expand=False, fillcolor=(255, 255, 255))
            det_leveled = _detect_face(leveled)
            # Keep the unleveled image if rotation makes the face undetectable,
            # so autofit still works with the original detection.
            if det_leveled is not None:
                img, det = leveled, det_leveled

    cutout = _segment(img)
    cutout, pad_left = _extend_canvas(cutout)

    alpha = np.asarray(cutout.getchannel("A"))
    person_edges = EdgeContact(
        top=bool((alpha[0] > 128).any()),
        right=bool((alpha[:, -1] > 128).any()),
        bottom=bool((alpha[-1] > 128).any()),
        left=bool((alpha[:, 0] > 128).any()),
    )

    autofit = None
    faces_found = 0
    if det is not None:
        (x, y, bw, bh), _, _, faces_found = det
        x += pad_left  # detection ran on the unpadded image
        crown = _crown_from_mask(cutout, x, x + bw)
        if crown is None or crown >= y + bh:
            crown = y  # fall back to the face box top
        chin = y + bh
        autofit = AutoFit(face_cx=x + bw / 2.0, crown_y=crown, chin_y=chin)

    return ProcessedPhoto(cutout=cutout, autofit=autofit,
                          faces_found=faces_found,
                          person_edges=person_edges)


def composite_on_background(cutout: Image.Image,
                            rgb: tuple[int, int, int]) -> Image.Image:
    bg = Image.new("RGB", cutout.size, rgb)
    bg.paste(cutout, mask=cutout.getchannel("A"))
    return bg
