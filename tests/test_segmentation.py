"""Structural checks on the onnxruntime-based u2net segmentation.

Self-skips when onnxruntime or the model file is unavailable (e.g. the
lightweight CI job), and runs in the release build where both exist.
"""
import os

import pytest
from PIL import Image, ImageDraw

pytest.importorskip("onnxruntime")

_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                      "models", "u2net.onnx")
pytestmark = pytest.mark.skipif(not os.path.exists(_MODEL),
                                reason="u2net.onnx not downloaded")


def test_segment_returns_rgba_cutout():
    from app.processing import _segment

    # A crude figure on a plain background; we only assert structure, not
    # segmentation quality.
    img = Image.new("RGB", (200, 260), (140, 170, 210))
    d = ImageDraw.Draw(img)
    d.ellipse([70, 40, 130, 110], fill=(224, 172, 140))
    d.rectangle([60, 110, 140, 260], fill=(60, 60, 90))

    cutout = _segment(img)
    assert cutout.mode == "RGBA"
    assert cutout.size == img.size
    alpha = cutout.getchannel("A")
    lo, hi = alpha.getextrema()
    assert 0 <= lo <= hi <= 255
