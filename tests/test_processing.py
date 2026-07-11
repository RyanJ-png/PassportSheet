import numpy as np
from PIL import Image

from app.processing import (
    AutoFit, _crown_from_mask, _extend_canvas, composite_on_background,
)


def _person_cutout() -> Image.Image:
    """RGBA test image: opaque red block from y=30 down, columns 20..79."""
    a = np.zeros((100, 100, 4), dtype=np.uint8)
    a[30:, 20:80] = (255, 0, 0, 255)
    return Image.fromarray(a, "RGBA")


def test_crown_found_at_top_of_mask():
    assert _crown_from_mask(_person_cutout(), 40, 60) == 30.0


def test_crown_none_outside_mask_columns():
    assert _crown_from_mask(_person_cutout(), 0, 10) is None


def test_crown_none_for_degenerate_band():
    assert _crown_from_mask(_person_cutout(), 50, 50) is None


def test_crown_band_clamped_to_image():
    assert _crown_from_mask(_person_cutout(), -5, 500) == 30.0


def test_autofit_head_px():
    assert AutoFit(face_cx=0, crown_y=10, chin_y=50).head_px == 40


def test_extend_canvas_replicates_person_downward():
    padded, pad_left = _extend_canvas(_person_cutout())
    assert pad_left > 0
    assert padded.width == 100 + 2 * pad_left
    assert padded.height > 100
    # The person touched the bottom edge, so the padding below continues it
    # (softened by a blur, hence approximate values).
    r, g, b, a = padded.getpixel((pad_left + 50, padded.height - 1))
    assert r > 240 and g < 15 and b < 15 and a > 240
    # Transparent edges replicate as transparent — padding stays invisible.
    assert padded.getpixel((0, 0))[3] == 0
    assert padded.getpixel((0, padded.height - 1))[3] == 0


def test_extend_canvas_never_pads_top():
    padded, pad_left = _extend_canvas(_person_cutout())
    # Crown row must be unchanged: still transparent above y=30, opaque at 30.
    assert padded.getpixel((pad_left + 50, 29))[3] == 0
    assert padded.getpixel((pad_left + 50, 30))[3] == 255


def test_composite_on_background():
    flat = composite_on_background(_person_cutout(), (255, 255, 255))
    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (255, 255, 255)   # transparent -> bg
    assert flat.getpixel((50, 50)) == (255, 0, 0)     # opaque -> person
