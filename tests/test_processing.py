import numpy as np
from PIL import Image

from app.processing import AutoFit, _crown_from_mask, composite_on_background


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


def test_composite_on_background():
    flat = composite_on_background(_person_cutout(), (255, 255, 255))
    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (255, 255, 255)   # transparent -> bg
    assert flat.getpixel((50, 50)) == (255, 0, 0)     # opaque -> person
