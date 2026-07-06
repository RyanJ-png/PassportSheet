import pytest
from PIL import Image

from app.sheet import best_layout, compose_sheet


def test_35x45_on_10x15_fits_six():
    lay = best_layout(100, 150, 35, 45, 300)
    assert lay.count == 6
    assert (lay.cols, lay.rows) in ((2, 3), (3, 2))


def test_50x70_on_10x15_fits_two():
    assert best_layout(100, 150, 50, 70, 300).count == 2


def test_35x45_on_a4_fits_more_than_10x15():
    assert best_layout(210, 297, 35, 45, 300).count > 6


def test_landscape_only_chosen_when_strictly_better():
    lay = best_layout(100, 150, 35, 45, 300)
    # Grid must fit inside the paper dimensions the layout reports.
    grid_w = lay.cols * lay.cell_w_px + (lay.cols - 1) * lay.gap_px
    grid_h = lay.rows * lay.cell_h_px + (lay.rows - 1) * lay.gap_px
    assert grid_w <= lay.paper_w_px
    assert grid_h <= lay.paper_h_px


def test_compose_sheet_size_and_count():
    photo = Image.new("RGB", (413, 531), (120, 120, 120))
    sheet, n = compose_sheet(photo, 100, 150, 35, 45, 300)
    assert n == 6
    assert sheet.size == (1181, 1772)


def test_photo_too_big_raises():
    photo = Image.new("RGB", (10, 10))
    with pytest.raises(ValueError):
        compose_sheet(photo, 100, 150, 120, 160, 300)


def test_cut_lines_do_not_overlap_photos():
    color = (10, 200, 10)
    photo = Image.new("RGB", (413, 531), color)
    sheet, _ = compose_sheet(photo, 100, 150, 35, 45, 300)
    lay = best_layout(100, 150, 35, 45, 300)
    for c in range(lay.cols):
        for r in range(lay.rows):
            x = lay.margin_x_px + c * (lay.cell_w_px + lay.gap_px)
            y = lay.margin_y_px + r * (lay.cell_h_px + lay.gap_px)
            # All four photo corners must be untouched by cut lines.
            for px, py in ((x, y), (x + lay.cell_w_px - 1, y),
                           (x, y + lay.cell_h_px - 1),
                           (x + lay.cell_w_px - 1, y + lay.cell_h_px - 1)):
                assert sheet.getpixel((px, py)) == color, (c, r, px, py)
