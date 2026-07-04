"""Compose a printable sheet by tiling the passport photo on paper.

Tries both paper orientations and keeps whichever fits more photos.
Thin gray cut lines are drawn through the gaps between photos.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

GAP_MM = 2.0
MARGIN_MM = 4.0
CUT_LINE_COLOR = (170, 170, 170)


@dataclass
class SheetLayout:
    paper_w_px: int
    paper_h_px: int
    cols: int
    rows: int
    cell_w_px: int
    cell_h_px: int
    gap_px: int
    margin_x_px: int
    margin_y_px: int

    @property
    def count(self) -> int:
        return self.cols * self.rows


def _mm_to_px(mm: float, dpi: int) -> int:
    return round(mm / 25.4 * dpi)


def _layout(paper_w_mm: float, paper_h_mm: float,
            photo_w_mm: float, photo_h_mm: float, dpi: int) -> SheetLayout:
    paper_w = _mm_to_px(paper_w_mm, dpi)
    paper_h = _mm_to_px(paper_h_mm, dpi)
    cell_w = _mm_to_px(photo_w_mm, dpi)
    cell_h = _mm_to_px(photo_h_mm, dpi)
    gap = _mm_to_px(GAP_MM, dpi)
    margin = _mm_to_px(MARGIN_MM, dpi)

    usable_w = paper_w - 2 * margin
    usable_h = paper_h - 2 * margin
    cols = max(0, (usable_w + gap) // (cell_w + gap))
    rows = max(0, (usable_h + gap) // (cell_h + gap))

    # Center the grid on the paper.
    grid_w = cols * cell_w + max(0, cols - 1) * gap
    grid_h = rows * cell_h + max(0, rows - 1) * gap
    mx = (paper_w - grid_w) // 2 if cols else margin
    my = (paper_h - grid_h) // 2 if rows else margin

    return SheetLayout(paper_w, paper_h, int(cols), int(rows),
                       cell_w, cell_h, gap, mx, my)


def best_layout(paper_w_mm: float, paper_h_mm: float,
                photo_w_mm: float, photo_h_mm: float,
                dpi: int = 300) -> SheetLayout:
    portrait = _layout(paper_w_mm, paper_h_mm, photo_w_mm, photo_h_mm, dpi)
    landscape = _layout(paper_h_mm, paper_w_mm, photo_w_mm, photo_h_mm, dpi)
    return landscape if landscape.count > portrait.count else portrait


def compose_sheet(photo: Image.Image,
                  paper_w_mm: float, paper_h_mm: float,
                  photo_w_mm: float, photo_h_mm: float,
                  dpi: int = 300,
                  cut_lines: bool = True) -> tuple[Image.Image, int]:
    """Return (sheet image, number of photos placed)."""
    lay = best_layout(paper_w_mm, paper_h_mm, photo_w_mm, photo_h_mm, dpi)
    if lay.count == 0:
        raise ValueError("The photo does not fit on the selected paper size.")

    sheet = Image.new("RGB", (lay.paper_w_px, lay.paper_h_px), (255, 255, 255))
    tile = photo.resize((lay.cell_w_px, lay.cell_h_px), Image.LANCZOS)

    xs = [lay.margin_x_px + c * (lay.cell_w_px + lay.gap_px) for c in range(lay.cols)]
    ys = [lay.margin_y_px + r * (lay.cell_h_px + lay.gap_px) for r in range(lay.rows)]
    for y in ys:
        for x in xs:
            sheet.paste(tile, (x, y))

    if cut_lines:
        draw = ImageDraw.Draw(sheet)
        line_w = max(2, round(dpi / 150))
        # Full-length guide lines placed just OUTSIDE every photo edge, drawn
        # after pasting so all four sides of each photo get a symmetric line.
        v_edges = sorted({x - line_w for x in xs} | {x + lay.cell_w_px for x in xs})
        h_edges = sorted({y - line_w for y in ys} | {y + lay.cell_h_px for y in ys})
        for x in v_edges:
            draw.rectangle([x, 0, x + line_w - 1, lay.paper_h_px],
                           fill=CUT_LINE_COLOR)
        for y in h_edges:
            draw.rectangle([0, y, lay.paper_w_px, y + line_w - 1],
                           fill=CUT_LINE_COLOR)

    return sheet, lay.count
