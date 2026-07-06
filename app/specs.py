"""Loading and modelling of per-country passport photo requirements."""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class CountrySpec:
    key: str
    name: str
    photo_width_mm: float
    photo_height_mm: float
    head_min_mm: float
    head_max_mm: float
    crown_to_top_mm: float
    background: str  # hex color, e.g. "#FFFFFF"

    @property
    def head_target_mm(self) -> float:
        """Midpoint of the allowed head-height band (chin to crown)."""
        return (self.head_min_mm + self.head_max_mm) / 2.0

    @property
    def aspect(self) -> float:
        return self.photo_width_mm / self.photo_height_mm

    def background_rgb(self) -> tuple[int, int, int]:
        h = self.background.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


@dataclass(frozen=True)
class PaperSize:
    key: str
    name: str
    width_mm: float
    height_mm: float


PAPER_SIZES: list[PaperSize] = [
    PaperSize("10x15", '10x15 cm (4x6")', 100.0, 150.0),
    PaperSize("13x18", '13x18 cm (5x7")', 130.0, 180.0),
    PaperSize("a4", "A4 (21x29.7 cm)", 210.0, 297.0),
    PaperSize("letter", 'US Letter (8.5x11")', 215.9, 279.4),
]


def resource_path(relative: str) -> str:
    """Resolve a bundled resource both in dev and in a PyInstaller build."""
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


def load_specs(path: str | None = None) -> dict[str, CountrySpec]:
    if path is None:
        path = resource_path("requirements.json")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    specs: dict[str, CountrySpec] = {}
    for key, entry in raw.items():
        spec = CountrySpec(
            key=key,
            name=entry["name"],
            photo_width_mm=float(entry["photo_width_mm"]),
            photo_height_mm=float(entry["photo_height_mm"]),
            head_min_mm=float(entry["head_min_mm"]),
            head_max_mm=float(entry["head_max_mm"]),
            crown_to_top_mm=float(entry["crown_to_top_mm"]),
            background=entry["background"],
        )
        _validate(spec)
        specs[key] = spec
    return specs


def _validate(spec: CountrySpec) -> None:
    if spec.head_min_mm > spec.head_max_mm:
        raise ValueError(f"{spec.key}: head_min_mm > head_max_mm")
    if spec.head_max_mm + spec.crown_to_top_mm > spec.photo_height_mm:
        raise ValueError(
            f"{spec.key}: head band + crown_to_top exceeds photo height"
        )
    if not re.fullmatch(r"#?[0-9a-fA-F]{6}", spec.background):
        raise ValueError(
            f'{spec.key}: background must be a "#RRGGBB" hex color, '
            f"got {spec.background!r}"
        )
