import json

import pytest

from app.specs import PAPER_SIZES, load_specs


def _spec_dict(**overrides):
    base = {
        "name": "Test", "photo_width_mm": 35, "photo_height_mm": 45,
        "head_min_mm": 32, "head_max_mm": 36, "crown_to_top_mm": 4,
        "background": "#FFFFFF",
    }
    base.update(overrides)
    return {"test": base}


def _write(tmp_path, data):
    p = tmp_path / "specs.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_bundled_specs_load():
    specs = load_specs()
    assert {"usa", "uk", "france", "germany", "japan"} <= specs.keys()
    for spec in specs.values():
        rgb = spec.background_rgb()
        assert len(rgb) == 3
        assert all(0 <= c <= 255 for c in rgb)
        assert spec.head_min_mm <= spec.head_target_mm <= spec.head_max_mm


def test_head_band_inverted_rejected(tmp_path):
    path = _write(tmp_path, _spec_dict(head_min_mm=37))
    with pytest.raises(ValueError, match="head_min_mm > head_max_mm"):
        load_specs(path)


def test_head_taller_than_photo_rejected(tmp_path):
    path = _write(tmp_path, _spec_dict(head_max_mm=44, crown_to_top_mm=9))
    with pytest.raises(ValueError, match="exceeds photo height"):
        load_specs(path)


@pytest.mark.parametrize("bg", ["white", "#FFF", "#GGGGGG", "", "##FFFFFF"])
def test_bad_background_rejected(tmp_path, bg):
    path = _write(tmp_path, _spec_dict(background=bg))
    with pytest.raises(ValueError, match="background"):
        load_specs(path)


def test_background_without_hash_accepted(tmp_path):
    path = _write(tmp_path, _spec_dict(background="EBEBEB"))
    specs = load_specs(path)
    assert specs["test"].background_rgb() == (0xEB, 0xEB, 0xEB)


def test_paper_sizes_unique_keys():
    keys = [p.key for p in PAPER_SIZES]
    assert len(keys) == len(set(keys))
    assert {"10x15", "a4", "letter"} <= set(keys)
