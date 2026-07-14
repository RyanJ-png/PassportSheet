# PassportSheet

Desktop app (PySide6) that turns any photo into a printable sheet of compliant passport photos.

Flow: load a photo (file dialog, drag-and-drop, or Ctrl+V paste) → pick a country → the app auto-detects the face, levels it using the eye line, removes the background and replaces it with the country's required color, and auto-crops so the head height and crown-to-top margin match the country spec. You then fine-tune with drag / scroll-zoom / rotate against live guide overlays — a compliance readout in the status bar shows head height, crown gap, centering, and effective print resolution as you adjust. Export a 300 DPI sheet (10x15 cm, 13x18 cm, A4, or US Letter) with cut lines — a preview dialog shows the exact printout before saving — or export a single 600 DPI photo for online applications.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python download_models.py       # fetches YuNet + u2net ONNX models into ./models
python main.py
```

First run of a photo takes a few seconds (u2net segmentation). Switching countries afterwards is instant — the segmentation is cached and only the background color is re-composited.

## Editing country specs

`requirements.json` is plain data — add or tweak countries freely:

```json
"key": {
  "name": "Display Name",
  "photo_width_mm": 35, "photo_height_mm": 45,
  "head_min_mm": 32, "head_max_mm": 36,
  "crown_to_top_mm": 4,
  "background": "#FFFFFF"
}
```

`head_min/max_mm` is the allowed chin-to-crown band (shown as the green chin zone in the editor); `crown_to_top_mm` is where the cyan crown line sits. The bundled values are sensible defaults — verify against the official source for any country before client delivery, since embassies do revise these.


## Project layout

```
main.py                 entry point, frozen-build guards
app/specs.py            requirements.json loader + paper sizes
app/processing.py       YuNet face detection, u2net segmentation, measurements
app/fine_tune.py        interactive editor (drag/zoom/rotate + guide overlays)
app/main_window.py      UI wiring, worker thread, export
app/sheet.py            tiling + cut lines
requirements.json       per-country photo specs (user-editable)
assets/                 app icon
download_models.py      one-time model fetch (checksum-verified)
PassportSheet.spec      PyInstaller build config
tests/                  pytest suite (pure functions — no Qt or models needed)
```

## Development

```
pip install pytest
pytest
```

Releases are built by GitHub Actions: push a tag like `v1.2.0` and the
workflow tests, builds the Windows bundle, and attaches the zip to a GitHub
release automatically.
