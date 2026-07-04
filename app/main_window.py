"""Main application window."""
from __future__ import annotations

import os
import traceback

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSlider, QStatusBar, QVBoxLayout, QWidget,
)

from . import processing
from .fine_tune import PhotoEditor
from .sheet import best_layout, compose_sheet
from .specs import PAPER_SIZES, CountrySpec, load_specs

DPI = 300


class ProcessWorker(QThread):
    finished_ok = Signal(object)   # ProcessedPhoto
    failed = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        try:
            result = processing.process_photo(self._path)
            self.finished_ok.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PassportSheet")
        self.resize(900, 720)

        self._specs = load_specs()
        self._processed: processing.ProcessedPhoto | None = None
        self._worker: ProcessWorker | None = None

        # --- top controls -------------------------------------------------
        self.btn_load = QPushButton("Load Photo…")
        self.btn_load.clicked.connect(self._on_load)

        self.combo_country = QComboBox()
        for key, spec in self._specs.items():
            self.combo_country.addItem(spec.name, key)
        self.combo_country.currentIndexChanged.connect(self._on_country_changed)

        self.combo_paper = QComboBox()
        for paper in PAPER_SIZES:
            self.combo_paper.addItem(paper.name, paper.key)
        self.combo_paper.currentIndexChanged.connect(self._update_capacity_label)

        self.lbl_capacity = QLabel("")

        top = QHBoxLayout()
        top.addWidget(self.btn_load)
        top.addWidget(QLabel("Country:"))
        top.addWidget(self.combo_country)
        top.addWidget(QLabel("Paper:"))
        top.addWidget(self.combo_paper)
        top.addWidget(self.lbl_capacity)
        top.addStretch(1)

        # --- editor --------------------------------------------------------
        self.editor = PhotoEditor()
        self.editor.set_spec(self._current_spec())

        # --- bottom controls ------------------------------------------------
        self.btn_autofit = QPushButton("Auto-Fit")
        self.btn_autofit.clicked.connect(self._on_autofit)

        self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(2, 400)          # maps to 0.02 .. 4.0
        self.slider_zoom.setValue(100)
        self.slider_zoom.valueChanged.connect(self._on_zoom_slider)

        self.slider_rot = QSlider(Qt.Orientation.Horizontal)
        self.slider_rot.setRange(-200, 200)        # maps to -20 .. +20 deg
        self.slider_rot.setValue(0)
        self.slider_rot.valueChanged.connect(self._on_rot_slider)

        self.btn_export = QPushButton("Export Sheet…")
        self.btn_export.clicked.connect(self._on_export)

        bottom = QHBoxLayout()
        bottom.addWidget(self.btn_autofit)
        bottom.addWidget(QLabel("Zoom"))
        bottom.addWidget(self.slider_zoom, 1)
        bottom.addWidget(QLabel("Rotate"))
        bottom.addWidget(self.slider_rot, 1)
        bottom.addWidget(self.btn_export)

        # --- layout ----------------------------------------------------------
        central = QWidget()
        root = QVBoxLayout(central)
        root.addLayout(top)
        root.addWidget(self.editor, 1)
        root.addLayout(bottom)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        self._set_editing_enabled(False)
        self._update_capacity_label()
        self.statusBar().showMessage(
            "Load a photo to begin. Drag to move, scroll to zoom.")

    # ------------------------------------------------------------------ state

    def _current_spec(self) -> CountrySpec:
        return self._specs[self.combo_country.currentData()]

    def _current_paper(self):
        key = self.combo_paper.currentData()
        return next(p for p in PAPER_SIZES if p.key == key)

    def _set_editing_enabled(self, enabled: bool) -> None:
        for w in (self.btn_autofit, self.slider_zoom, self.slider_rot,
                  self.btn_export):
            w.setEnabled(enabled)

    # ------------------------------------------------------------------ load

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select photo", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if not path:
            return
        self.btn_load.setEnabled(False)
        self._set_editing_enabled(False)
        self.statusBar().showMessage(
            "Processing photo (face detection + background removal)…")
        self._worker = ProcessWorker(path)
        self._worker.finished_ok.connect(self._on_processed)
        self._worker.failed.connect(self._on_process_failed)
        self._worker.start()

    def _on_processed(self, result: processing.ProcessedPhoto):
        self.btn_load.setEnabled(True)
        self._processed = result
        self._refresh_composite()
        self._set_editing_enabled(True)
        if result.autofit is not None:
            self._on_autofit()
            self.statusBar().showMessage(
                "Auto-fitted. Drag / zoom / rotate to fine-tune, then export.")
        else:
            self.statusBar().showMessage(
                "No face detected — position the photo manually.")

    def _on_process_failed(self, err: str):
        self.btn_load.setEnabled(True)
        QMessageBox.critical(self, "Processing failed", err[-2000:])
        self.statusBar().showMessage("Processing failed.")

    # ------------------------------------------------------------- composite

    def _refresh_composite(self):
        if self._processed is None:
            return
        spec = self._current_spec()
        flat = processing.composite_on_background(
            self._processed.cutout, spec.background_rgb())
        self.editor.set_image(flat)

    def _on_country_changed(self):
        spec = self._current_spec()
        self.editor.set_spec(spec)
        if self._processed is not None:
            self._refresh_composite()
            if self._processed.autofit is not None:
                self._on_autofit()
        self._update_capacity_label()

    def _update_capacity_label(self):
        spec = self._current_spec()
        paper = self._current_paper()
        lay = best_layout(paper.width_mm, paper.height_mm,
                          spec.photo_width_mm, spec.photo_height_mm, DPI)
        self.lbl_capacity.setText(f"{lay.count} photos per sheet")

    # ------------------------------------------------------------- fine-tune

    def _on_autofit(self):
        if self._processed is None or self._processed.autofit is None:
            return
        self.editor.auto_fit(self._processed.autofit)
        self._sync_sliders()

    def _sync_sliders(self):
        self.slider_zoom.blockSignals(True)
        self.slider_zoom.setValue(int(round(self.editor.zoom() * 100)))
        self.slider_zoom.blockSignals(False)
        self.slider_rot.blockSignals(True)
        self.slider_rot.setValue(int(round(self.editor.rotation() * 10)))
        self.slider_rot.blockSignals(False)

    def _on_zoom_slider(self, value: int):
        self.editor.set_zoom(value / 100.0)

    def _on_rot_slider(self, value: int):
        self.editor.set_rotation(value / 10.0)

    # ---------------------------------------------------------------- export

    def _on_export(self):
        if not self.editor.has_image():
            return
        spec = self._current_spec()
        paper = self._current_paper()
        default_name = f"passport_{spec.key}_{paper.key}.jpg"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save printable sheet", default_name,
            "JPEG image (*.jpg);;PNG image (*.png)")
        if not path:
            return
        try:
            photo = self.editor.render_photo(DPI)
            sheet, count = compose_sheet(
                photo, paper.width_mm, paper.height_mm,
                spec.photo_width_mm, spec.photo_height_mm, DPI)
            ext = os.path.splitext(path)[1].lower()
            if ext == ".png":
                sheet.save(path, dpi=(DPI, DPI))
            else:
                if ext not in (".jpg", ".jpeg"):
                    path += ".jpg"
                sheet.save(path, quality=95, dpi=(DPI, DPI),
                           subsampling=0)
            self.statusBar().showMessage(
                f"Saved {count} photos to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
