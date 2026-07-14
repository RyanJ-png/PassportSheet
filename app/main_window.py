"""Main application window."""
from __future__ import annotations

import math
import os
import traceback

from PIL import Image
from PySide6.QtCore import QPointF, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QPalette, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSlider, QStatusBar, QVBoxLayout, QWidget,
)

from . import __version__, processing
from .fine_tune import SCENE_PER_MM, PhotoEditor, pil_to_qimage, qimage_to_pil
from .sheet import best_layout, compose_sheet
from .specs import PAPER_SIZES, CountrySpec, load_specs

DPI = 300
PHOTO_EXPORT_DPI = 600     # single-photo export; online portals want pixels
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Compliance readout tolerances. Head height is checked strictly against the
# spec band; these cover the measurements specs don't give a range for.
CROWN_TOL_MM = 1.5
CENTER_TOL_MM = 2.0
MIN_EFFECTIVE_DPI = 200

# The zoom slider works in log space: photos land anywhere between thumbnail
# and poster resolution, so a linear 0.02..50 slider would park every real
# photo in the first few percent of travel.
ZOOM_MIN, ZOOM_MAX = 0.02, 50.0
ZOOM_SLIDER_MAX = 1000


def zoom_to_slider(zoom: float) -> int:
    zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
    return round(ZOOM_SLIDER_MAX
                 * math.log(zoom / ZOOM_MIN) / math.log(ZOOM_MAX / ZOOM_MIN))


def slider_to_zoom(value: int) -> float:
    return ZOOM_MIN * (ZOOM_MAX / ZOOM_MIN) ** (value / ZOOM_SLIDER_MAX)


class ResetSlider(QSlider):
    """Slider that returns to a default value on double-click."""

    def __init__(self, orientation, default: int, parent=None):
        super().__init__(orientation, parent)
        self._default = default

    def mouseDoubleClickEvent(self, event):
        self.setValue(self._default)
        event.accept()


class ProcessWorker(QThread):
    finished_ok = Signal(object)   # ProcessedPhoto
    failed = Signal(str)

    def __init__(self, source: str | Image.Image, parent=None):
        super().__init__(parent)
        self._source = source

    def run(self):
        try:
            if isinstance(self._source, str):
                result = processing.process_photo(self._source)
            else:
                result = processing.process_image(self._source)
            self.finished_ok.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class SheetPreviewDialog(QDialog):
    """Shows the composed sheet so the user can check it before saving."""

    def __init__(self, sheet: Image.Image, count: int, paper_name: str,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sheet preview")
        pix = QPixmap.fromImage(pil_to_qimage(sheet)).scaled(
            560, 640, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        preview = QLabel()
        preview.setPixmap(pix)
        info = QLabel(f"{count} photos on {paper_name} at {DPI} DPI")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(preview, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(info, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(buttons)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PassportSheet {__version__}")
        self.setMinimumSize(860, 620)
        self.resize(900, 720)

        self._specs = load_specs()
        self._processed: processing.ProcessedPhoto | None = None
        self._worker: ProcessWorker | None = None
        self._settings = QSettings("PassportSheet", "PassportSheet")

        # --- top controls -------------------------------------------------
        self.btn_load = QPushButton("Load Photo…")
        self.btn_load.setToolTip("Open an image file (Ctrl+O)")
        self.btn_load.clicked.connect(self._on_load)

        self.combo_country = QComboBox()
        self.combo_country.setToolTip("Country whose photo spec to follow")
        for key, spec in self._specs.items():
            self.combo_country.addItem(spec.name, key)
        self.combo_country.currentIndexChanged.connect(self._on_country_changed)

        self.combo_paper = QComboBox()
        self.combo_paper.setToolTip("Print paper size for the sheet export")
        for paper in PAPER_SIZES:
            self.combo_paper.addItem(paper.name, paper.key)
        self.combo_paper.currentIndexChanged.connect(self._update_capacity_label)

        self.lbl_capacity = QLabel("")
        self.lbl_capacity.setToolTip(
            "How many photos fit on one sheet of the selected paper")

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
        self.editor.setToolTip(
            "Drag to move · scroll to zoom · arrow keys to nudge 0.1 mm "
            "(Shift = 1 mm)")
        self.editor.set_spec(self._current_spec())
        self.editor.transformChanged.connect(self._sync_sliders)
        self.editor.transformChanged.connect(self._update_compliance)

        # --- bottom controls ------------------------------------------------
        self.btn_autofit = QPushButton("Auto-Fit")
        self.btn_autofit.setToolTip(
            "Re-apply the automatic head-size and position fit")
        self.btn_autofit.clicked.connect(self._on_autofit)

        self.slider_zoom = ResetSlider(Qt.Orientation.Horizontal,
                                       zoom_to_slider(1.0))
        self.slider_zoom.setRange(0, ZOOM_SLIDER_MAX)
        self.slider_zoom.setValue(zoom_to_slider(1.0))
        self.slider_zoom.setToolTip("Zoom — double-click to reset to 100%")
        self.slider_zoom.valueChanged.connect(self._on_zoom_slider)
        self.lbl_zoom_val = QLabel("100%")
        self.lbl_zoom_val.setMinimumWidth(46)

        self.slider_rot = ResetSlider(Qt.Orientation.Horizontal, 0)
        self.slider_rot.setRange(-200, 200)        # maps to -20 .. +20 deg
        self.slider_rot.setValue(0)
        self.slider_rot.setToolTip("Rotate ±20° — double-click to reset")
        self.slider_rot.valueChanged.connect(self._on_rot_slider)
        self.lbl_rot_val = QLabel("0.0°")
        self.lbl_rot_val.setMinimumWidth(40)

        self.btn_export_photo = QPushButton("Export Photo…")
        self.btn_export_photo.setToolTip(
            f"Save one photo at {PHOTO_EXPORT_DPI} DPI for online "
            "applications (Ctrl+Shift+E)")
        self.btn_export_photo.clicked.connect(self._on_export_photo)

        self.btn_export = QPushButton("Export Sheet…")
        self.btn_export.setToolTip(
            f"Preview and save a printable {DPI} DPI sheet (Ctrl+E)")
        self.btn_export.clicked.connect(self._on_export)

        bottom = QHBoxLayout()
        bottom.addWidget(self.btn_autofit)
        bottom.addWidget(QLabel("Zoom"))
        bottom.addWidget(self.slider_zoom, 1)
        bottom.addWidget(self.lbl_zoom_val)
        bottom.addWidget(QLabel("Rotate"))
        bottom.addWidget(self.slider_rot, 1)
        bottom.addWidget(self.lbl_rot_val)
        bottom.addWidget(self.btn_export_photo)
        bottom.addWidget(self.btn_export)

        # --- layout ----------------------------------------------------------
        central = QWidget()
        root = QVBoxLayout(central)
        root.addLayout(top)
        root.addWidget(self.editor, 1)
        root.addLayout(bottom)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        # Busy indicator + live spec-compliance readout on the right.
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)               # indeterminate
        self.progress.setFixedWidth(120)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.lbl_compliance = QLabel()
        self.lbl_compliance.setTextFormat(Qt.TextFormat.RichText)
        self.statusBar().addPermanentWidget(self.lbl_compliance)
        self._update_compliance_tooltip()

        self.setAcceptDrops(True)
        QShortcut(QKeySequence.StandardKey.Paste, self,
                  activated=self._on_paste)
        QShortcut(QKeySequence.StandardKey.Open, self,
                  activated=self._on_load)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self._on_export)
        QShortcut(QKeySequence("Ctrl+Shift+E"), self,
                  activated=self._on_export_photo)

        self._set_editing_enabled(False)
        self._restore_settings()
        self._update_capacity_label()
        self.statusBar().showMessage(
            "Load, drop, or paste (Ctrl+V) a photo to begin. "
            "Drag to move, scroll to zoom.")

    # ------------------------------------------------------------------ state

    def _current_spec(self) -> CountrySpec:
        return self._specs[self.combo_country.currentData()]

    def _current_paper(self):
        key = self.combo_paper.currentData()
        return next(p for p in PAPER_SIZES if p.key == key)

    def _set_editing_enabled(self, enabled: bool) -> None:
        for w in (self.btn_autofit, self.slider_zoom, self.slider_rot,
                  self.btn_export_photo, self.btn_export):
            w.setEnabled(enabled)

    def _restore_settings(self) -> None:
        geometry = self._settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        for combo, key in ((self.combo_country, "country"),
                           (self.combo_paper, "paper")):
            idx = combo.findData(self._settings.value(key))
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _update_compliance_tooltip(self) -> None:
        spec = self._current_spec()
        self.lbl_compliance.setToolTip(
            f"Live check against the {spec.name} spec:\n"
            f"• head height {spec.head_min_mm}–{spec.head_max_mm} mm\n"
            f"• crown-to-top gap {spec.crown_to_top_mm} ±{CROWN_TOL_MM} mm\n"
            f"• horizontal centering ±{CENTER_TOL_MM} mm\n"
            f"• person not cut off at the frame edges\n"
            f"• at least {MIN_EFFECTIVE_DPI} DPI effective print resolution")

    # ------------------------------------------------------------------ load

    def _on_load(self):
        if not self.btn_load.isEnabled():       # Ctrl+O while processing
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select photo", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if path:
            self._start_processing(path)

    def _start_processing(self, source: str | Image.Image):
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Still processing the previous photo…")
            return
        self.btn_load.setEnabled(False)
        self._set_editing_enabled(False)
        self.editor.set_busy(True)
        self.progress.show()
        self.statusBar().showMessage(
            "Processing photo (face detection + background removal)…")
        self._worker = ProcessWorker(source)
        self._worker.finished_ok.connect(self._on_processed)
        self._worker.failed.connect(self._on_process_failed)
        self._worker.start()

    def _on_processed(self, result: processing.ProcessedPhoto):
        self.btn_load.setEnabled(True)
        self.editor.set_busy(False)
        self.progress.hide()
        self._processed = result
        self._refresh_composite()
        self._set_editing_enabled(True)
        if result.faces_found > 1:
            self.statusBar().showMessage(
                f"Warning: {result.faces_found} faces detected — everyone in "
                "the photo will appear on the printed sheet.")
        elif result.autofit is not None:
            self.statusBar().showMessage(
                "Auto-fitted. Drag / zoom / rotate to fine-tune, then export.")
        else:
            self.statusBar().showMessage(
                "No face detected — position the photo manually.")
        if result.autofit is not None:
            self._on_autofit()
        self._update_compliance()

    # -------------------------------------------------------- drop and paste

    @staticmethod
    def _dropped_image_path(mime) -> str | None:
        urls = mime.urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            if os.path.splitext(path)[1].lower() in IMAGE_EXTS:
                return path
        return None

    def dragEnterEvent(self, event):
        if self._dropped_image_path(event.mimeData()) is not None:
            event.acceptProposedAction()
            self.editor.setStyleSheet(
                "QGraphicsView { border: 2px dashed #00d0ff; }")

    def dragLeaveEvent(self, event):
        self.editor.setStyleSheet("")

    def dropEvent(self, event):
        self.editor.setStyleSheet("")
        path = self._dropped_image_path(event.mimeData())
        if path is not None:
            event.acceptProposedAction()
            self._start_processing(path)

    def _on_paste(self):
        mime = QApplication.clipboard().mimeData()
        path = self._dropped_image_path(mime)
        if path is not None:
            self._start_processing(path)
            return
        if mime.hasImage():
            qimg = QApplication.clipboard().image()
            if not qimg.isNull():
                self._start_processing(qimage_to_pil(qimg))
                return
        self.statusBar().showMessage("Clipboard has no image.")

    def _on_process_failed(self, err: str):
        self.btn_load.setEnabled(True)
        self.editor.set_busy(False)
        self.progress.hide()
        # A previously loaded photo is still in the editor — keep it editable.
        self._set_editing_enabled(self._processed is not None)
        box = QMessageBox(QMessageBox.Icon.Critical, "Processing failed",
                          "Couldn't process this photo. Try a different "
                          "image, or check the details below.",
                          QMessageBox.StandardButton.Ok, self)
        box.setDetailedText(err)
        box.exec()
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
        self._update_compliance_tooltip()
        self._update_compliance()

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
        self.slider_zoom.setValue(zoom_to_slider(self.editor.zoom()))
        self.slider_zoom.blockSignals(False)
        self.slider_rot.blockSignals(True)
        self.slider_rot.setValue(int(round(self.editor.rotation() * 10)))
        self.slider_rot.blockSignals(False)
        self._update_readouts()

    def _update_readouts(self):
        self.lbl_zoom_val.setText(f"{self.editor.zoom() * 100:.0f}%")
        self.lbl_rot_val.setText(f"{self.editor.rotation():.1f}°")

    def _on_zoom_slider(self, value: int):
        self.editor.set_zoom(slider_to_zoom(value))
        self._update_readouts()

    def _on_rot_slider(self, value: int):
        self.editor.set_rotation(value / 10.0)
        self._update_readouts()

    # ------------------------------------------------------------ compliance

    def _readout_colors(self) -> tuple[str, str, str]:
        """(ok, bad, warn) colors with enough contrast for the active theme."""
        dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        if dark:
            return "#66bb6a", "#ef5350", "#ffb74d"
        return "#2e7d32", "#c62828", "#b26a00"

    def _update_compliance(self):
        """Live pass/fail readout of the current crop against the spec.

        Measurements (head, crown gap) always show; boolean checks appear
        only when they fail, so the readout stays short in the common
        all-good case and never crowds the status bar.
        """
        if self._processed is None or not self.editor.has_image():
            self.lbl_compliance.setText("")
            return
        spec = self._current_spec()
        col_ok, col_bad, col_warn = self._readout_colors()
        ok = f'<span style="color:{col_ok}">{{}} ✓</span>'
        bad = f'<span style="color:{col_bad}">{{}} ✗</span>'
        warn = f'<span style="color:{col_warn}">{{}}</span>'
        parts: list[str] = []

        if self._processed.faces_found > 1:
            parts.append(bad.format(f"{self._processed.faces_found} faces"))

        af = self._processed.autofit
        if af is None:
            parts.append(warn.format("no face — check size manually"))
        else:
            crown = self.editor.map_image_point(QPointF(af.face_cx, af.crown_y))
            chin = self.editor.map_image_point(QPointF(af.face_cx, af.chin_y))
            head_mm = (chin.y() - crown.y()) / SCENE_PER_MM
            gap_mm = crown.y() / SCENE_PER_MM
            center_off_mm = abs((crown.x() + chin.x()) / 2.0 / SCENE_PER_MM
                                - spec.photo_width_mm / 2.0)
            head_ok = spec.head_min_mm <= head_mm <= spec.head_max_mm
            parts.append((ok if head_ok else bad).format(
                f"head {head_mm:.1f} mm"))
            gap_ok = abs(gap_mm - spec.crown_to_top_mm) <= CROWN_TOL_MM
            parts.append((ok if gap_ok else bad).format(
                f"crown gap {gap_mm:.1f} mm"))
            if center_off_mm > CENTER_TOL_MM:
                parts.append(bad.format("off-center"))

        # The frame reaching past the image is only a defect where the person
        # touches that image edge; past background edges the fill is seamless.
        exposed = self.editor.exposed_edges()
        if exposed is not None:
            touched = self._processed.person_edges
            cut = sorted(e for e in exposed if getattr(touched, e))
            if cut:
                parts.append(bad.format("person cut at " + "/".join(cut)))

        # One source pixel should cover at least ~1 output pixel at print DPI.
        out_px_per_img_px = self.editor.zoom() * (DPI / 25.4) / SCENE_PER_MM
        if out_px_per_img_px > DPI / MIN_EFFECTIVE_DPI:
            parts.append(warn.format(
                f"low res (~{DPI / out_px_per_img_px:.0f} DPI print)"))

        self.lbl_compliance.setText(" &nbsp;·&nbsp; ".join(parts))

    # ----------------------------------------------------------------- close

    def closeEvent(self, event):
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("country", self.combo_country.currentData())
        self._settings.setValue("paper", self.combo_paper.currentData())
        # Don't destroy a running worker thread; let it finish quietly.
        if self._worker is not None and self._worker.isRunning():
            self._worker.finished_ok.disconnect()
            self._worker.failed.disconnect()
            self._worker.wait()
        super().closeEvent(event)

    # ---------------------------------------------------------------- export

    def _on_export(self):
        # Shortcut may fire while the buttons are disabled (processing).
        if not self.editor.has_image() or not self.btn_export.isEnabled():
            return
        spec = self._current_spec()
        paper = self._current_paper()
        try:
            photo = self.editor.render_photo(DPI)
            sheet, count = compose_sheet(
                photo, paper.width_mm, paper.height_mm,
                spec.photo_width_mm, spec.photo_height_mm, DPI)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        dlg = SheetPreviewDialog(sheet, count, paper.name, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        path = self._save_image(sheet, "Save printable sheet",
                                f"passport_{spec.key}_{paper.key}.jpg", DPI)
        if path:
            self.statusBar().showMessage(f"Saved {count} photos to {path}")

    def _on_export_photo(self):
        if (not self.editor.has_image()
                or not self.btn_export_photo.isEnabled()):
            return
        spec = self._current_spec()
        try:
            photo = self.editor.render_photo(PHOTO_EXPORT_DPI)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        path = self._save_image(photo, "Save single photo",
                                f"passport_{spec.key}_photo.jpg",
                                PHOTO_EXPORT_DPI)
        if path:
            self.statusBar().showMessage(
                f"Saved single {photo.width}x{photo.height} px photo to {path}")

    def _save_image(self, image: Image.Image, title: str,
                    default_name: str, dpi: int) -> str | None:
        """Ask for a filename and save; returns the path or None."""
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, title, default_name,
            "JPEG image (*.jpg);;PNG image (*.png)")
        if not path:
            return None
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            # No usable extension typed — honor the selected filter.
            ext = ".png" if "PNG" in chosen_filter else ".jpg"
            path += ext
        try:
            if ext == ".png":
                image.save(path, dpi=(dpi, dpi))
            else:
                image.save(path, quality=95, dpi=(dpi, dpi), subsampling=0)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return None
        return path
