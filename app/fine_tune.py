"""Interactive crop editor.

The scene is the output photo at a fixed scale (SCENE_PER_MM units per mm).
The photo is a movable/scalable/rotatable pixmap item; overlay guides show
the crown-to-top line and the allowed chin band so the user can see spec
compliance while adjusting. Export renders the scene rect at print DPI.
"""
from __future__ import annotations

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from .processing import AutoFit
from .specs import CountrySpec

SCENE_PER_MM = 10.0


def pil_to_qimage(img: Image.Image) -> QImage:
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.width, img.height, img.width * 4,
                  QImage.Format.Format_RGBA8888)
    return qimg.copy()  # detach from the python buffer


def qimage_to_pil(qimg: QImage) -> Image.Image:
    qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = qimg.constBits()
    data = bytes(ptr)[: qimg.width() * qimg.height() * 4]
    return Image.frombytes("RGBA", (qimg.width(), qimg.height()), data).convert("RGB")


class PhotoEditor(QGraphicsView):
    # Emitted whenever the photo's placement changes (zoom, rotation, drag,
    # auto-fit, new image), so sliders and the compliance readout stay in sync.
    transformChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing
                            | QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor("#2b2b2b"))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self._item: QGraphicsPixmapItem | None = None
        self._spec: CountrySpec | None = None
        self._scale = 1.0
        self._rotation = 0.0
        self._dragging = False
        self._last_pos = QPointF()

    # ------------------------------------------------------------------ setup

    def set_spec(self, spec: CountrySpec) -> None:
        self._spec = spec
        w = spec.photo_width_mm * SCENE_PER_MM
        h = spec.photo_height_mm * SCENE_PER_MM
        self._scene.setSceneRect(0, 0, w, h)
        self._fit_view()
        self.viewport().update()

    def set_image(self, img: Image.Image) -> None:
        pix = QPixmap.fromImage(pil_to_qimage(img))
        if self._item is None:
            self._item = self._scene.addPixmap(pix)
            self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self._item.setZValue(-1)
        else:
            self._item.setPixmap(pix)
        self._item.setTransformOriginPoint(pix.width() / 2, pix.height() / 2)
        self._apply_transform()
        self.transformChanged.emit()

    def has_image(self) -> bool:
        return self._item is not None

    # ------------------------------------------------------------ transforms

    def _apply_transform(self) -> None:
        if self._item is None:
            return
        self._item.setScale(self._scale)
        self._item.setRotation(self._rotation)

    def _map_image_to_scene(self, p: QPointF) -> QPointF:
        return self._item.mapToScene(p)

    def _pos_to_map(self, img_pt: QPointF, scene_pt: QPointF) -> None:
        """Move the item so that img_pt lands exactly on scene_pt."""
        current = self._item.mapToScene(img_pt)
        delta = scene_pt - current
        self._item.setPos(self._item.pos() + delta)

    def map_image_point(self, img_pt: QPointF) -> QPointF | None:
        """Map a point in image pixel coordinates to scene coordinates."""
        if self._item is None:
            return None
        return self._item.mapToScene(img_pt)

    def auto_fit(self, fit: AutoFit) -> None:
        if self._item is None or self._spec is None or fit.head_px <= 0:
            return
        spec = self._spec
        self._rotation = 0.0
        self._scale = (spec.head_target_mm * SCENE_PER_MM) / fit.head_px
        self._apply_transform()
        crown_img = QPointF(fit.face_cx, fit.crown_y)
        crown_scene = QPointF(spec.photo_width_mm * SCENE_PER_MM / 2.0,
                              spec.crown_to_top_mm * SCENE_PER_MM)
        self._pos_to_map(crown_img, crown_scene)
        self.viewport().update()
        self.transformChanged.emit()

    def set_zoom(self, factor: float, anchor_scene: QPointF | None = None) -> None:
        if self._item is None:
            return
        factor = max(0.02, min(50.0, factor))
        if anchor_scene is None:
            anchor_scene = self._scene.sceneRect().center()
        img_pt = self._item.mapFromScene(anchor_scene)
        self._scale = factor
        self._apply_transform()
        self._pos_to_map(img_pt, anchor_scene)
        self.viewport().update()
        self.transformChanged.emit()

    def zoom(self) -> float:
        return self._scale

    def set_rotation(self, degrees: float) -> None:
        if self._item is None:
            return
        center = self._scene.sceneRect().center()
        img_pt = self._item.mapFromScene(center)
        self._rotation = degrees
        self._apply_transform()
        self._pos_to_map(img_pt, center)
        self.viewport().update()
        self.transformChanged.emit()

    def rotation(self) -> float:
        return self._rotation

    # ----------------------------------------------------------------- input

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._item is not None:
            self._dragging = True
            self._last_pos = self.mapToScene(event.position().toPoint())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._item is not None:
            pos = self.mapToScene(event.position().toPoint())
            self._item.setPos(self._item.pos() + (pos - self._last_pos))
            self._last_pos = pos
            self.viewport().update()
            self.transformChanged.emit()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._dragging = False
        self.unsetCursor()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if self._item is None:
            return
        steps = event.angleDelta().y() / 120.0
        anchor = self.mapToScene(event.position().toPoint())
        self.set_zoom(self._scale * (1.1 ** steps), anchor)  # emits transformChanged

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_view()

    def _fit_view(self) -> None:
        if not self._scene.sceneRect().isEmpty():
            self.fitInView(self._scene.sceneRect().adjusted(-40, -40, 40, 40),
                           Qt.AspectRatioMode.KeepAspectRatio)

    # ---------------------------------------------------------------- guides

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        if self._spec is None:
            return
        spec = self._spec
        sr = self._scene.sceneRect()

        # Dim everything outside the photo frame.
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 140))
        outside = QRectF(rect)
        for piece in (
            QRectF(outside.left(), outside.top(), outside.width(), sr.top() - outside.top()),
            QRectF(outside.left(), sr.bottom(), outside.width(), outside.bottom() - sr.bottom()),
            QRectF(outside.left(), sr.top(), sr.left() - outside.left(), sr.height()),
            QRectF(sr.right(), sr.top(), outside.right() - sr.right(), sr.height()),
        ):
            if piece.isValid():
                painter.drawRect(piece)
        painter.restore()

        # Photo border.
        painter.setPen(QPen(QColor("#ffffff"), 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(sr)

        crown_y = spec.crown_to_top_mm * SCENE_PER_MM
        chin_min_y = crown_y + spec.head_min_mm * SCENE_PER_MM
        chin_max_y = crown_y + spec.head_max_mm * SCENE_PER_MM

        # Crown line (cyan) and chin band (green).
        painter.setPen(QPen(QColor("#00d0ff"), 2, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(sr.left(), crown_y), QPointF(sr.right(), crown_y))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 220, 90, 60))
        painter.drawRect(QRectF(sr.left(), chin_min_y, sr.width(),
                                chin_max_y - chin_min_y))
        painter.setPen(QPen(QColor("#00dc5a"), 2, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(sr.left(), chin_min_y), QPointF(sr.right(), chin_min_y))
        painter.drawLine(QPointF(sr.left(), chin_max_y), QPointF(sr.right(), chin_max_y))

        # Vertical center line.
        painter.setPen(QPen(QColor(255, 255, 255, 90), 1, Qt.PenStyle.DotLine))
        cx = sr.center().x()
        painter.drawLine(QPointF(cx, sr.top()), QPointF(cx, sr.bottom()))

        # Labels.
        painter.setPen(QColor("#00d0ff"))
        painter.drawText(QPointF(sr.left() + 6, crown_y - 5), "crown")
        painter.setPen(QColor("#00dc5a"))
        painter.drawText(QPointF(sr.left() + 6, chin_max_y + 16), "chin zone")

    # ---------------------------------------------------------------- export

    def render_photo(self, dpi: int = 300) -> Image.Image | None:
        """Render the crop frame to a print-resolution PIL image."""
        if self._item is None or self._spec is None:
            return None
        spec = self._spec
        out_w = round(spec.photo_width_mm / 25.4 * dpi)
        out_h = round(spec.photo_height_mm / 25.4 * dpi)

        qimg = QImage(out_w, out_h, QImage.Format.Format_RGBA8888)
        qimg.fill(QColor(*spec.background_rgb()))
        painter = QPainter(qimg)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing
                               | QPainter.RenderHint.SmoothPixmapTransform)
        sr = self._scene.sceneRect()
        painter.setTransform(QTransform.fromScale(out_w / sr.width(),
                                                  out_h / sr.height()))
        # Render only the pixmap item (no foreground guides).
        self._render_item_only(painter, sr)
        painter.end()
        return qimage_to_pil(qimg)

    def _render_item_only(self, painter: QPainter, source: QRectF) -> None:
        painter.save()
        painter.translate(-source.left(), -source.top())
        painter.setTransform(self._item.sceneTransform(), combine=True)
        pix = self._item.pixmap()
        painter.drawPixmap(0, 0, pix)
        painter.restore()
