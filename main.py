"""PassportSheet entry point."""
import os
import sys

# --- PyInstaller console=False guard -----------------------------------------
# In a windowed (console=False) build, sys.stdout / sys.stderr are None and any
# library that prints (onnxruntime, tqdm inside rembg, ...) crashes with a
# NoneType write error. Redirect them to devnull before anything else imports.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

def main() -> int:
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox

    if sys.platform == "win32":
        # Give the process its own taskbar identity so the window icon is
        # used there too (instead of the python.exe / launcher icon).
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "PassportSheet")

    app = QApplication(sys.argv)
    app.setApplicationName("PassportSheet")
    from app.specs import resource_path
    icon_path = resource_path(os.path.join("assets", "icon.ico"))
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    try:
        from app.main_window import MainWindow
        win = MainWindow()
    except Exception:
        import traceback
        # stderr goes to devnull in windowed builds, so a dialog is the only
        # way to learn why startup failed (usually an edited requirements.json).
        QMessageBox.critical(
            None, "PassportSheet failed to start",
            traceback.format_exc()[-2000:])
        return 1
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
