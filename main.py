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
    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv)
    app.setApplicationName("PassportSheet")
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
