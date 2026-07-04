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

# In a frozen build, point rembg at the bundled u2net model so it never tries
# to download on the client's machine.
if getattr(sys, "frozen", False):
    os.environ.setdefault("U2NET_HOME", os.path.join(sys._MEIPASS, "models"))


def main() -> int:
    from PySide6.QtWidgets import QApplication
    from app.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("PassportSheet")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
