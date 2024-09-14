import sys
from PyQt6.QtWidgets import QApplication
from tensorforge.gui.main_window import MainWindow
import logging

def exception_hook(exctype, value, traceback):
    logging.error("Uncaught exception", exc_info=(exctype, value, traceback))
    sys.__excepthook__(exctype, value, traceback)

def main():
    app = QApplication(sys.argv)

    # Set up global exception handling
    sys.excepthook = exception_hook

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()