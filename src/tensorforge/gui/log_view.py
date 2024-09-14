import logging
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtCore import Qt, pyqtSlot


class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)


class LogView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.log_widget = QTextEditLogger(self)
        layout.addWidget(self.log_widget.widget)

        # Create a logger
        self.logger = logging.getLogger('TensorForge')
        self.logger.setLevel(logging.DEBUG)

        # Add the QTextEdit as a handler
        self.log_widget.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.log_widget)

    @pyqtSlot(str)
    def log(self, message):
        self.logger.info(message)