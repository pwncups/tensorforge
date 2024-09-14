from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFileDialog, QTreeWidget,
                             QTreeWidgetItem, QMessageBox, QLabel, QProgressBar, QMenu)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData, QPoint
from PyQt6.QtGui import QDrag
from tensorforge.models.safetensor_model import SafetensorModel
import logging
import json
import traceback

logger = logging.getLogger(__name__)


class ModelLoadThread(QThread):
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(SafetensorModel)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            model = SafetensorModel(self.file_path)
            self.progress_update.emit(33)

            metadata = model.load_metadata()
            self.progress_update.emit(66)

            # Instead of loading all layers at once, we'll load them in chunks
            model.load_layer_info_chunked(chunk_size=100, callback=self.update_progress)

            self.finished.emit(model)
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    def update_progress(self, progress):
        self.progress_update.emit(progress)


class DraggableTreeWidget(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        if item:
            menu = QMenu(self)
            add_action = menu.addAction("Add to Model Builder")
            action = menu.exec(self.mapToGlobal(event.pos()))
            if action == add_action:
                self.parent().add_to_model_builder(item)

    def get_item_data(self, item):
        data = {"text": item.text(0), "children": []}
        if item.childCount() == 0:
            data["layer_info"] = item.data(0, Qt.ItemDataRole.UserRole)
        else:
            for i in range(item.childCount()):
                data["children"].append(self.get_item_data(item.child(i)))
        return data


class ComparisonView(QWidget):
    add_to_builder_signal = pyqtSignal(QTreeWidgetItem, str)
    model_loaded_signal = pyqtSignal(dict, int)

    def __init__(self):
        super().__init__()
        self.model1 = None
        self.model2 = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.model1_label = QLabel("Model 1: Not loaded")
        self.model2_label = QLabel("Model 2: Not loaded")
        layout.addWidget(self.model1_label)
        layout.addWidget(self.model2_label)

        self.model1_tree = DraggableTreeWidget(self)
        self.model2_tree = DraggableTreeWidget(self)

        layout.addWidget(self.model1_tree)
        layout.addWidget(self.model2_tree)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        load_model1_btn = QPushButton("Load Model 1")
        load_model2_btn = QPushButton("Load Model 2")
        load_model1_btn.clicked.connect(lambda: self.load_model(1))
        load_model2_btn.clicked.connect(lambda: self.load_model(2))
        layout.addWidget(load_model1_btn)
        layout.addWidget(load_model2_btn)

        self.setLayout(layout)

    def load_model(self, model_num):
        file_path, _ = QFileDialog.getOpenFileName(self, f"Open Model {model_num}", "",
                                                   "Safetensor Files (*.safetensors)")
        if file_path:
            self.progress_bar.setValue(0)
            self.load_thread = ModelLoadThread(file_path)
            self.load_thread.progress_update.connect(self.update_progress)
            self.load_thread.finished.connect(lambda model: self.on_model_loaded(model, model_num))
            self.load_thread.error.connect(self.on_load_error)
            self.load_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_model_loaded(self, model, model_num):
        try:
            if model_num == 1:
                self.model1 = model
                self.model1_label.setText(f"Model 1: {self.model1.file_path.split('/')[-1]}")
                tree = self.model1_tree
            else:
                self.model2 = model
                self.model2_label.setText(f"Model 2: {self.model2.file_path.split('/')[-1]}")
                tree = self.model2_tree

            tree.clear()

            metadata_item = QTreeWidgetItem(tree, ["Metadata"])
            if model.metadata:
                for key, value in model.metadata.items():
                    QTreeWidgetItem(metadata_item, [f"{key}: {value}"])
            else:
                QTreeWidgetItem(metadata_item, ["No metadata available"])

            layers_item = QTreeWidgetItem(tree, [f"Layers (Total: {len(model.layer_info)})"])
            self.add_layer_items(layers_item, model.get_model_structure())

            tree.expandToDepth(0)
            logger.info(f"Model {model_num} loaded successfully")
            self.model_loaded_signal.emit(model.layer_info, model_num)
        except Exception as e:
            error_msg = f"Error processing loaded model: {str(e)}\n{traceback.format_exc()}"
            self.on_load_error(error_msg)

    def on_load_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"Failed to load model: {error_msg}")
        logger.error(f"Model loading error: {error_msg}")

    def add_layer_items(self, parent, structure):
        for key, value in structure.items():
            if isinstance(value, dict) and 'shape' not in value:
                item = QTreeWidgetItem(parent, [key])
                self.add_layer_items(item, value)
            else:
                shape_str = 'x'.join(map(str, value['shape']))
                size_mb = value['size_bytes'] / (1024 * 1024)
                item = QTreeWidgetItem(parent, [
                    f"{key}: {shape_str}, {value['dtype']}, {size_mb:.2f} MB"
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, value)

    def add_to_model_builder(self, item):
        try:
            logger.info(f"Adding item to model builder: {item.text(0)}")
            source_model = self.model1 if item.treeWidget() == self.model1_tree else self.model2
            if hasattr(source_model, 'file_path'):
                self.add_to_builder_signal.emit(item, source_model.file_path)
            else:
                logger.warning("Source model does not have file_path attribute")
                QMessageBox.warning(self, "Warning", "Source model path not available")
        except Exception as e:
            logger.error(f"Error adding item to model builder: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add item to model builder: {str(e)}")

    def get_item_data(self, item):
        return self.model1_tree.get_item_data(
            item) if item.treeWidget() == self.model1_tree else self.model2_tree.get_item_data(item)

    def _collect_child_layers(self, item):
        layers = {}
        for i in range(item.childCount()):
            child = item.child(i)
            layer_info = child.data(0, Qt.ItemDataRole.UserRole)
            if layer_info:
                layers[child.text(0)] = layer_info
            layers.update(self._collect_child_layers(child))
        return layers