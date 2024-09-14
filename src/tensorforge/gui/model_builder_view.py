from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem,
                             QScrollArea, QPushButton, QMenu, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal
import logging
import json
from tensorforge.models.safetensor_model import SafetensorModel

logger = logging.getLogger(__name__)

class ModelBuilderView(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.layer_sources = {}

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Model Builder")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        self.layer_tree = QTreeWidget()
        self.layer_tree.setHeaderLabels(["Layer"])
        self.layer_tree.setAcceptDrops(True)
        self.layer_tree.setDragDropMode(QTreeWidget.DragDropMode.InternalMove)
        self.layer_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layer_tree.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.layer_tree)

        save_button = QPushButton("Save New Model")
        save_button.clicked.connect(self.save_new_model)
        layout.addWidget(save_button)

        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/json"):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/json"):
            item_data = json.loads(event.mimeData().data("application/json").data().decode())
            self.add_item(item_data, None)  # Pass None as source_path for drag-and-drop
            event.accept()
        else:
            event.ignore()

    def add_item(self, item_data, source_path=None, parent_item=None):
        try:
            if isinstance(item_data, dict):  # For drag and drop
                text = item_data["text"].split(': ')[0]  # Remove any extra info after the colon
                new_item = QTreeWidgetItem([text])
                if "layer_info" in item_data and item_data["layer_info"]:
                    new_item.setData(0, Qt.ItemDataRole.UserRole, item_data["layer_info"])
                    if source_path:
                        self.layer_sources[id(new_item)] = source_path
                if parent_item:
                    parent_item.addChild(new_item)
                else:
                    self.layer_tree.addTopLevelItem(new_item)
                for child in item_data.get("children", []):
                    self.add_item(child, source_path, new_item)
            else:  # For right-click add
                new_item = self._copy_item(item_data, source_path)
                if parent_item:
                    parent_item.addChild(new_item)
                else:
                    self.layer_tree.addTopLevelItem(new_item)
                self._add_children(item_data, new_item, source_path)

            if not parent_item:
                self.layer_tree.expandItem(new_item)
        except Exception as e:
            logger.error(f"Error adding item to model builder: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add item to model builder: {str(e)}")

    def _add_children(self, source_item, target_item, source_path):
        for i in range(source_item.childCount()):
            child = source_item.child(i)
            new_child = self._copy_item(child, source_path)
            target_item.addChild(new_child)
            self._add_children(child, new_child, source_path)

    def _copy_item(self, item, source_path):
        new_item = QTreeWidgetItem([item.text(0)])
        new_item.setData(0, Qt.ItemDataRole.UserRole, item.data(0, Qt.ItemDataRole.UserRole))
        self.layer_sources[id(new_item)] = source_path
        return new_item

    def show_context_menu(self, position):
        item = self.layer_tree.itemAt(position)
        if item:
            menu = QMenu(self)
            remove_action = menu.addAction("Remove")
            action = menu.exec(self.layer_tree.mapToGlobal(position))
            if action == remove_action:
                self.layer_tree.takeTopLevelItem(self.layer_tree.indexOfTopLevelItem(item))

    def save_new_model(self):
        layers = self.collect_layers()
        if not layers:
            QMessageBox.warning(self, "No Layers", "No layers to save. Please add layers to the model first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save New Model", "", "Safetensor Files (*.safetensors)")
        if file_path:
            try:
                SafetensorModel.save_new_model(layers, file_path, self.layer_sources)
                QMessageBox.information(self, "Success", f"Model saved successfully to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")

    def collect_layers(self):
        layers = {}
        root = self.layer_tree.invisibleRootItem()
        for i in range(root.childCount()):
            self._collect_layers_recursive(root.child(i), layers)
        return layers

    def _collect_layers_recursive(self, item, layers, prefix=''):
        full_name = f"{prefix}{item.text(0)}" if prefix else item.text(0)
        layer_info = item.data(0, Qt.ItemDataRole.UserRole)
        if layer_info:
            layers[full_name] = {
                'info': layer_info,
                'source': self.layer_sources.get(id(item), None)
            }

        for i in range(item.childCount()):
            self._collect_layers_recursive(item.child(i), layers, f"{full_name}.")
