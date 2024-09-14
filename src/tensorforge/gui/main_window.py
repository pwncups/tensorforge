from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, QSplitter,
                             QHBoxLayout, QPushButton, QScrollArea, QFileDialog)
from PyQt6.QtCore import Qt
from tensorforge.gui.comparison_view import ComparisonView
from tensorforge.gui.diff_view import DiffView
from tensorforge.gui.log_view import LogView
from tensorforge.gui.model_builder_view import ModelBuilderView
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.setWindowTitle("TensorForge")
            self.setGeometry(100, 100, 1600, 900)
            self.init_ui()
        except Exception as e:
            logger.error(f"Error initializing main window: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to initialize application: {str(e)}")

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Create a splitter for the main area
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Create a horizontal splitter for comparison, model builder, and diff views
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Add ComparisonView to the splitter
        self.comparison_view = ComparisonView()
        top_splitter.addWidget(self.comparison_view)

        # Add ModelBuilderView to the splitter (in the middle)
        self.model_builder_view = ModelBuilderView()
        top_splitter.addWidget(self.model_builder_view)

        # Add DiffView to the splitter
        self.diff_view = DiffView()
        top_splitter.addWidget(self.diff_view)

        # Set the initial sizes of the top splitter
        top_splitter.setSizes([500, 600, 500])

        main_splitter.addWidget(top_splitter)

        # Add LogView to the main splitter
        self.log_view = LogView()
        main_splitter.addWidget(self.log_view)

        # Set the initial sizes of the main splitter
        main_splitter.setSizes([700, 200])

        main_layout.addWidget(main_splitter)

        # Add buttons
        button_layout = QHBoxLayout()
        diff_button = QPushButton("DIFF Models")
        diff_button.clicked.connect(self.diff_models)
        button_layout.addWidget(diff_button)

        extract_button = QPushButton("Save Diff Report")
        extract_button.clicked.connect(self.save_diff_report)
        button_layout.addWidget(extract_button)

        build_model_button = QPushButton("Build New Model")
        build_model_button.clicked.connect(self.build_new_model)
        button_layout.addWidget(build_model_button)

        main_layout.addLayout(button_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Connect signals
        self.comparison_view.add_to_builder_signal.connect(self.model_builder_view.add_item)

    def add_to_model_builder(self, data):
        self.model_builder_view.add_item(data["item"])

    def add_layers_to_builder(self, layers):
        for layer_name, layer_info in layers.items():
            self.model_builder_view.add_layer(layer_name, layer_info)

    def diff_models(self):
        self.log_view.log("Starting model comparison...")
        model1 = self.comparison_view.model1
        model2 = self.comparison_view.model2
        if model1 and model2:
            try:
                diff_result = model1.diff(model2)
                self.diff_view.display_diff(diff_result)
                self.log_view.log("Model comparison completed successfully.")
            except Exception as e:
                self.log_view.log(f"Error during model comparison: {str(e)}")
        else:
            self.log_view.log("Please load both models before diffing")

    def extract_checkpoint(self):
        if not self.comparison_view.model1 or not self.comparison_view.model2:
            self.log_view.log("Please load both models before extracting checkpoint")
            return

        diff_result = self.comparison_view.model1.diff(self.comparison_view.model2)
        if diff_result['similarity'] < 50:  # Lowered the threshold
            self.log_view.log("Models are not similar enough for extraction")
            return

        # Determine which model is the larger one
        larger_model = max(self.comparison_view.model1, self.comparison_view.model2,
                           key=lambda m: sum(np.prod(v['shape']) for v in m.layer_info.values()))

        output_path, _ = QFileDialog.getSaveFileName(self, "Save Extracted Checkpoint", "", "Safetensor Files (*.safetensors)")
        if output_path:
            try:
                extracted_path = larger_model.extract_diffusion_model(output_path)
                self.log_view.log(f"Diffusion model extracted successfully to {extracted_path}")
            except Exception as e:
                self.log_view.log(f"Error extracting checkpoint: {str(e)}")

    def save_diff_report(self):
        if not self.comparison_view.model1 or not self.comparison_view.model2:
            self.log_view.log("Please load both models before saving diff report")
            return

        diff_result = self.comparison_view.model1.diff(self.comparison_view.model2)

        # Convert numpy types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Diff Report", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(diff_result, f, indent=2, default=convert_to_serializable)
                self.log_view.log(f"Diff report saved successfully to {file_path}")
            except Exception as e:
                self.log_view.log(f"Error saving diff report: {str(e)}")

    def build_new_model(self):
        # Implement the logic to build a new model based on the ModelBuilderView
        pass

