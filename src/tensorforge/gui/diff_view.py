from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QScrollArea, QLabel, QTreeWidget, QTreeWidgetItem
from PyQt6.QtCore import Qt


class DiffView(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.diff_widget = QWidget()
        self.diff_layout = QVBoxLayout(self.diff_widget)

        self.scroll_area.setWidget(self.diff_widget)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

    def display_diff(self, diff_result):
        # Clear previous results
        for i in reversed(range(self.diff_layout.count())):
            self.diff_layout.itemAt(i).widget().setParent(None)

        # Display summary
        summary = f"""
        Overall similarity: {diff_result['similarity']:.2f}%
        Total parameters (Model 1): {diff_result['total_params_self']:,}
        Total parameters (Model 2): {diff_result['total_params_other']:,}
        Matched parameters: {diff_result['matched_params']:,}
        Model 1 size: {diff_result['self_size'] / (1024 * 1024):.2f} MB
        Model 2 size: {diff_result['other_size'] / (1024 * 1024):.2f} MB
        """
        summary_label = QLabel(summary)
        summary_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.diff_layout.addWidget(summary_label)

        # Display matched layers
        if diff_result['matched_layers']:
            matched_label = QLabel("Matched Layers:")
            matched_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.diff_layout.addWidget(matched_label)

            matched_tree = QTreeWidget()
            matched_tree.setHeaderLabels(["Model 1 Layer", "Model 2 Layer", "Similarity", "Shape", "Dtype"])
            for match in diff_result['matched_layers']:
                QTreeWidgetItem(matched_tree, [
                    match['self_key'],
                    match['other_key'],
                    f"{match['similarity']:.2f}",
                    str(match['shape']),
                    match['dtype']
                ])
            self.diff_layout.addWidget(matched_tree)

        # Display partial matches
        if diff_result['partial_matches']:
            partial_label = QLabel("Partial Matches:")
            partial_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.diff_layout.addWidget(partial_label)

            partial_tree = QTreeWidget()
            partial_tree.setHeaderLabels(["Model 1 Layer", "Model 2 Layer", "Similarity", "Shape 1", "Shape 2"])
            for match in diff_result['partial_matches']:
                QTreeWidgetItem(partial_tree, [
                    match['self_key'],
                    match['other_key'],
                    f"{match['similarity']:.2f}",
                    str(match['self_shape']),
                    str(match['other_shape'])
                ])
            self.diff_layout.addWidget(partial_tree)

        # Display unmatched layers
        unmatched_label = QLabel("Unmatched Layers:")
        unmatched_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.diff_layout.addWidget(unmatched_label)

        unmatched_tree = QTreeWidget()
        unmatched_tree.setHeaderLabels(["Model", "Layer", "Shape", "Dtype"])
        for layer in diff_result['unmatched_layers_self']:
            QTreeWidgetItem(unmatched_tree, ["Model 1", layer['key'], str(layer['shape']), layer['dtype']])
        for layer in diff_result['unmatched_layers_other']:
            QTreeWidgetItem(unmatched_tree, ["Model 2", layer['key'], str(layer['shape']), layer['dtype']])
        self.diff_layout.addWidget(unmatched_tree)

        # Display weight comparison results
        if 'weight_similarity' in diff_result:
            weight_summary = f"Weight similarity: {diff_result['weight_similarity']:.2f}%"
            weight_summary_label = QLabel(weight_summary)
            weight_summary_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.diff_layout.addWidget(weight_summary_label)

        if diff_result.get('weight_differences'):
            weight_diff_label = QLabel("Weight Differences:")
            weight_diff_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.diff_layout.addWidget(weight_diff_label)

            weight_diff_tree = QTreeWidget()
            weight_diff_tree.setHeaderLabels(["Layer", "Differences"])
            for layer, diffs in diff_result['weight_differences'].items():
                layer_item = QTreeWidgetItem(weight_diff_tree, [layer])
                for diff in diffs:
                    QTreeWidgetItem(layer_item, ["", diff])
            self.diff_layout.addWidget(weight_diff_tree)

        # Add stretch to push all widgets to the top
        self.diff_layout.addStretch()