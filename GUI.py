import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLineEdit, QTextEdit, QMessageBox, QRadioButton
import scipy.io
import GUI_functions
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arrhythmia Prediction")
        self.setGeometry(100, 100, 400, 400)

        # Create widgets
        self.label = QLabel("Select a .mat file:")
        self.button = QPushButton("Browse")
        self.button.clicked.connect(self.browse_file)
        self.fs_label = QLabel("Enter Sampling Frequency (Hz):")
        self.fs_input = QLineEdit()
        self.result_label = QTextEdit()  # Use QTextEdit for multi-line text display

        # Radio buttons for selecting model
        self.model_label = QLabel("Select Model:")
        self.bilstm_radio = QRadioButton("BiLSTM")
        self.lstm_radio = QRadioButton("LSTM")
        self.lstm_radio.setChecked(True)  # Set LSTM as default
        self.model_selection = "LSTM"  # Default model selection
        self.bilstm_radio.toggled.connect(self.model_selection_changed)
        self.lstm_radio.toggled.connect(self.model_selection_changed)

        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.fs_label)
        layout.addWidget(self.fs_input)
        layout.addWidget(self.model_label)
        layout.addWidget(self.bilstm_radio)
        layout.addWidget(self.lstm_radio)
        layout.addWidget(self.result_label)

        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def model_selection_changed(self):
        sender = self.sender()
        if sender.isChecked():
            self.model_selection = sender.text()

    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open .mat file", "", "MAT Files (*.mat)")
        if filename:
            filename_mat = os.path.basename(filename)
            self.result_label.append(f"MAT file: {filename_mat}")
            pleth_signal = self.extract_pleth_signal(filename)
            # Update the result_label with the array of the extracted PPG signal
            fs = self.get_sampling_frequency()
            if fs is None:
                QMessageBox.warning(self, "Missing Information", "Please enter the sampling frequency")
                return

            elif fs is not None:
                model = GUI_functions.bilstm_model if self.model_selection == "BiLSTM" else GUI_functions.lstm_model
                prediction_dict = GUI_functions.predict_arrhythmia(fs, pleth_signal, model)
                # Display prediction_dict
                self.display_prediction(prediction_dict)

    def display_prediction(self, prediction_dict):
        # Format prediction_dict for display
        prediction_text = "Prediction:\n"
        for key, value in prediction_dict.items():
            prediction_text += f"{key}: {value}\n"
        self.result_label.append(prediction_text)

    def extract_pleth_signal(self, record_path):
        mat_data = scipy.io.loadmat(record_path)
        signals = mat_data['val']
        pleth_signal = signals[2]  # PLETH is the third signal
        return pleth_signal

    def get_sampling_frequency(self):
        fs_text = self.fs_input.text()
        try:
            fs = float(fs_text)
            return fs
        except ValueError:
            return None

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())