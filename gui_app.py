# VERSION 1

# import sys
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QMessageBox
# from PyQt5.QtGui import QPixmap
# from PyQt5.QtCore import Qt
#
# # ===== Load model =====
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model = models.resnet50(weights=None)  # no need to download weights
# num_ftrs = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Linear(num_ftrs, 256),
#     nn.ReLU(),
#     nn.Dropout(0.4),
#     nn.Linear(256, 4)
# )
# model.load_state_dict(torch.load("best_resnet_model.pth", map_location=device))
# model.eval()
#
# # Class names (same as in training)
# class_names = [
#     'adenocarcinoma',
#     'large.cell.carcinoma',
#     'normal',
#     'squamous.cell.carcinoma'
# ]
#
# # ===== Image Transform =====
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])
#
# # ===== GUI Class =====
# class TumorClassifierApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Lung Tumor Classifier")
#         self.setFixedSize(400, 500)
#
#         self.layout = QVBoxLayout()
#
#         self.image_label = QLabel("Upload a CT scan image")
#         self.image_label.setAlignment(Qt.AlignCenter)
#         self.layout.addWidget(self.image_label)
#
#         self.result_label = QLabel("")
#         self.result_label.setAlignment(Qt.AlignCenter)
#         self.layout.addWidget(self.result_label)
#
#         self.upload_btn = QPushButton("Browse Image")
#         self.upload_btn.clicked.connect(self.load_image)
#         self.layout.addWidget(self.upload_btn)
#
#         self.setLayout(self.layout)
#
#     def load_image(self):
#         path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Image Files (*.png *.jpg *.jpeg)")
#         if path:
#             self.display_image(path)
#             self.predict_image(path)
#
#     def display_image(self, path):
#         pixmap = QPixmap(path).scaled(300, 300, Qt.KeepAspectRatio)
#         self.image_label.setPixmap(pixmap)
#
#     def predict_image(self, path):
#         try:
#             image = Image.open(path).convert("RGB")
#             input_tensor = transform(image).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 _, predicted = torch.max(output, 1)
#                 pred_class = class_names[predicted.item()]
#                 self.result_label.setText(f"Prediction: {pred_class}")
#         except Exception as e:
#             QMessageBox.critical(self, "Error", str(e))
#
#
# # ===== Main Runner =====
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = TumorClassifierApp()
#     window.show()
#     sys.exit(app.exec_())

# ------------------------------------------------------------------------------------------------------

# VERSION 2

import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFrame
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

# ===== Load model =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 4)
)
model.load_state_dict(torch.load("best_resnet_model.pth", map_location=device))
model.eval()

class_names = [
    'Adenocarcinoma',
    'Large Cell Carcinoma',
    'Normal',
    'Squamous Cell Carcinoma'
]

# ===== Image Transform =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ===== GUI App =====
class TumorClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung Tumor Recognition")
        self.setFixedSize(500, 600)
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()

        self.title = QLabel("Lung Tumor Recognition")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(QFont("Arial", 16, QFont.Bold))
        self.layout.addWidget(self.title)

        self.image_label = QLabel("Upload a CT scan image")
        self.image_label.setFixedSize(350, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.result_label)

        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setFont(QFont("Arial", 10))
        self.layout.addWidget(self.confidence_label)

        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.line)

        self.upload_btn = QPushButton("Browse Image")
        self.upload_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        self.upload_btn.clicked.connect(self.load_image)
        self.layout.addWidget(self.upload_btn, alignment=Qt.AlignCenter)

        self.setLayout(self.layout)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            self.display_image(path)
            self.predict_image(path)

    def display_image(self, path):
        pixmap = QPixmap(path).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def predict_image(self, path):
        try:
            image = Image.open(path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                pred_class = class_names[pred_idx]

                # Format top 2 predictions
                top2 = torch.topk(probs, 2)
                top2_classes = [class_names[i] for i in top2.indices.tolist()]
                top2_scores = top2.values.tolist()

                self.result_label.setText(f"🧠 Prediction: <b>{pred_class}</b>")
                self.confidence_label.setText(
                    f"Confidence:\n• {top2_classes[0]}: {top2_scores[0]*100:.2f}%\n"
                    f"• {top2_classes[1]}: {top2_scores[1]*100:.2f}%"
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# ===== Main Runner =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TumorClassifierApp()
    window.show()
    sys.exit(app.exec_())

# -----------------------------------------------------------------------------------------------------