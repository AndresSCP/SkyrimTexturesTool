from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog, QSlider, QHBoxLayout
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
import sys

def load_image(path):
    return Image.open(path).convert('RGBA')

def pil_to_numpy(img):
    return np.array(img).astype(np.float32) / 255.0

def numpy_to_pil(arr):
    arr = np.clip((arr*255.0).round(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skyrim Textures Tools Demo")
        
        self.albedo = None
        self.ao = None

        self.preview_label = QLabel("No image")
        self.preview_label.setFixedSize(512,512)
        self.preview_label.setStyleSheet("background: #222")

        layout = QVBoxLayout()
        layout.addWidget(self.preview_label)

        # --------- ALBEDO SLOT ---------
        self.albedo_path_label = QLabel("No albedo loaded")
        load_alb_btn = QPushButton("Load Albedo")
        load_alb_btn.clicked.connect(self.load_albedo)
        remove_alb_btn = QPushButton("Remove Albedo")
        remove_alb_btn.clicked.connect(self.remove_albedo)

        row_alb = QHBoxLayout()
        row_alb.addWidget(load_alb_btn)
        row_alb.addWidget(self.albedo_path_label)
        row_alb.addWidget(remove_alb_btn)
        layout.addLayout(row_alb)

        # --------- AO SLOT ---------
        self.ao_path_label = QLabel("No AO loaded")
        load_ao_btn = QPushButton("Load AO")
        load_ao_btn.clicked.connect(self.load_ao)
        remove_ao_btn = QPushButton("Remove AO")
        remove_ao_btn.clicked.connect(self.remove_ao)

        self.ao_slider = QSlider(Qt.Horizontal)
        self.ao_slider.setMinimum(0); self.ao_slider.setMaximum(100); self.ao_slider.setValue(100)

        row_ao = QHBoxLayout()
        row_ao.addWidget(load_ao_btn)
        row_ao.addWidget(self.ao_path_label)
        row_ao.addWidget(remove_ao_btn)
        layout.addLayout(row_ao)
        layout.addWidget(QLabel("AO Strength"))
        layout.addWidget(self.ao_slider)

        self.setLayout(layout)

    # --------- Funciones ALBEDO ---------
    def load_albedo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Albedo")
        if p:
            self.albedo = load_image(p)
            self.albedo_path_label.setText(p)
            self.show_preview(self.albedo)

    def remove_albedo(self):
        self.albedo = None
        self.albedo_path_label.setText("No albedo loaded")
        self.preview_label.setText("No image")
        self.preview_label.setPixmap(QPixmap())

    # --------- Funciones AO ---------
    def load_ao(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open AO")
        if p:
            self.ao = Image.open(p).convert('L')
            self.ao_path_label.setText(p)

    def remove_ao(self):
        self.ao = None
        self.ao_path_label.setText("No AO loaded")

    # --------- Preview ---------
    def show_preview(self, pil_img):
        resized = pil_img.convert("RGBA").resize((512,512), Image.LANCZOS)
        arr = np.array(resized)
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
