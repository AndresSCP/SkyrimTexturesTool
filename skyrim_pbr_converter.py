# skyrim_pbr_converter.py
# Requisitos: Pillow, numpy, PyQt5
from PIL import Image, ImageQt
import numpy as np
import sys
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout,
                             QPushButton, QFileDialog, QSlider, QHBoxLayout)
from PyQt5.QtCore import Qt

# ---------- Utilidades de imagen ----------
def load_image(path):
    return Image.open(path).convert('RGBA')

def pil_to_numpy(img):
    return np.array(img).astype(np.float32) / 255.0

def numpy_to_pil(arr):
    arr = np.clip((arr * 255.0).round(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# A: Multiplicar AO en albedo
def apply_ao_to_albedo(albedo_img, ao_img, ao_strength=1.0):
    albedo = pil_to_numpy(albedo_img)[..., :3]
    ao = pil_to_numpy(ao_img)[..., 0:1]  # asumimos grayscale AO
    out = albedo * (1.0 - ao_strength * (1.0 - ao))  # alternativa: albedo * ao
    # preferible: albedo * (ao * ao_strength + (1-ao_strength))
    return numpy_to_pil(np.dstack((out, np.ones_like(out[...,0:1]))))

# B: Aplicar máscara de opacidad a canal Alpha
def apply_opacity_mask(albedo_img, mask_img, opacity=1.0, invert=False):
    al = albedo_img.convert('RGBA')
    al_np = pil_to_numpy(al)
    mask = pil_to_numpy(mask_img)[..., 0]
    if invert:
        mask = 1.0 - mask
    al_np[..., 3] = np.clip(mask * opacity, 0.0, 1.0)
    return numpy_to_pil(al_np)

# C: Convertir normal OpenGL->DirectX (invertir canal G)
def convert_normal_to_directx(normal_img):
    n = normal_img.convert('RGB')
    arr = np.array(n).astype(np.uint8)
    arr[..., 1] = 255 - arr[..., 1]
    return Image.fromarray(arr)

# D: Combinar dos normal maps (vector add + renormalize)
def combine_normals(normal_base_img, normal_detail_img, detail_strength=1.0):
    b = pil_to_numpy(normal_base_img.convert('RGB'))[..., :3]
    d = pil_to_numpy(normal_detail_img.convert('RGB'))[..., :3]
    # RGB->vector [-1,1]
    vb = b * 2.0 - 1.0
    vd = d * 2.0 - 1.0
    v = vb + vd * detail_strength
    norm = np.linalg.norm(v, axis=2, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    v = v / norm
    out = (v + 1.0) * 0.5
    return numpy_to_pil(np.dstack((out, np.ones_like(out[...,0:1]))))

# ---------- GUI mínima ----------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skyrim PBR → Skyrim converter (demo)")
        self.albedo = None
        self.ao = None
        self.mask = None
        self.normal = None
        self.normal_detail = None

        self.preview_label = QLabel("No image")
        self.preview_label.setFixedSize(512, 512)
        self.preview_label.setStyleSheet("background: #222;")

        load_alb_btn = QPushButton("Load Albedo")
        load_alb_btn.clicked.connect(self.load_albedo)
        load_ao_btn = QPushButton("Load AO")
        load_ao_btn.clicked.connect(self.load_ao)
        load_mask_btn = QPushButton("Load Opacity Mask")
        load_mask_btn.clicked.connect(self.load_mask)
        load_n_btn = QPushButton("Load Normal")
        load_n_btn.clicked.connect(self.load_normal)
        load_nd_btn = QPushButton("Load Normal Detail")
        load_nd_btn.clicked.connect(self.load_normal_detail)
        process_btn = QPushButton("Apply AO + Mask + Convert Normal")
        process_btn.clicked.connect(self.process_and_preview)

        self.ao_slider = QSlider(Qt.Horizontal)
        self.ao_slider.setMinimum(0); self.ao_slider.setMaximum(100); self.ao_slider.setValue(100)

        self.detail_slider = QSlider(Qt.Horizontal)
        self.detail_slider.setMinimum(0); self.detail_slider.setMaximum(100); self.detail_slider.setValue(50)

        layout = QVBoxLayout()
        layout.addWidget(self.preview_label)
        row = QHBoxLayout()
        row.addWidget(load_alb_btn); row.addWidget(load_ao_btn); row.addWidget(load_mask_btn)
        layout.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(load_n_btn); row2.addWidget(load_nd_btn); row2.addWidget(process_btn)
        layout.addLayout(row2)
        layout.addWidget(QLabel("AO Strength")); layout.addWidget(self.ao_slider)
        layout.addWidget(QLabel("Normal Detail Strength")); layout.addWidget(self.detail_slider)
        self.setLayout(layout)

    def load_albedo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Albedo")
        if p:
            self.albedo = load_image(p)
            self.show_preview(self.albedo)

    def load_ao(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open AO")
        if p:
            self.ao = Image.open(p).convert('L')

    def load_mask(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Mask")
        if p:
            self.mask = Image.open(p).convert('L')

    def load_normal(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Normal")
        if p:
            self.normal = Image.open(p).convert('RGB')

    def load_normal_detail(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Normal Detail")
        if p:
            self.normal_detail = Image.open(p).convert('RGB')

    def process_and_preview(self):
        if self.albedo is None:
            return
        ao_strength = self.ao_slider.value() / 100.0
        detail_strength = self.detail_slider.value() / 100.0

        out = self.albedo
        if self.ao is not None:
            out = apply_ao_to_albedo(out, self.ao, ao_strength)
        if self.mask is not None:
            out = apply_opacity_mask(out, self.mask, opacity=1.0)

        # prepare final normal
        final_normal = None
        if self.normal is not None and self.normal_detail is not None:
            final_normal = combine_normals(self.normal, self.normal_detail, detail_strength)
            final_normal = convert_normal_to_directx(final_normal)
        elif self.normal is not None:
            final_normal = convert_normal_to_directx(self.normal)

        # for preview show albedo
        self.show_preview(out)

        # save previews to tmp if you want
        out.save("preview_albedo.png")
        if final_normal:
            final_normal.save("preview_normal_dx.png")

    def show_preview(self, pil_img):
        qimg = ImageQt.ImageQt(pil_img.resize((512,512), Image.LANCZOS))
        pix = qimg.copy()
        self.preview_label.setPixmap(pix)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
