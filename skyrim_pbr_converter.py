# skyrim_pbr_converter_final.py
# Requisitos: Pillow, numpy, PyQt5

from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout, QPushButton,
                             QFileDialog, QSlider, QHBoxLayout, QLineEdit, QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt
import numpy as np
import sys

# ---------- Image Utilities ----------
def pil_to_numpy(img):
    """Converts a PIL image to a normalized NumPy array."""
    return np.array(img).astype(np.float32) / 255.0

def numpy_to_pil(arr):
    """Converts a NumPy array back to a PIL image."""
    arr = np.clip((arr * 255.0).round(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_ao_to_albedo_np(albedo_np, ao_np, ao_strength=1.0):
    """Applies ambient occlusion to an albedo texture using NumPy arrays."""
    if ao_np.ndim == 2:
        ao_np = ao_np[..., np.newaxis]
    albedo_rgb = albedo_np[..., :3]
    ao_mod = ao_np * ao_strength + (1.0 - ao_strength)
    out_rgb = albedo_rgb * ao_mod
    
    final_np = np.dstack((out_rgb, albedo_np[..., 3]))
    return final_np

def convert_normal_to_directx_np(normal_np):
    """Converts a normal map from OpenGL to DirectX (inverts G channel)."""
    normal_np[..., 1] = 1.0 - normal_np[..., 1]
    return normal_np

def apply_specular_np(normal_np, roughness_np, roughness_strength=1.0):
    """Applies a specular map (inverted roughness) to the alpha channel of a normal map."""
    # Invertir Roughness (1.0 - roughness)
    specular_map = 1.0 - roughness_np
    
    # Atenuar con el valor del slider
    specular_map_mod = specular_map * roughness_strength
    
    # Asegurarse de que el mapa especular tiene una dimensión extra
    if specular_map_mod.ndim == 2:
        specular_map_mod = specular_map_mod[..., np.newaxis]
        
    # Crear un nuevo normal map con canal alpha
    if normal_np.shape[2] == 3:
        normal_with_alpha = np.dstack((normal_np, np.ones_like(normal_np[..., 0:1])))
    else:
        normal_with_alpha = normal_np.copy()

    # Copiar al canal alpha del normal map
    normal_with_alpha[..., 3] = specular_map_mod[..., 0]
    return normal_with_alpha

def darken_albedo_metallic_np(albedo_np, metallic_np, metallic_strength=1.0):
    """Darkens albedo based on a metallic map to improve cubemap appearance."""
    if metallic_np.ndim == 2:
        metallic_np = metallic_np[..., np.newaxis]
    
    metallic_mod = metallic_np * metallic_strength
    
    out_rgb = albedo_np[..., :3] * (1 - metallic_mod)
    
    final_np = np.dstack((out_rgb, albedo_np[..., 3]))
    return final_np


# ---------- GUI ----------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skyrim PBR -> Preview")
        self.preview_size = (512, 512)

        # Full-resolution texture slots
        self.albedo = None
        self.ao = None
        self.normal = None
        self.roughness = None
        self.metallic = None
        self.mask = None
        
        # Pre-resized NumPy versions for fast previewing
        self.albedo_preview_np = None
        self.ao_preview_np = None
        self.normal_preview_np = None
        self.roughness_preview_np = None
        self.metallic_preview_np = None
        self.mask_preview_np = None

        main_layout = QVBoxLayout()

        # Preview Widget
        self.preview_label = QLabel("No image loaded")
        self.preview_label.setFixedSize(*self.preview_size)
        self.preview_label.setStyleSheet("background: #222; border: 1px solid #444;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.preview_label)

        # Texture Slots Group Box
        texture_group_box = QGroupBox("Texture Slots")
        texture_grid = QGridLayout()
        texture_group_box.setLayout(texture_grid)

        # Create texture rows and store references to the QLineEdits
        self.albedo_line = self.create_texture_row("Albedo:", texture_grid, 0, self.load_albedo, self.remove_albedo)
        self.ao_line = self.create_texture_row("AO:", texture_grid, 1, self.load_ao, self.remove_ao)
        self.normal_line = self.create_texture_row("Normal:", texture_grid, 2, self.load_normal, self.remove_normal)
        self.roughness_line = self.create_texture_row("Roughness:", texture_grid, 3, self.load_roughness, self.remove_roughness)
        self.metallic_line = self.create_texture_row("Metallic:", texture_grid, 4, self.load_metallic, self.remove_metallic)
        self.mask_line = self.create_texture_row("Opacity:", texture_grid, 5, self.load_mask, self.remove_mask)
        
        main_layout.addWidget(texture_group_box)

        # Sliders
        main_layout.addWidget(QLabel("AO Strength"))
        self.ao_slider = QSlider(Qt.Horizontal)
        self.ao_slider.setMinimum(0); self.ao_slider.setMaximum(100); self.ao_slider.setValue(100)
        self.ao_slider.valueChanged.connect(self.process_and_preview)
        main_layout.addWidget(self.ao_slider)

        main_layout.addWidget(QLabel("Roughness to Specular Strength"))
        self.roughness_slider = QSlider(Qt.Horizontal)
        self.roughness_slider.setMinimum(0); self.roughness_slider.setMaximum(100); self.roughness_slider.setValue(100)
        self.roughness_slider.valueChanged.connect(self.process_and_preview)
        main_layout.addWidget(self.roughness_slider)

        main_layout.addWidget(QLabel("Metallic Strength (for cubemap)"))
        self.metallic_slider = QSlider(Qt.Horizontal)
        self.metallic_slider.setMinimum(0); self.metallic_slider.setMaximum(100); self.metallic_slider.setValue(50)
        self.metallic_slider.valueChanged.connect(self.process_and_preview)
        main_layout.addWidget(self.metallic_slider)

        # Output buttons
        output_layout = QHBoxLayout()
        self.save_albedo_btn = QPushButton("Save Albedo Output")
        self.save_albedo_btn.clicked.connect(self.save_albedo_output)
        output_layout.addWidget(self.save_albedo_btn)
        
        self.save_normal_btn = QPushButton("Save Normal Output")
        self.save_normal_btn.clicked.connect(self.save_normal_output)
        output_layout.addWidget(self.save_normal_btn)
        
        main_layout.addLayout(output_layout)

        self.setLayout(main_layout)

    def create_texture_row(self, name, grid_layout, row, load_func, remove_func):
        """Creates a row with a label, QLineEdit, and two buttons."""
        label = QLabel(name)
        file_line = QLineEdit()
        file_line.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(load_func)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(remove_func)
        
        grid_layout.addWidget(label, row, 0)
        grid_layout.addWidget(file_line, row, 1)
        grid_layout.addWidget(browse_btn, row, 2)
        grid_layout.addWidget(clear_btn, row, 3)
        
        return file_line

    # ---------- Load / Remove ----------
    def load_albedo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Albedo")
        if p:
            self.albedo_line.setText(p)
            self.albedo = Image.open(p).convert('RGBA')
            self.albedo_preview_np = pil_to_numpy(self.albedo.resize(self.preview_size, Image.LANCZOS))
            self.process_and_preview()

    def remove_albedo(self):
        self.albedo = None
        self.albedo_preview_np = None
        self.albedo_line.clear()
        self.process_and_preview()

    def load_ao(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open AO")
        if p and self.albedo:
            self.ao_line.setText(p)
            self.ao = Image.open(p).convert('L')
            self.ao_preview_np = pil_to_numpy(self.ao.resize(self.preview_size, Image.LANCZOS))
            self.process_and_preview()

    def remove_ao(self):
        self.ao = None
        self.ao_preview_np = None
        self.ao_line.clear()
        self.process_and_preview()

    def load_normal(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Normal")
        if p and self.albedo:
            self.normal_line.setText(p)
            self.normal = Image.open(p).convert('RGBA')
            self.normal_preview_np = pil_to_numpy(self.normal.resize(self.preview_size, Image.LANCZOS))
            self.process_and_preview()

    def remove_normal(self):
        self.normal = None
        self.normal_preview_np = None
        self.normal_line.clear()
        self.process_and_preview()
        
    def load_roughness(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Roughness")
        if p and self.albedo:
            self.roughness_line.setText(p)
            self.roughness = Image.open(p).convert('L')
            self.roughness_preview_np = pil_to_numpy(self.roughness.resize(self.preview_size, Image.LANCZOS))
            self.process_and_preview()

    def remove_roughness(self):
        self.roughness = None
        self.roughness_preview_np = None
        self.roughness_line.clear()
        self.process_and_preview()
        
    def load_metallic(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Metallic")
        if p and self.albedo:
            self.metallic_line.setText(p)
            self.metallic = Image.open(p).convert('L')
            self.metallic_preview_np = pil_to_numpy(self.metallic.resize(self.preview_size, Image.LANCZOS))
            self.process_and_preview()

    def remove_metallic(self):
        self.metallic = None
        self.metallic_preview_np = None
        self.metallic_line.clear()
        self.process_and_preview()

    def load_mask(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Opacity Mask")
        if p and self.albedo:
            self.mask_line.setText(p)
            self.mask = Image.open(p).convert('L')
            self.mask_preview_np = pil_to_numpy(self.mask.resize(self.preview_size, Image.LANCZOS))
            self.process_and_preview()

    def remove_mask(self):
        self.mask = None
        self.mask_preview_np = None
        self.mask_line.clear()
        self.process_and_preview()

    # ---------- Process and Preview ----------
    def process_and_preview(self):
        if self.albedo_preview_np is None:
            self.clear_preview()
            return
        
        # Albedo Output
        albedo_out_np = self.albedo_preview_np.copy()
        
        if self.ao_preview_np is not None:
            ao_strength = self.ao_slider.value() / 100.0
            albedo_out_np = apply_ao_to_albedo_np(albedo_out_np, self.ao_preview_np, ao_strength)

        if self.metallic_preview_np is not None:
            metallic_strength = self.metallic_slider.value() / 100.0
            albedo_out_np = darken_albedo_metallic_np(albedo_out_np, self.metallic_preview_np, metallic_strength)

        if self.mask_preview_np is not None:
            albedo_out_np = apply_opacity_mask_np(albedo_out_np, self.mask_preview_np)

        # Normal Output (separate processing pipeline)
        normal_out_np = None
        if self.normal_preview_np is not None:
            normal_out_np = self.normal_preview_np.copy()
            
            # Convert normal from OpenGL to DirectX (invert green channel)
            normal_out_np = convert_normal_to_directx_np(normal_out_np)
            
            # Add specular to the alpha channel
            if self.roughness_preview_np is not None:
                roughness_strength = self.roughness_slider.value() / 100.0
                normal_out_np = apply_specular_np(normal_out_np, self.roughness_preview_np, roughness_strength)

        # Show Albedo output in preview for now
        self.show_preview(numpy_to_pil(albedo_out_np))

    # ---------- Output Functions ----------
    def save_albedo_output(self):
        if self.albedo is None:
            return
            
        albedo_out_np = pil_to_numpy(self.albedo.copy())
        
        if self.ao is not None:
            ao_strength = self.ao_slider.value() / 100.0
            ao_full_np = pil_to_numpy(self.ao.resize(self.albedo.size))
            albedo_out_np = apply_ao_to_albedo_np(albedo_out_np, ao_full_np, ao_strength)
        
        if self.metallic is not None:
            metallic_strength = self.metallic_slider.value() / 100.0
            metallic_full_np = pil_to_numpy(self.metallic.resize(self.albedo.size))
            albedo_out_np = darken_albedo_metallic_np(albedo_out_np, metallic_full_np, metallic_strength)
            
        if self.mask is not None:
            mask_full_np = pil_to_numpy(self.mask.resize(self.albedo.size))
            albedo_out_np = apply_opacity_mask_np(albedo_out_np, mask_full_np)
            
        out_pil = numpy_to_pil(albedo_out_np)
        p, _ = QFileDialog.getSaveFileName(self, "Save Albedo Texture", "albedo_out.png", "PNG (*.png);;DDS (*.dds)")
        if p:
            out_pil.save(p)
            print(f"Albedo texture saved to {p}")

    def save_normal_output(self):
        if self.normal is None:
            return
            
        normal_out_np = pil_to_numpy(self.normal.copy())
        
        # Convert to DirectX (invert green channel)
        normal_out_np = convert_normal_to_directx_np(normal_out_np)
        
        # Add specular to alpha channel
        if self.roughness is not None:
            roughness_strength = self.roughness_slider.value() / 100.0
            roughness_full_np = pil_to_numpy(self.roughness.resize(self.normal.size))
            normal_out_np = apply_specular_np(normal_out_np, roughness_full_np, roughness_strength)
        
        out_pil = numpy_to_pil(normal_out_np)
        p, _ = QFileDialog.getSaveFileName(self, "Save Normal Texture", "normal_out.png", "PNG (*.png);;DDS (*.dds)")
        if p:
            out_pil.save(p)
            print(f"Normal texture saved to {p}")

    # ---------- Preview Helper ----------
    def show_preview(self, pil_img):
        arr = np.array(pil_img.convert("RGBA"))
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGBA8888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    def clear_preview(self):
        self.preview_label.setText("No image loaded")
        self.preview_label.setPixmap(QPixmap())

# ---------- Run ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())