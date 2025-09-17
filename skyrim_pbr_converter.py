# skyrim_pbr_converter_final_v3.py
# Requisitos: Pillow, numpy, PyQt5, os, subprocess

import os
import sys
import subprocess
import tempfile
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout, QPushButton,
                             QFileDialog, QSlider, QHBoxLayout, QLineEdit, QGridLayout,
                             QGroupBox, QComboBox, QFrame, QMessageBox)
from PyQt5.QtCore import Qt
import numpy as np

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
    normal_out_np = normal_np.copy()
    normal_out_np[..., 1] = 1.0 - normal_out_np[..., 1]
    return normal_out_np

def apply_specular_np(normal_np, roughness_np, roughness_strength=1.0):
    """Applies a specular map (inverted roughness) to the alpha channel of a normal map."""
    specular_map = 1.0 - roughness_np
    specular_map_mod = specular_map * roughness_strength
    
    if specular_map_mod.ndim == 2:
        specular_map_mod = specular_map_mod[..., np.newaxis]
        
    if normal_np.shape[2] == 3:
        normal_with_alpha = np.dstack((normal_np, np.ones_like(normal_np[..., 0:1])))
    else:
        normal_with_alpha = normal_np.copy()

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

def apply_opacity_mask_np(albedo_np, mask_np, opacity=1.0):
    if mask_np.ndim == 2:
        mask_np = mask_np[..., np.newaxis]
    albedo_np[..., 3] = np.clip(mask_np[..., 0] * opacity, 0.0, 1.0)
    return albedo_np

# ---------- GUI ----------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skyrim PBR -> Preview")
        self.preview_size = (512, 512)

        self.albedo = None
        self.ao = None
        self.normal = None
        self.roughness = None
        self.mask = None

        self.metallic_diffuse = None
        self.metallic_other = None

        self.albedo_preview_np = None
        self.ao_preview_np = None
        self.normal_preview_np = None
        self.roughness_preview_np = None
        self.metallic_diffuse_preview_np = None
        self.metallic_other_preview_np = None
        self.mask_preview_np = None

        self.output_folder_path = None

        main_layout = QHBoxLayout()

        # Left side: Preview widget
        left_layout = QVBoxLayout()
        self.preview_label = QLabel("No image loaded")
        self.preview_label.setFixedSize(*self.preview_size)
        self.preview_label.setStyleSheet("background: #222; border: 1px solid #444;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.preview_label)

        preview_selector_layout = QHBoxLayout()
        preview_selector_layout.addWidget(QLabel("View:"))
        self.preview_selector = QComboBox()
        self.preview_selector.addItems(["Albedo Output", "Normal Output", "Roughness Map", "Metallic Map", "AO Map"])
        self.preview_selector.currentIndexChanged.connect(self.process_and_preview)
        preview_selector_layout.addWidget(self.preview_selector)
        left_layout.addLayout(preview_selector_layout)
        
        left_layout.addStretch(1) 

        main_layout.addLayout(left_layout, 1)

        # Right side: Menus and sliders
        right_layout = QVBoxLayout()
        
        # Diffuse group box
        diffuse_group_box = QGroupBox("Diffuse")
        diffuse_grid = QGridLayout()
        diffuse_group_box.setLayout(diffuse_grid)
        
        self.albedo_line = self.create_texture_row("Albedo/Basecolor:", diffuse_grid, 0, self.load_albedo, self.remove_albedo)
        
        self.ao_line = self.create_texture_row("Ambient Occlusion:", diffuse_grid, 1, self.load_ao, self.remove_ao)
        ao_slider_layout = QHBoxLayout()
        ao_slider_label = QLabel("AO Strength:")
        self.ao_slider = QSlider(Qt.Horizontal)
        self.ao_slider.setMinimum(0)
        self.ao_slider.setMaximum(100)
        self.ao_slider.setValue(100)
        self.ao_slider.valueChanged.connect(self.process_and_preview)
        ao_slider_layout.addWidget(ao_slider_label)
        ao_slider_layout.addWidget(self.ao_slider)
        diffuse_grid.addLayout(ao_slider_layout, 2, 0, 1, 4)
        
        self.metallic_diffuse_line = self.create_texture_row("Metallic (for darkening):", diffuse_grid, 3, self.load_metallic_diffuse, self.remove_metallic_diffuse)
        darken_layout = QHBoxLayout()
        darken_label = QLabel("Darken by Metallic:")
        self.darken_metallic_slider = QSlider(Qt.Horizontal)
        self.darken_metallic_slider.setMinimum(0)
        self.darken_metallic_slider.setMaximum(100)
        self.darken_metallic_slider.setValue(50)
        self.darken_metallic_slider.valueChanged.connect(self.process_and_preview)
        
        darken_layout.addWidget(darken_label)
        darken_layout.addWidget(self.darken_metallic_slider)
        diffuse_grid.addLayout(darken_layout, 4, 0, 1, 4)

        right_layout.addWidget(diffuse_group_box)
        
        # Normal Map group box
        normal_group_box = QGroupBox("Normal Map")
        normal_grid = QGridLayout()
        normal_group_box.setLayout(normal_grid)
        
        self.normal_line = self.create_texture_row("Normal:", normal_grid, 0, self.load_normal, self.remove_normal)
        
        normal_type_layout = QHBoxLayout()
        normal_type_layout.addWidget(QLabel("Normal Map Type:"))
        self.normal_type_selector = QComboBox()
        self.normal_type_selector.addItems(["DirectX", "OpenGL"])
        self.normal_type_selector.currentIndexChanged.connect(self.process_and_preview)
        normal_type_layout.addWidget(self.normal_type_selector)
        normal_grid.addLayout(normal_type_layout, 1, 0, 1, 4)

        self.roughness_line = self.create_texture_row("Roughness:", normal_grid, 2, self.load_roughness, self.remove_roughness)
        roughness_slider_layout = QHBoxLayout()
        roughness_slider_label = QLabel("Roughness to Specular Strength:")
        self.roughness_slider = QSlider(Qt.Horizontal)
        self.roughness_slider.setMinimum(0)
        self.roughness_slider.setMaximum(100)
        self.roughness_slider.setValue(100)
        self.roughness_slider.valueChanged.connect(self.process_and_preview)
        roughness_slider_layout.addWidget(roughness_slider_label)
        roughness_slider_layout.addWidget(self.roughness_slider)
        normal_grid.addLayout(roughness_slider_layout, 3, 0, 1, 4)

        right_layout.addWidget(normal_group_box)
        
        # Other Textures group box
        other_group_box = QGroupBox("Other Textures")
        other_grid = QGridLayout()
        other_group_box.setLayout(other_grid)
        
        self.metallic_other_line = self.create_texture_row("Metallic:", other_grid, 0, self.load_metallic_other, self.remove_metallic_other)
        self.mask_line = self.create_texture_row("Opacity:", other_grid, 1, self.load_mask, self.remove_mask)

        right_layout.addWidget(other_group_box)
        
        # Output folder selection
        output_folder_layout = QHBoxLayout()
        output_folder_layout.addWidget(QLabel("Output Folder:"))
        self.output_folder_line = QLineEdit()
        self.output_folder_line.setReadOnly(True)
        output_folder_layout.addWidget(self.output_folder_line)
        self.browse_folder_btn = QPushButton("Browse Folder...")
        self.browse_folder_btn.clicked.connect(self.select_output_folder)
        output_folder_layout.addWidget(self.browse_folder_btn)
        right_layout.addLayout(output_folder_layout)

        # Output buttons
        output_layout = QHBoxLayout()
        self.save_diffuse_btn = QPushButton("Save Diffuse Output")
        self.save_diffuse_btn.clicked.connect(self.save_diffuse_output)
        output_layout.addWidget(self.save_diffuse_btn)
        
        self.save_normal_btn = QPushButton("Save Normal Output")
        self.save_normal_btn.clicked.connect(self.save_normal_output)
        output_layout.addWidget(self.save_normal_btn)
        
        right_layout.addLayout(output_layout)
        right_layout.addStretch(1)

        main_layout.addLayout(right_layout, 1)

        self.setLayout(main_layout)

    def create_texture_row(self, name, grid_layout, row, load_func, remove_func):
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
        p, _ = QFileDialog.getOpenFileName(self, "Open Ambient Occlusion")
        if p:
            self.ao_line.setText(p)
            self.ao = Image.open(p).convert('L')
            self.ao_preview_np = pil_to_numpy(self.ao.resize(self.preview_size, Image.LANCZOS))
            self.process_and_preview()

    def remove_ao(self):
        self.ao = None
        self.ao_preview_np = None
        self.ao_line.clear()
        self.process_and_preview()
        
    def load_metallic_diffuse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Metallic (for darkening)")
        if p:
            self.metallic_diffuse_line.setText(p)
            self.metallic_diffuse = Image.open(p).convert('L')
            self.metallic_diffuse_preview_np = pil_to_numpy(self.metallic_diffuse.resize(self.preview_size, Image.LANCZOS))
            self.process_and_preview()

    def remove_metallic_diffuse(self):
        self.metallic_diffuse = None
        self.metallic_diffuse_preview_np = None
        self.metallic_diffuse_line.clear()
        self.process_and_preview()

    def load_metallic_other(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Metallic")
        if p:
            self.metallic_other_line.setText(p)
            self.metallic_other = Image.open(p).convert('L')
            self.metallic_other_preview_np = pil_to_numpy(self.metallic_other.resize(self.preview_size, Image.LANCZOS))
            self.process_and_preview()

    def remove_metallic_other(self):
        self.metallic_other = None
        self.metallic_other_preview_np = None
        self.metallic_other_line.clear()
        self.process_and_preview()

    def load_normal(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Normal")
        if p:
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
        if p:
            self.roughness_line.setText(p)
            self.roughness = Image.open(p).convert('L')
            self.roughness_preview_np = pil_to_numpy(self.roughness.resize(self.preview_size, Image.LANCZOS))
            self.process_and_preview()

    def remove_roughness(self):
        self.roughness = None
        self.roughness_preview_np = None
        self.roughness_line.clear()
        self.process_and_preview()
        
    def load_mask(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Opacity Mask")
        if p:
            self.mask_line.setText(p)
            self.mask = Image.open(p).convert('L')
            self.mask_preview_np = pil_to_numpy(self.mask.resize(self.preview_size, Image.LANCZOS))
            self.process_and_preview()

    def remove_mask(self):
        self.mask = None
        self.mask_preview_np = None
        self.mask_line.clear()
        self.process_and_preview()

    # ---------- Folder Selection ----------
    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.output_folder_path = folder_path
            self.output_folder_line.setText(folder_path)

    # ---------- Process and Preview ----------
    def process_and_preview(self):
        view_mode = self.preview_selector.currentText()
        preview_np = None
        
        if view_mode == "Albedo Output":
            if self.albedo_preview_np is None:
                self.clear_preview()
                return
            preview_np = self.albedo_preview_np.copy()
            
            if self.ao_preview_np is not None:
                ao_strength = self.ao_slider.value() / 100.0
                preview_np = apply_ao_to_albedo_np(preview_np, self.ao_preview_np, ao_strength)

            if self.metallic_diffuse_preview_np is not None:
                metallic_strength = self.darken_metallic_slider.value() / 100.0
                preview_np = darken_albedo_metallic_np(preview_np, self.metallic_diffuse_preview_np, metallic_strength)

            if self.mask_preview_np is not None:
                preview_np = apply_opacity_mask_np(preview_np, self.mask_preview_np)
        
        elif view_mode == "Normal Output":
            if self.normal_preview_np is None:
                self.clear_preview()
                return
            preview_np = self.normal_preview_np.copy()

            if self.normal_type_selector.currentText() == "OpenGL":
                preview_np = convert_normal_to_directx_np(preview_np)
            
            if self.roughness_preview_np is not None:
                roughness_strength = self.roughness_slider.value() / 100.0
                preview_np = apply_specular_np(preview_np, self.roughness_preview_np, roughness_strength)

        elif view_mode == "Roughness Map":
            if self.roughness_preview_np is not None:
                preview_np = self.roughness_preview_np
        
        elif view_mode == "Metallic Map":
            if self.metallic_other_preview_np is not None:
                preview_np = self.metallic_other_preview_np
            elif self.metallic_diffuse_preview_np is not None:
                preview_np = self.metallic_diffuse_preview_np
        
        elif view_mode == "AO Map":
            if self.ao_preview_np is not None:
                preview_np = self.ao_preview_np

        if preview_np is not None:
            if preview_np.ndim == 2:
                preview_np = np.dstack((preview_np, preview_np, preview_np))
            self.show_preview(numpy_to_pil(preview_np))
        else:
            self.clear_preview()

    # ---------- Texconv Helper Function ----------
    def _save_with_texconv(self, pil_image, file_path, dds_format):
        texconv_path = os.path.join(os.path.dirname(sys.argv[0]), 'resources', 'texconv.exe')
        
        if not os.path.exists(texconv_path):
            QMessageBox.critical(self, "Error", f"Could not find texconv.exe at: {texconv_path}\nMake sure it's in a 'resources' folder next to the script.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_png_path = os.path.join(temp_dir, "temp_image.png")
            pil_image.save(temp_png_path)
            
            try:
                # Command to run texconv.exe
                # -f [format] -o [output_dir] [input_file]
                command = [texconv_path, "-f", dds_format, "-o", self.output_folder_path, temp_png_path]
                
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"Texconv Output: {result.stdout}")
                print(f"Texconv Errors: {result.stderr}")
                
                QMessageBox.information(self, "Success", f"Texture saved as {dds_format} to {file_path}")
            except subprocess.CalledProcessError as e:
                QMessageBox.critical(self, "Conversion Error", f"Texconv failed with error: {e.stderr}\nCommand: {' '.join(command)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")


    # ---------- Output Functions ----------
    def save_diffuse_output(self):
        if self.albedo is None:
            QMessageBox.warning(self, "Error", "No Albedo/Basecolor texture loaded.")
            return

        if self.output_folder_path is None:
            QMessageBox.warning(self, "Error", "Please select an output folder first.")
            return
            
        albedo_out_np = pil_to_numpy(self.albedo.copy())
        
        if self.ao is not None:
            ao_strength = self.ao_slider.value() / 100.0
            ao_full_np = pil_to_numpy(self.ao.resize(self.albedo.size))
            albedo_out_np = apply_ao_to_albedo_np(albedo_out_np, ao_full_np, ao_strength)
        
        if self.metallic_diffuse is not None:
            metallic_strength = self.darken_metallic_slider.value() / 100.0
            metallic_full_np = pil_to_numpy(self.metallic_diffuse.resize(self.albedo.size))
            albedo_out_np = darken_albedo_metallic_np(albedo_out_np, metallic_full_np, metallic_strength)
            
        if self.mask is not None:
            mask_full_np = pil_to_numpy(self.mask.resize(self.albedo.size))
            albedo_out_np = apply_opacity_mask_np(albedo_out_np, mask_full_np)
            
        out_pil = numpy_to_pil(albedo_out_np)

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Diffuse Texture", os.path.join(self.output_folder_path, "diffuse.dds"), "DDS (*.dds)")
        
        if file_path:
            self._save_with_texconv(out_pil, file_path, "BC3_UNORM")


    def save_normal_output(self):
        if self.normal is None:
            QMessageBox.warning(self, "Error", "No Normal texture loaded.")
            return
        
        if self.output_folder_path is None:
            QMessageBox.warning(self, "Error", "Please select an output folder first.")
            return

        normal_out_np = pil_to_numpy(self.normal.copy())
        
        if self.normal_type_selector.currentText() == "OpenGL":
            normal_out_np = convert_normal_to_directx_np(normal_out_np)
        
        if self.roughness is not None:
            roughness_strength = self.roughness_slider.value() / 100.0
            roughness_full_np = pil_to_numpy(self.roughness.resize(self.normal.size))
            normal_out_np = apply_specular_np(normal_out_np, roughness_full_np, roughness_strength)
        
        out_pil = numpy_to_pil(normal_out_np)

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Normal Texture", os.path.join(self.output_folder_path, "normal.dds"), "DDS (*.dds)")
        
        if file_path:
            self._save_with_texconv(out_pil, file_path, "BC5_UNORM")

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