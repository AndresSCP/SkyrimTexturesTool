# skyrim_pbr_converter_improved.py
# Requisitos: Pillow, numpy, PyQt5, os, subprocess, scipy

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap, QFont, QFontDatabase
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout, QPushButton,
                             QFileDialog, QSlider, QHBoxLayout, QLineEdit, QGridLayout,
                             QGroupBox, QComboBox, QFrame, QMessageBox, QTabWidget, QCheckBox,
                             QScrollArea, QSplitter, QSizePolicy, QLayout, QStackedWidget)
from PyQt5.QtCore import Qt, QTimer
import numpy as np
from scipy import ndimage
from nif_preview import load_nif, load_mesh_textures, load_cubemap, texture_roots, resolve_texture

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

def apply_roughness_to_normal_alpha(normal_np, roughness_np, roughness_strength=1.0, min_specular=0.15, max_specular=1.0):
    """
    Aplica roughness invertido al canal alpha del normal map con remapping.
    
    Esta es la forma correcta para Skyrim:
    - Invierte el roughness (PBR) a specular/gloss (Skyrim)
    - Remapea el rango para evitar manchas negras en el juego
    - El motor de Skyrim no maneja bien valores de specular muy bajos (cercanos a 0)
    
    Args:
        normal_np: Normal map RGB(A)
        roughness_np: Roughness map (0=suave, 1=rugoso)
        roughness_strength: Multiplicador de intensidad final
        min_specular: Valor mínimo de specular (evita manchas negras, default 0.15)
        max_specular: Valor máximo de specular (default 1.0)
    """
    if roughness_np is None:
        return normal_np
    
    # Asegurar que roughness sea 2D
    if roughness_np.ndim == 3:
        roughness_np = roughness_np[..., 0]
    
    # Invertir roughness: blanco (rugoso) -> negro, negro (suave) -> blanco
    inverted_roughness = 1.0 - roughness_np
    
    # CRÍTICO: Remapear de [0, 1] a [min_specular, max_specular]
    # Esto evita los valores extremadamente bajos que causan manchas negras en Skyrim
    # Ejemplo: roughness=1.0 (muy rugoso) -> inverted=0.0 -> remapped=0.15 (mate pero visible)
    #          roughness=0.0 (espejo)     -> inverted=1.0 -> remapped=1.0 (muy brillante)
    specular_alpha = inverted_roughness * (max_specular - min_specular) + min_specular
    
    # Aplicar strength (después del remapping)
    specular_alpha = specular_alpha * roughness_strength
    
    # Clip final
    specular_alpha = np.clip(specular_alpha, 0.0, 1.0)
    
    # Crear normal con alpha si no lo tiene
    if normal_np.shape[2] == 3:
        normal_with_alpha = np.dstack((normal_np, np.ones_like(normal_np[..., 0])))
    else:
        normal_with_alpha = normal_np.copy()
    
    # CRÍTICO: Reemplazar el canal alpha completamente
    normal_with_alpha[..., 3] = specular_alpha
    
    return normal_with_alpha

def visualize_normal_with_alpha(normal_np):
    """
    Visualiza el normal map con el alpha aplicado como brillo/intensidad.
    Esto permite ver cómo se vería el normal final con su información de especular.
    """
    if normal_np.shape[2] == 4:
        rgb = normal_np[..., :3]
        alpha = normal_np[..., 3:4]
        # Aplicar el alpha como multiplicador de intensidad
        result = rgb * alpha + rgb * 0.3 * (1.0 - alpha)
        return result
    return normal_np[..., :3]

def blend_multiply(base, blend, opacity=1.0):
    """Multiply blend mode - darkens the base"""
    if blend.ndim == 2:
        blend = blend[..., np.newaxis]
    result = base * blend
    return base * (1.0 - opacity) + result * opacity

def blend_screen(base, blend, opacity=1.0):
    """Screen blend mode - lightens the base"""
    if blend.ndim == 2:
        blend = blend[..., np.newaxis]
    result = 1.0 - (1.0 - base) * (1.0 - blend)
    return base * (1.0 - opacity) + result * opacity

def blend_overlay(base, blend, opacity=1.0):
    """Overlay blend mode - combination of multiply and screen"""
    if blend.ndim == 2:
        blend = blend[..., np.newaxis]
    
    # Overlay formula
    result = np.where(base < 0.5, 
                     2.0 * base * blend,
                     1.0 - 2.0 * (1.0 - base) * (1.0 - blend))
    return base * (1.0 - opacity) + result * opacity

def blend_soft_light(base, blend, opacity=1.0):
    """Soft light blend mode - subtle lighting effect"""
    if blend.ndim == 2:
        blend = blend[..., np.newaxis]
    
    # Soft light formula (Pegtop)
    result = (1.0 - 2.0 * blend) * base**2 + 2.0 * blend * base
    return base * (1.0 - opacity) + result * opacity

def blend_add(base, blend, opacity=1.0):
    """Additive blend - brightens"""
    if blend.ndim == 2:
        blend = blend[..., np.newaxis]
    result = np.clip(base + blend, 0.0, 1.0)
    return base * (1.0 - opacity) + result * opacity

def generate_cavity_from_normal(normal_np, strength=1.0, blur_radius=2):
    """
    Genera un cavity map desde un normal map para exportar.
    Cavity = grietas/hendiduras oscuras.
    
    Usa la divergencia del normal map para detectar áreas cóncavas.
    """
    if normal_np.ndim == 4 or (normal_np.ndim == 3 and normal_np.shape[2] == 4):
        normal_rgb = normal_np[..., :3]
    else:
        normal_rgb = normal_np
    
    # Convertir normal de [0,1] a [-1,1]
    normal_centered = normal_rgb * 2.0 - 1.0
    
    # Calcular derivadas
    dx = ndimage.sobel(normal_centered[..., 0], axis=1)
    dy = ndimage.sobel(normal_centered[..., 1], axis=0)
    
    # Cavity = divergencia (donde la superficie se hunde)
    cavity = dx + dy
    
    # Normalizar
    cavity = cavity - cavity.min()
    if cavity.max() > 0:
        cavity = cavity / cavity.max()
    
    # Invertir (queremos oscuro en cavidades)
    cavity = 1.0 - cavity
    
    # Aplicar blur para suavizar
    if blur_radius > 0:
        cavity = ndimage.gaussian_filter(cavity, sigma=blur_radius)
    
    # Aplicar strength
    cavity = cavity * strength + (1.0 - strength)
    
    return cavity

def generate_edge_from_normal(normal_np, strength=1.0, blur_radius=1):
    """
    Genera un edge/ridge map desde un normal map para exportar.
    Edge = bordes salientes claros.
    
    Usa el gradiente del normal para detectar cambios bruscos (bordes).
    """
    if normal_np.ndim == 4 or (normal_np.ndim == 3 and normal_np.shape[2] == 4):
        normal_rgb = normal_np[..., :3]
    else:
        normal_rgb = normal_np
    
    # Calcular magnitud del gradiente en cada canal
    edge = np.zeros_like(normal_rgb[..., 0])
    
    for i in range(3):
        gx = ndimage.sobel(normal_rgb[..., i], axis=1)
        gy = ndimage.sobel(normal_rgb[..., i], axis=0)
        edge += np.sqrt(gx**2 + gy**2)
    
    # Normalizar
    edge = edge - edge.min()
    if edge.max() > 0:
        edge = edge / edge.max()
    
    # Aplicar blur
    if blur_radius > 0:
        edge = ndimage.gaussian_filter(edge, sigma=blur_radius)
    
    # Aplicar strength (1.0 = neutral, >1.0 = más claro en bordes)
    edge = np.clip(edge * strength, 0.0, 1.0)
    
    return edge

def apply_custom_cavity_map(albedo_np, cavity_np, mode="Multiply (darken)", strength=0.5):
    """Apply custom cavity map to albedo with selected blend mode"""
    if cavity_np is None:
        return albedo_np
    
    if cavity_np.ndim == 2:
        cavity_np = cavity_np[..., np.newaxis]
    
    albedo_rgb = albedo_np[..., :3]
    
    # Apply blend mode
    if mode == "Multiply (darken)":
        result_rgb = blend_multiply(albedo_rgb, cavity_np, strength)
    elif mode == "Overlay":
        result_rgb = blend_overlay(albedo_rgb, cavity_np, strength)
    elif mode == "Soft Light":
        result_rgb = blend_soft_light(albedo_rgb, cavity_np, strength)
    else:
        result_rgb = albedo_rgb
    
    return np.dstack((result_rgb, albedo_np[..., 3]))

def apply_custom_edge_map(albedo_np, edge_np, mode="Add (brighten)", strength=0.3):
    """Apply custom edge/highlight map to albedo with selected blend mode"""
    if edge_np is None:
        return albedo_np
    
    if edge_np.ndim == 2:
        edge_np = edge_np[..., np.newaxis]
    
    albedo_rgb = albedo_np[..., :3]
    
    # Apply blend mode
    if mode == "Add (brighten)":
        result_rgb = blend_add(albedo_rgb, edge_np, strength)
    elif mode == "Screen":
        result_rgb = blend_screen(albedo_rgb, edge_np, strength)
    elif mode == "Overlay":
        result_rgb = blend_overlay(albedo_rgb, edge_np, strength)
    else:
        result_rgb = albedo_rgb
    
    return np.dstack((result_rgb, albedo_np[..., 3]))

# ---------- Other Utility Functions ----------
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

def darken_metallic_np(metallic_np, darken_strength=1.0):
    """Darkens a metallic texture using a multiplication factor."""
    if metallic_np.ndim == 2:
        metallic_np = metallic_np[..., np.newaxis]
    
    out_metallic = metallic_np * (1.0 - darken_strength)
    
    return out_metallic.squeeze()

def apply_base_to_metallic_np(metallic_np, base_np, blend_strength=1.0):
    """
    Aplica un mapa base (como albedo) al metallic para modular su intensidad.
    Útil para crear variaciones basadas en el color o detalle del albedo.
    SIEMPRE retorna grayscale (2D array).
    """
    if metallic_np.ndim == 2:
        metallic_np = metallic_np[..., np.newaxis]
    
    if base_np.ndim == 3 and base_np.shape[2] > 1:
        # Convertir a grayscale si es RGB/RGBA
        base_gray = np.mean(base_np[..., :3], axis=2, keepdims=True)
    else:
        base_gray = base_np if base_np.ndim == 3 else base_np[..., np.newaxis]
    
    # Mezclar el metallic con el mapa base
    modulated = metallic_np * (base_gray * blend_strength + (1.0 - blend_strength))
    
    # CRÍTICO: Siempre retornar como 2D (grayscale)
    if modulated.ndim == 3:
        modulated = modulated.squeeze()
    
    return modulated

# ---------- GUI ----------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skyrim PBR Converter")
        if not QFontDatabase().families() and os.name == 'nt':
            # Some restricted Windows sessions cannot enumerate installed fonts.
            font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'segoeui.ttf'
            if font_path.is_file():
                QFontDatabase.addApplicationFont(str(font_path))
        font = QFont("Segoe UI")
        font.setPointSizeF(9.5)
        self.setFont(font)
        self.setMinimumSize(620, 580)
        self._compact = None
        self._last_preview = None
        self.preview_size = (512, 512)

        self.albedo = None
        self.ao = None
        self.normal = None
        self.roughness = None
        self.mask = None
        self.metallic_diffuse = None
        self.metallic_other = None
        self.metallic_base = None
        self.height = None
        self.cavity_map = None
        self.edge_map = None

        self.albedo_preview_np = None
        self.ao_preview_np = None
        self.normal_preview_np = None
        self.roughness_preview_np = None
        self.metallic_diffuse_preview_np = None
        self.metallic_other_preview_np = None
        self.metallic_base_preview_np = None
        self.mask_preview_np = None
        self.height_preview_np = None
        self.cavity_map_preview_np = None
        self.edge_map_preview_np = None

        self.output_folder_path = None
        self.nif_meshes = []
        self.nif_path = None
        self.nif_roots = []
        self.extra_texture_root = None
        self.viewport_3d = None
        self.cubemap_folder = None
        self.cubemap_faces = None
        self.active_cubemap_path = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(140)
        self._preview_timer.timeout.connect(self._update_3d_preview)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 18, 20, 18)
        main_layout.setSpacing(16)
        header = QHBoxLayout()
        heading = QVBoxLayout()
        title = QLabel("SKYRIM  /  Texture Studio")
        title.setObjectName("appTitle")
        subtitle = QLabel("Prepare materials, refine surface details, and export DDS textures.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitle")
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading, 1)
        badge = QLabel("PBR → DDS")
        badge.setObjectName("badge")
        header.addWidget(badge)
        main_layout.addLayout(header)
        self.workspace_splitter = QSplitter(Qt.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.workspace_splitter, 1)

        # Left side: Preview widget
        preview_panel = QFrame()
        preview_panel.setObjectName("previewPanel")
        left_layout = QVBoxLayout(preview_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        preview_heading = QLabel("PREVIEW")
        preview_heading.setObjectName("sectionTitle")
        preview_header = QHBoxLayout()
        preview_header.addWidget(preview_heading, 1)
        self.preview_mode_selector = QComboBox()
        self.preview_mode_selector.addItems(["2D", "3D"])
        self.preview_mode_selector.setToolTip("Switch between texture inspection and an interactive NIF material preview. Your texture settings are preserved.")
        self.preview_mode_selector.currentIndexChanged.connect(self._set_preview_mode)
        preview_header.addWidget(self.preview_mode_selector)
        left_layout.addLayout(preview_header)
        self.preview_label = QLabel("Load a texture to get started")
        self.preview_label.setMinimumSize(180, 80)
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview_label.setObjectName("previewCanvas")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setScaledContents(False)  # Don't stretch, we'll handle scaling
        self.preview_stack = QStackedWidget()
        self.preview_stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview_stack.setMinimumSize(180, 80)
        self.preview_stack.addWidget(self.preview_label)
        left_layout.addWidget(self.preview_stack, 1)

        preview_selector_layout = QHBoxLayout()
        preview_selector_layout.addWidget(QLabel("View:"))
        self.preview_selector = QComboBox()
        self.preview_selector.addItems([
            "Diffuse Output", 
            "Diffuse with Opacity",
            "Normal Output (RGB + Alpha)", 
            "Normal Alpha Only (Specular)", 
            "Metallic Output"
        ])
        self.preview_selector.currentIndexChanged.connect(self.process_and_preview)
        preview_selector_layout.addWidget(self.preview_selector)
        left_layout.addLayout(preview_selector_layout)

        preview_hint = QLabel("Preview your adjustments here. Use the export buttons to save the results.")
        preview_hint.setWordWrap(True)
        preview_hint.setObjectName("subtitle")
        left_layout.addWidget(preview_hint)
        self.workspace_splitter.addWidget(preview_panel)

        # Right side: Tabbed interface
        controls_panel = QWidget()
        right_layout = QVBoxLayout(controls_panel)
        right_layout.setContentsMargins(0, 0, 8, 0)
        right_layout.setSpacing(14)
        right_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # ========== STANDARD TAB ==========
        standard_tab = QWidget()
        standard_layout = QVBoxLayout()
        
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
        self.ao_value_label = QLabel("100%")
        self.ao_slider.valueChanged.connect(lambda v: self.ao_value_label.setText(f"{v}%"))
        ao_slider_layout.addWidget(ao_slider_label)
        ao_slider_layout.addWidget(self.ao_slider)
        ao_slider_layout.addWidget(self.ao_value_label)
        diffuse_grid.addLayout(ao_slider_layout, 2, 0, 1, 4)
        
        self.metallic_diffuse_line = self.create_texture_row("Metallic (for darkening):", diffuse_grid, 3, self.load_metallic_diffuse, self.remove_metallic_diffuse)
        darken_layout = QHBoxLayout()
        darken_label = QLabel("Darken by Metallic:")
        self.darken_metallic_slider = QSlider(Qt.Horizontal)
        self.darken_metallic_slider.setMinimum(0)
        self.darken_metallic_slider.setMaximum(100)
        self.darken_metallic_slider.setValue(50)
        self.darken_metallic_slider.valueChanged.connect(self.process_and_preview)
        self.darken_metallic_value_label = QLabel("50%")
        self.darken_metallic_slider.valueChanged.connect(lambda v: self.darken_metallic_value_label.setText(f"{v}%"))
        darken_layout.addWidget(darken_label)
        darken_layout.addWidget(self.darken_metallic_slider)
        darken_layout.addWidget(self.darken_metallic_value_label)
        diffuse_grid.addLayout(darken_layout, 4, 0, 1, 4)

        self.mask_line = self.create_texture_row("Opacity Mask:", diffuse_grid, 5, self.load_mask, self.remove_mask)
        opacity_layout = QHBoxLayout()
        opacity_label = QLabel("Opacity Strength:")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.process_and_preview)
        self.opacity_value_label = QLabel("100%")
        self.opacity_slider.valueChanged.connect(lambda v: self.opacity_value_label.setText(f"{v}%"))
        opacity_layout.addWidget(opacity_label)
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_value_label)
        diffuse_grid.addLayout(opacity_layout, 6, 0, 1, 4)

        standard_layout.addWidget(diffuse_group_box)
        
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
        
        # Auto-detect checkbox
        self.normal_autodetect_checkbox = QCheckBox("Detect from filename")
        self.normal_autodetect_checkbox.setChecked(False)
        self.normal_autodetect_checkbox.setToolTip("Recognizes normal_gl / normal_dx in filenames. For other names, select the format manually.")
        self.normal_autodetect_checkbox.stateChanged.connect(self.on_autodetect_changed)
        normal_type_layout.addWidget(self.normal_autodetect_checkbox)
        
        normal_grid.addLayout(normal_type_layout, 1, 0, 1, 4)

        # DDS Format selector for Normal
        normal_format_layout = QHBoxLayout()
        normal_format_layout.addWidget(QLabel("DDS Format:"))
        self.normal_format_selector = QComboBox()
        self.normal_format_selector.addItems(["BC7_UNORM (Recommended)", "BC3_UNORM (Legacy)"])
        normal_format_layout.addWidget(self.normal_format_selector)
        normal_grid.addLayout(normal_format_layout, 2, 0, 1, 4)

        self.roughness_line = self.create_texture_row("Roughness:", normal_grid, 3, self.load_roughness, self.remove_roughness)
        roughness_slider_layout = QHBoxLayout()
        roughness_slider_label = QLabel("Specular Strength:")
        self.roughness_slider = QSlider(Qt.Horizontal)
        self.roughness_slider.setMinimum(0)
        self.roughness_slider.setMaximum(100)
        self.roughness_slider.setValue(100)
        self.roughness_slider.valueChanged.connect(self.process_and_preview)
        self.roughness_value_label = QLabel("100%")
        self.roughness_slider.valueChanged.connect(lambda v: self.roughness_value_label.setText(f"{v}%"))
        roughness_slider_layout.addWidget(roughness_slider_label)
        roughness_slider_layout.addWidget(self.roughness_slider)
        roughness_slider_layout.addWidget(self.roughness_value_label)
        normal_grid.addLayout(roughness_slider_layout, 4, 0, 1, 4)

        # Min Specular slider (evita manchas negras)
        min_spec_slider_layout = QHBoxLayout()
        min_spec_label = QLabel("Min Specular (dark areas):")
        self.min_specular_slider = QSlider(Qt.Horizontal)
        self.min_specular_slider.setMinimum(0)
        self.min_specular_slider.setMaximum(50)
        self.min_specular_slider.setValue(15)  # 15% por defecto
        self.min_specular_slider.valueChanged.connect(self.process_and_preview)
        self.min_specular_value_label = QLabel("15%")
        self.min_specular_slider.valueChanged.connect(lambda v: self.min_specular_value_label.setText(f"{v}%"))
        min_spec_slider_layout.addWidget(min_spec_label)
        min_spec_slider_layout.addWidget(self.min_specular_slider)
        min_spec_slider_layout.addWidget(self.min_specular_value_label)
        normal_grid.addLayout(min_spec_slider_layout, 5, 0, 1, 4)

        # Max Specular slider
        max_spec_slider_layout = QHBoxLayout()
        max_spec_label = QLabel("Max Specular (bright areas):")
        self.max_specular_slider = QSlider(Qt.Horizontal)
        self.max_specular_slider.setMinimum(0)
        self.max_specular_slider.setMaximum(100)
        self.max_specular_slider.setValue(100)  # 100% por defecto
        self.max_specular_slider.valueChanged.connect(self.process_and_preview)
        self.max_specular_value_label = QLabel("100%")
        self.max_specular_slider.valueChanged.connect(lambda v: self.max_specular_value_label.setText(f"{v}%"))
        max_spec_slider_layout.addWidget(max_spec_label)
        max_spec_slider_layout.addWidget(self.max_specular_slider)
        max_spec_slider_layout.addWidget(self.max_specular_value_label)
        normal_grid.addLayout(max_spec_slider_layout, 6, 0, 1, 4)

        standard_layout.addWidget(normal_group_box)
        
        # Metallic Textures group box
        metallic_group_box = QGroupBox("Metallic Texture")
        metallic_grid = QGridLayout()
        metallic_group_box.setLayout(metallic_grid)
        
        self.metallic_other_line = self.create_texture_row("Metallic:", metallic_grid, 0, self.load_metallic_other, self.remove_metallic_other)
        
        metallic_darken_slider_layout = QHBoxLayout()
        metallic_darken_label = QLabel("Darken Metallic:")
        self.metallic_darken_slider = QSlider(Qt.Horizontal)
        self.metallic_darken_slider.setMinimum(0)
        self.metallic_darken_slider.setMaximum(100)
        self.metallic_darken_slider.setValue(0)
        self.metallic_darken_slider.valueChanged.connect(self.process_and_preview)
        self.metallic_darken_value_label = QLabel("0%")
        self.metallic_darken_slider.valueChanged.connect(lambda v: self.metallic_darken_value_label.setText(f"{v}%"))
        metallic_darken_slider_layout.addWidget(metallic_darken_label)
        metallic_darken_slider_layout.addWidget(self.metallic_darken_slider)
        metallic_darken_slider_layout.addWidget(self.metallic_darken_value_label)
        metallic_grid.addLayout(metallic_darken_slider_layout, 1, 0, 1, 4)
        
        self.metallic_base_line = self.create_texture_row("Base Map (Albedo/etc):", metallic_grid, 2, self.load_metallic_base, self.remove_metallic_base)
        
        metallic_base_slider_layout = QHBoxLayout()
        metallic_base_label = QLabel("Base Map Influence:")
        self.metallic_base_slider = QSlider(Qt.Horizontal)
        self.metallic_base_slider.setMinimum(0)
        self.metallic_base_slider.setMaximum(100)
        self.metallic_base_slider.setValue(0)
        self.metallic_base_slider.valueChanged.connect(self.process_and_preview)
        self.metallic_base_value_label = QLabel("0%")
        self.metallic_base_slider.valueChanged.connect(lambda v: self.metallic_base_value_label.setText(f"{v}%"))
        metallic_base_slider_layout.addWidget(metallic_base_label)
        metallic_base_slider_layout.addWidget(self.metallic_base_slider)
        metallic_base_slider_layout.addWidget(self.metallic_base_value_label)
        metallic_grid.addLayout(metallic_base_slider_layout, 3, 0, 1, 4)
        
        standard_layout.addWidget(metallic_group_box)
        standard_layout.addStretch(1)
        standard_tab.setLayout(standard_layout)
        
        # ========== ADVANCED TAB ==========
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout()
        
        # Height to Normal group box
        height_to_normal_group_box = QGroupBox("Height to Normal Conversion (Experimental)")
        height_to_normal_layout = QVBoxLayout()
        height_to_normal_group_box.setLayout(height_to_normal_layout)
        
        # Height map loader
        height_loader_layout = QHBoxLayout()
        height_loader_layout.addWidget(QLabel("Height Map:"))
        self.height_line = QLineEdit()
        self.height_line.setReadOnly(True)
        height_loader_layout.addWidget(self.height_line)
        height_browse_btn = QPushButton("Browse")
        height_browse_btn.clicked.connect(self.load_height)
        height_loader_layout.addWidget(height_browse_btn)
        height_clear_btn = QPushButton("Clear")
        height_clear_btn.clicked.connect(self.remove_height)
        height_loader_layout.addWidget(height_clear_btn)
        height_to_normal_layout.addLayout(height_loader_layout)
        
        # Height strength slider
        height_strength_layout = QHBoxLayout()
        height_strength_layout.addWidget(QLabel("Height Intensity:"))
        self.height_strength_slider = QSlider(Qt.Horizontal)
        self.height_strength_slider.setMinimum(1)
        self.height_strength_slider.setMaximum(100)
        self.height_strength_slider.setValue(10)
        self.height_strength_value_label = QLabel("10")
        self.height_strength_slider.valueChanged.connect(lambda v: self.height_strength_value_label.setText(f"{v}"))
        height_strength_layout.addWidget(self.height_strength_slider)
        height_strength_layout.addWidget(self.height_strength_value_label)
        height_to_normal_layout.addLayout(height_strength_layout)
        
        # Convert button
        convert_height_btn = QPushButton("Convert height to normal")
        convert_height_btn.clicked.connect(self.convert_height_to_normal)
        height_to_normal_layout.addWidget(convert_height_btn)
        
        height_info = QLabel("This feature converts a height/displacement map into a normal map.\nUseful when you only have height maps from Substance or other tools.")
        height_info.setWordWrap(True)
        height_info.setStyleSheet("color: #b6c2d1;")
        height_to_normal_layout.addWidget(height_info)
        
        advanced_layout.addWidget(height_to_normal_group_box)
        
        # Cavity/Edge Generator group box
        generator_group_box = QGroupBox("Generate Cavity/Edge Maps from Normal (Export to PNG)")
        generator_layout = QVBoxLayout()
        generator_group_box.setLayout(generator_layout)
        
        generator_info = QLabel("Generate cavity and edge maps from your normal map, save them as PNG, edit in Photoshop/GIMP, then load them back into the Custom Detail Maps section above.")
        generator_info.setWordWrap(True)
        generator_info.setStyleSheet("color: #b6c2d1;")
        generator_layout.addWidget(generator_info)
        
        # Cavity generation
        cavity_gen_label = QLabel("Cavity map")
        cavity_gen_label.setStyleSheet("font-weight: 600; color: #dce5ef;")
        generator_layout.addWidget(cavity_gen_label)
        
        cavity_gen_strength_layout = QHBoxLayout()
        cavity_gen_strength_layout.addWidget(QLabel("Cavity Intensity:"))
        self.cavity_gen_strength_slider = QSlider(Qt.Horizontal)
        self.cavity_gen_strength_slider.setMinimum(1)
        self.cavity_gen_strength_slider.setMaximum(100)
        self.cavity_gen_strength_slider.setValue(100)
        self.cavity_gen_strength_value_label = QLabel("100")
        self.cavity_gen_strength_slider.valueChanged.connect(lambda v: self.cavity_gen_strength_value_label.setText(f"{v}"))
        cavity_gen_strength_layout.addWidget(self.cavity_gen_strength_slider)
        cavity_gen_strength_layout.addWidget(self.cavity_gen_strength_value_label)
        generator_layout.addLayout(cavity_gen_strength_layout)
        
        cavity_gen_blur_layout = QHBoxLayout()
        cavity_gen_blur_layout.addWidget(QLabel("Cavity Blur:"))
        self.cavity_gen_blur_slider = QSlider(Qt.Horizontal)
        self.cavity_gen_blur_slider.setMinimum(0)
        self.cavity_gen_blur_slider.setMaximum(10)
        self.cavity_gen_blur_slider.setValue(2)
        self.cavity_gen_blur_value_label = QLabel("2")
        self.cavity_gen_blur_slider.valueChanged.connect(lambda v: self.cavity_gen_blur_value_label.setText(f"{v}"))
        cavity_gen_blur_layout.addWidget(self.cavity_gen_blur_slider)
        cavity_gen_blur_layout.addWidget(self.cavity_gen_blur_value_label)
        generator_layout.addLayout(cavity_gen_blur_layout)
        
        generate_cavity_btn = QPushButton("Generate cavity PNG")
        generate_cavity_btn.clicked.connect(self.generate_and_save_cavity)
        generator_layout.addWidget(generate_cavity_btn)
        
        # Edge generation
        edge_gen_label = QLabel("Edge / highlight map")
        edge_gen_label.setStyleSheet("font-weight: 600; color: #dce5ef;")
        generator_layout.addWidget(edge_gen_label)
        
        edge_gen_strength_layout = QHBoxLayout()
        edge_gen_strength_layout.addWidget(QLabel("Edge Intensity:"))
        self.edge_gen_strength_slider = QSlider(Qt.Horizontal)
        self.edge_gen_strength_slider.setMinimum(1)
        self.edge_gen_strength_slider.setMaximum(100)
        self.edge_gen_strength_slider.setValue(100)
        self.edge_gen_strength_value_label = QLabel("100")
        self.edge_gen_strength_slider.valueChanged.connect(lambda v: self.edge_gen_strength_value_label.setText(f"{v}"))
        edge_gen_strength_layout.addWidget(self.edge_gen_strength_slider)
        edge_gen_strength_layout.addWidget(self.edge_gen_strength_value_label)
        generator_layout.addLayout(edge_gen_strength_layout)
        
        edge_gen_blur_layout = QHBoxLayout()
        edge_gen_blur_layout.addWidget(QLabel("Edge Blur:"))
        self.edge_gen_blur_slider = QSlider(Qt.Horizontal)
        self.edge_gen_blur_slider.setMinimum(0)
        self.edge_gen_blur_slider.setMaximum(10)
        self.edge_gen_blur_slider.setValue(1)
        self.edge_gen_blur_value_label = QLabel("1")
        self.edge_gen_blur_slider.valueChanged.connect(lambda v: self.edge_gen_blur_value_label.setText(f"{v}"))
        edge_gen_blur_layout.addWidget(self.edge_gen_blur_slider)
        edge_gen_blur_layout.addWidget(self.edge_gen_blur_value_label)
        generator_layout.addLayout(edge_gen_blur_layout)
        
        generate_edge_btn = QPushButton("Generate edge PNG")
        generate_edge_btn.clicked.connect(self.generate_and_save_edge)
        generator_layout.addWidget(generate_edge_btn)
        
        advanced_layout.addWidget(generator_group_box)
        
        # Custom Detail Maps group box
        detail_maps_group_box = QGroupBox("Custom Detail Maps (Cavity/Curvature/Dirt)")
        detail_maps_layout = QVBoxLayout()
        detail_maps_group_box.setLayout(detail_maps_layout)
        
        # Cavity/Curvature map loader
        cavity_loader_layout = QHBoxLayout()
        cavity_loader_layout.addWidget(QLabel("Cavity/Curvature Map:"))
        self.cavity_map_line = QLineEdit()
        self.cavity_map_line.setReadOnly(True)
        cavity_loader_layout.addWidget(self.cavity_map_line)
        cavity_browse_btn = QPushButton("Browse")
        cavity_browse_btn.clicked.connect(self.load_cavity_map)
        cavity_loader_layout.addWidget(cavity_browse_btn)
        cavity_clear_btn = QPushButton("Clear")
        cavity_clear_btn.clicked.connect(self.remove_cavity_map)
        cavity_loader_layout.addWidget(cavity_clear_btn)
        detail_maps_layout.addLayout(cavity_loader_layout)
        
        # Cavity mode selector
        cavity_mode_layout = QHBoxLayout()
        cavity_mode_layout.addWidget(QLabel("Cavity Mode:"))
        self.cavity_mode_selector = QComboBox()
        self.cavity_mode_selector.addItems(["Multiply (darken)", "Overlay", "Soft Light"])
        self.cavity_mode_selector.currentIndexChanged.connect(self.process_and_preview)
        cavity_mode_layout.addWidget(self.cavity_mode_selector)
        detail_maps_layout.addLayout(cavity_mode_layout)
        
        # Cavity strength
        cavity_strength_layout = QHBoxLayout()
        cavity_strength_layout.addWidget(QLabel("Cavity Strength:"))
        self.cavity_strength_slider = QSlider(Qt.Horizontal)
        self.cavity_strength_slider.setMinimum(0)
        self.cavity_strength_slider.setMaximum(100)
        self.cavity_strength_slider.setValue(50)
        self.cavity_strength_slider.valueChanged.connect(self.process_and_preview)
        self.cavity_strength_value_label = QLabel("50%")
        self.cavity_strength_slider.valueChanged.connect(lambda v: self.cavity_strength_value_label.setText(f"{v}%"))
        cavity_strength_layout.addWidget(self.cavity_strength_slider)
        cavity_strength_layout.addWidget(self.cavity_strength_value_label)
        detail_maps_layout.addLayout(cavity_strength_layout)
        
        # Edge/Highlight map loader
        edge_loader_layout = QHBoxLayout()
        edge_loader_layout.addWidget(QLabel("Edge/Highlight Map:"))
        self.edge_map_line = QLineEdit()
        self.edge_map_line.setReadOnly(True)
        edge_loader_layout.addWidget(self.edge_map_line)
        edge_browse_btn = QPushButton("Browse")
        edge_browse_btn.clicked.connect(self.load_edge_map)
        edge_loader_layout.addWidget(edge_browse_btn)
        edge_clear_btn = QPushButton("Clear")
        edge_clear_btn.clicked.connect(self.remove_edge_map)
        edge_loader_layout.addWidget(edge_clear_btn)
        detail_maps_layout.addLayout(edge_loader_layout)
        
        # Edge mode selector
        edge_mode_layout = QHBoxLayout()
        edge_mode_layout.addWidget(QLabel("Edge Mode:"))
        self.edge_mode_selector = QComboBox()
        self.edge_mode_selector.addItems(["Add (brighten)", "Screen", "Overlay"])
        self.edge_mode_selector.currentIndexChanged.connect(self.process_and_preview)
        edge_mode_layout.addWidget(self.edge_mode_selector)
        detail_maps_layout.addLayout(edge_mode_layout)
        
        # Edge strength
        edge_strength_layout = QHBoxLayout()
        edge_strength_layout.addWidget(QLabel("Edge Strength:"))
        self.edge_strength_slider = QSlider(Qt.Horizontal)
        self.edge_strength_slider.setMinimum(0)
        self.edge_strength_slider.setMaximum(100)
        self.edge_strength_slider.setValue(30)
        self.edge_strength_slider.valueChanged.connect(self.process_and_preview)
        self.edge_strength_value_label = QLabel("30%")
        self.edge_strength_slider.valueChanged.connect(lambda v: self.edge_strength_value_label.setText(f"{v}%"))
        edge_strength_layout.addWidget(self.edge_strength_slider)
        edge_strength_layout.addWidget(self.edge_strength_value_label)
        detail_maps_layout.addLayout(edge_strength_layout)
        
        detail_info = QLabel("Load your own cavity, curvature, or detail maps from Substance Painter or other tools.\n• Cavity: Darkens areas (crevices, engravings, dirt accumulation)\n• Edge/Highlight: Brightens areas (raised edges, wear)\nUse greyscale maps for best results. Experiment with blend modes!")
        detail_info.setWordWrap(True)
        detail_info.setStyleSheet("color: #b6c2d1;")
        detail_maps_layout.addWidget(detail_info)
        
        advanced_layout.addWidget(detail_maps_group_box)
        
        advanced_layout.addStretch(1)
        advanced_tab.setLayout(advanced_layout)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self._scroll_panel(standard_tab), "Standard")
        self.tab_widget.addTab(self._scroll_panel(advanced_tab), "Advanced")
        self.tab_widget.addTab(self._scroll_panel(self._create_3d_controls()), "3D Preview")
        self.tab_widget.setMinimumHeight(300)
        
        right_layout.addWidget(self.tab_widget, 1)
        
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(12)
        right_layout.addWidget(export_group)
        # Output folder selection and prefix
        output_folder_layout = QHBoxLayout()
        output_folder_layout.addWidget(QLabel("Output Folder:"))
        self.output_folder_line = QLineEdit()
        self.output_folder_line.setReadOnly(True)
        self.output_folder_line.setMinimumWidth(0)
        self.output_folder_line.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.output_folder_line.setPlaceholderText("Choose an output folder")
        output_folder_layout.addWidget(self.output_folder_line)
        self.browse_folder_btn = QPushButton("Browse Folder...")
        self.browse_folder_btn.clicked.connect(self.select_output_folder)
        output_folder_layout.addWidget(self.browse_folder_btn)
        export_layout.addLayout(output_folder_layout)
        
        # Prefix field
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("File Prefix:"))
        self.prefix_line = QLineEdit()
        prefix_layout.addWidget(self.prefix_line)
        export_layout.addLayout(prefix_layout)

        # Resolution selectors group
        resolution_group_box = QGroupBox("Output Resolutions")
        resolution_grid = QGridLayout()
        resolution_group_box.setLayout(resolution_grid)
        
        # Diffuse resolution
        resolution_grid.addWidget(QLabel("Diffuse Resolution:"), 0, 0)
        self.diffuse_resolution_selector = QComboBox()
        self.diffuse_resolution_selector.addItems(["512x512", "1024x1024 (1K)", "2048x2048 (2K)", "4096x4096 (4K)", "8192x8192 (8K)"])
        self.diffuse_resolution_selector.setCurrentIndex(2)  # Default: 2K
        resolution_grid.addWidget(self.diffuse_resolution_selector, 0, 1)
        
        # Normal resolution
        resolution_grid.addWidget(QLabel("Normal Resolution:"), 1, 0)
        self.normal_resolution_selector = QComboBox()
        self.normal_resolution_selector.addItems(["512x512", "1024x1024 (1K)", "2048x2048 (2K)", "4096x4096 (4K)", "8192x8192 (8K)"])
        self.normal_resolution_selector.setCurrentIndex(2)  # Default: 2K
        resolution_grid.addWidget(self.normal_resolution_selector, 1, 1)
        
        # Metallic resolution
        resolution_grid.addWidget(QLabel("Metallic Resolution:"), 2, 0)
        self.metallic_resolution_selector = QComboBox()
        self.metallic_resolution_selector.addItems(["512x512", "1024x1024 (1K)", "2048x2048 (2K)", "4096x4096 (4K)", "8192x8192 (8K)"])
        self.metallic_resolution_selector.setCurrentIndex(1)  # Default: 1K
        resolution_grid.addWidget(self.metallic_resolution_selector, 2, 1)
        
        export_layout.addWidget(resolution_group_box)

        # Output buttons
        output_layout = QGridLayout()
        self.save_diffuse_btn = QPushButton("Export diffuse")
        self.save_diffuse_btn.clicked.connect(self.save_diffuse_output)
        output_layout.addWidget(self.save_diffuse_btn, 0, 0)
        self.save_diffuse_btn.setProperty("primary", True)
        
        self.save_normal_btn = QPushButton("Export normal")
        self.save_normal_btn.clicked.connect(self.save_normal_output)
        output_layout.addWidget(self.save_normal_btn, 1, 0)
        self.save_normal_btn.setProperty("primary", True)
        
        self.save_metallic_btn = QPushButton("Export metallic")
        self.save_metallic_btn.clicked.connect(self.save_metallic_output)
        output_layout.addWidget(self.save_metallic_btn, 2, 0)
        self.save_metallic_btn.setProperty("primary", True)

        export_layout.addLayout(output_layout)

        self.controls_scroll = self._scroll_panel(controls_panel)
        self.controls_scroll.setMinimumWidth(0)
        self.workspace_splitter.addWidget(self.controls_scroll)
        self.workspace_splitter.setStretchFactor(0, 5)
        self.workspace_splitter.setStretchFactor(1, 6)

        self.setLayout(main_layout)
        
        # Set initial window size
        self._apply_theme()
        self._configure_help()
        self.resize(1280, 860)
        QTimer.singleShot(0, self._update_responsive_layout)
        QTimer.singleShot(0, self._load_viewer_defaults)

    def _create_3d_controls(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        model = QGroupBox("NIF model")
        grid = QGridLayout(model)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        self.nif_file_line = QLineEdit()
        self.nif_file_line.setReadOnly(True)
        self.nif_file_line.setPlaceholderText("No NIF loaded")
        self.nif_file_line.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.nif_file_line.setToolTip("Read-only NIF path. The original model and textures are never modified by the viewer.")
        load_button = QPushButton("Load NIF")
        load_button.setToolTip("Load Skyrim SE or AE geometry in its bind pose and find textures relative to the mod folder.")
        load_button.clicked.connect(self._browse_nif)
        grid.addWidget(self.nif_file_line, 0, 0)
        grid.addWidget(load_button, 0, 1)
        self.mesh_target_selector = QComboBox()
        self.mesh_target_selector.addItem("All shapes", -1)
        self.mesh_target_selector.setToolTip("Choose which part receives the textures from the editor. Other parts keep their NIF materials.")
        self.mesh_target_selector.currentIndexChanged.connect(self._update_3d_preview)
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Apply edits to:"))
        target_row.addWidget(self.mesh_target_selector, 1)
        grid.addLayout(target_row, 1, 0, 1, 2)
        comparison_row = QHBoxLayout()
        self.use_processed_checkbox = QCheckBox("Use processed textures")
        self.use_processed_checkbox.setChecked(True)
        self.use_processed_checkbox.setToolTip("Apply loaded editor maps and settings to the selected shapes. Uncheck to compare with the NIF's original textures. Empty editor maps keep the original material.")
        self.use_processed_checkbox.toggled.connect(self._update_3d_preview)
        comparison_row.addWidget(self.use_processed_checkbox)
        self.isolate_checkbox = QCheckBox("Show selected shape only")
        self.isolate_checkbox.setToolTip("Hide other shapes to inspect the selected part. Select a specific shape first; All shapes displays the complete model.")
        self.isolate_checkbox.toggled.connect(self._update_3d_preview)
        comparison_row.addWidget(self.isolate_checkbox)
        grid.addLayout(comparison_row, 2, 0, 1, 2)
        material_button = QPushButton("Use material textures")
        material_button.setToolTip("Replace editor maps with diffuse, normal and reflection mask from the selected NIF material. Other input maps are cleared. With All shapes selected, the first material is used.")
        material_button.clicked.connect(self._use_nif_material)
        material_row = QHBoxLayout()
        material_row.addWidget(material_button)
        roots_button = QPushButton("Texture search folder")
        roots_button.setToolTip("Choose a mod or Data folder containing textures. It is used when textures cannot be found beside the loaded NIF.")
        roots_button.clicked.connect(self._choose_texture_root)
        material_row.addWidget(roots_button)
        grid.addLayout(material_row, 3, 0, 1, 2)
        self.nif_status_label = QLabel("Load a NIF to start. Drag to orbit, scroll to zoom.")
        self.nif_status_label.setWordWrap(True)
        self.nif_status_label.setObjectName("subtitle")
        grid.addWidget(self.nif_status_label, 4, 0, 1, 2)
        layout.addWidget(model)
        environment = QGroupBox("Reflections and lighting")
        env_layout = QVBoxLayout(environment)
        self.cubemap_folder_line = QLineEdit()
        self.cubemap_folder_line.setReadOnly(True)
        self.cubemap_folder_line.setPlaceholderText("Choose your cubemap folder")
        self.cubemap_folder_line.setToolTip("Folder of unpacked DDS cubemaps. Game assets remain external to this application.")
        env_layout.addWidget(self.cubemap_folder_line)
        buttons = QHBoxLayout()
        folder_button = QPushButton("Cubemap folder")
        folder_button.setToolTip("List DDS files from a folder so you can switch environments without reopening a file picker.")
        folder_button.clicked.connect(self._choose_cubemap_folder)
        browse_cube = QPushButton("Load DDS")
        browse_cube.setToolTip("Load a complete six-face DDS cubemap. Ordinary 2D textures are not accepted as cubemaps.")
        browse_cube.clicked.connect(self._browse_cubemap)
        buttons.addWidget(folder_button)
        buttons.addWidget(browse_cube)
        env_layout.addLayout(buttons)
        self.cubemap_selector = QComboBox()
        self.cubemap_selector.addItem("No cubemap", None)
        self.cubemap_selector.setToolTip("Choose the environment reflected by metallic surfaces. No cubemap disables reflections but keeps direct lighting and specular highlights.")
        self.cubemap_selector.currentIndexChanged.connect(self._select_cubemap)
        env_layout.addWidget(self.cubemap_selector)
        self.cubemap_status_label = QLabel("No cubemap loaded")
        self.cubemap_status_label.setObjectName("subtitle")
        env_layout.addWidget(self.cubemap_status_label)
        for attribute, label, maximum, value, description in [
            ('reflection_slider', 'Reflection intensity', 300, 100, 'Scale cubemap reflections using the reflection mask and the NIF environment scale. 0% disables reflections. This affects the preview only.'),
            ('environment_rotation_slider', 'Environment rotation', 360, 0, 'Rotate the cubemap around the model to inspect reflections from different directions. This does not change texture exports.'),
            ('preview_light_slider', 'Light intensity', 300, 100, 'Change direct preview lighting to inspect normal detail and specular highlights. Ambient fill remains enabled. This does not change texture exports.')]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, maximum)
            slider.setValue(value)
            slider.setToolTip(description)
            value_label = QLabel(str(value) + ('°' if 'rotation' in attribute else '%'))
            value_label.setMinimumWidth(38)
            slider.valueChanged.connect(lambda v, target=value_label, rotation='rotation' in attribute: target.setText(str(v) + ('°' if rotation else '%')))
            slider.valueChanged.connect(self._update_3d_lighting)
            setattr(self, attribute, slider)
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            env_layout.addLayout(row)
        reset = QPushButton("Reset camera")
        reset.setToolTip("Restore the initial orbit angle and zoom. You can also double-click the 3D viewport.")
        reset.clicked.connect(lambda: self.viewport_3d.reset_camera() if self.viewport_3d else None)
        env_layout.addWidget(reset)
        note = QLabel("Material preview only. Lighting and reflections approximate Skyrim; animations, skin shading, parallax and game shader mods are not simulated.")
        note.setWordWrap(True)
        note.setObjectName("subtitle")
        env_layout.addWidget(note)
        layout.addWidget(environment)
        layout.addStretch(1)
        return panel

    def _load_viewer_defaults(self):
        import json
        from nif_preview import resource_path
        try:
            path = resource_path('viewer_defaults.json')
            self.viewer_defaults = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
            folder = self.viewer_defaults.get('cubemap_folder', '')
            if folder and Path(folder).is_dir():
                self._set_cubemap_folder(folder)
        except (OSError, ValueError):
            self.viewer_defaults = {}

    def _set_preview_mode(self, index):
        if index == 1 and self.viewport_3d is None:
            from viewer3d import NifViewport
            self.viewport_3d = NifViewport()
            self.viewport_3d.failed.connect(self.nif_status_label.setText)
            self.preview_stack.addWidget(self.viewport_3d)
            self.viewport_3d.set_environment(self.cubemap_faces)
            self._update_3d_preview()
        self.preview_stack.setCurrentIndex(index)
        self.preview_selector.setEnabled(index == 0)
        if index == 1:
            self.tab_widget.setCurrentIndex(2)
            self._update_3d_lighting()
        QTimer.singleShot(0, self._refresh_preview)

    def _browse_nif(self):
        initial = str(self.nif_path or getattr(self, 'viewer_defaults', {}).get('nif_file', ''))
        path, _ = QFileDialog.getOpenFileName(self, "Load NIF", initial, "Skyrim models (*.nif)")
        if path:
            self._load_nif_file(path)

    def _load_nif_file(self, path):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            meshes = load_nif(path)
            roots = texture_roots(path)
            if self.extra_texture_root:
                roots.append(self.extra_texture_root)
            if self.cubemap_folder:
                roots.append(self.cubemap_folder.parent.parent)
            missing = load_mesh_textures(meshes, roots)
        except Exception as error:
            QMessageBox.critical(self, "NIF loading error", str(error))
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self.nif_meshes, self.nif_path, self.nif_roots = meshes, Path(path), roots
        self.nif_file_line.setText(str(path))
        self.mesh_target_selector.blockSignals(True)
        self.mesh_target_selector.clear()
        self.mesh_target_selector.addItem("All shapes", -1)
        for index, mesh in enumerate(meshes):
            self.mesh_target_selector.addItem(mesh.name, index)
        self.mesh_target_selector.blockSignals(False)
        count = sum(len(mesh.triangles) for mesh in meshes)
        self.nif_status_label.setText(f"{len(meshes)} shapes · {count:,} triangles" + (f" · {len(missing)} missing textures (add a search folder)" if missing else " · Original textures found"))
        self.nif_status_label.setToolTip("Missing textures: " + '; '.join(missing) if missing else "All referenced diffuse, normal and reflection-mask textures were found.")
        self.preview_mode_selector.setCurrentIndex(1)
        if self.viewport_3d:
            self.viewport_3d.reset_camera()
        self._match_nif_cubemap()
        self._update_3d_preview()
        return True

    def _choose_texture_root(self):
        path = QFileDialog.getExistingDirectory(self, "Choose a mod or Data folder containing textures")
        if path:
            self.extra_texture_root = Path(path)
            if self.nif_path:
                self._load_nif_file(self.nif_path)

    def _choose_cubemap_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose unpacked cubemap folder", str(self.cubemap_folder or ''))
        if folder:
            self._set_cubemap_folder(folder)

    def _set_cubemap_folder(self, folder):
        folder = Path(folder)
        try:
            import struct
            files = []
            for candidate in sorted(folder.rglob('*.dds'), key=lambda path: str(path).casefold()):
                with candidate.open('rb') as stream:
                    header = stream.read(128)
                if len(header) == 128 and header[:4] == b'DDS ':
                    height, width = struct.unpack_from('<II', header, 12)
                    caps2 = struct.unpack_from('<I', header, 112)[0]
                    if width == height and width > 0 and caps2 & 0xFE00 == 0xFE00:
                        files.append(candidate)
        except OSError as error:
            QMessageBox.critical(self, "Cubemap folder error", str(error))
            return
        self.cubemap_folder = folder
        self.cubemap_folder_line.setText(str(folder))
        current_path = self.cubemap_selector.currentData()
        self.cubemap_selector.blockSignals(True)
        self.cubemap_selector.clear()
        self.cubemap_selector.addItem("No cubemap", None)
        for path in files:
            self.cubemap_selector.addItem(str(path.relative_to(folder)), str(path))
        current_index = self.cubemap_selector.findData(current_path)
        self.cubemap_selector.setCurrentIndex(max(0, current_index))
        self.cubemap_selector.blockSignals(False)
        if self.nif_meshes:
            self._match_nif_cubemap()
        elif current_index < 0:
            self._select_cubemap()

    def _match_nif_cubemap(self):
        if not self.nif_meshes:
            return
        relative = self.nif_meshes[0].texture_paths.get(4, '')
        filename = relative.replace('\\', '/').rsplit('/', 1)[-1].casefold()
        for index in range(1, self.cubemap_selector.count()):
            if Path(self.cubemap_selector.itemData(index)).name.casefold() == filename:
                self.cubemap_selector.setCurrentIndex(index)
                return
        path = resolve_texture(relative, self.nif_roots)
        if path:
            self._load_cubemap_file(path)

    def _browse_cubemap(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load DDS cubemap", str(self.cubemap_folder or ''), "DDS cubemaps (*.dds)")
        if path:
            self._load_cubemap_file(path)

    def _select_cubemap(self, *args):
        path = self.cubemap_selector.currentData()
        self._load_cubemap_file(path)

    def _load_cubemap_file(self, path):
        try:
            faces = load_cubemap(path) if path else None
        except Exception as error:
            QMessageBox.critical(self, "Cubemap loading error", str(error))
            self.cubemap_selector.blockSignals(True)
            self.cubemap_selector.setCurrentIndex(max(0, self.cubemap_selector.findData(self.active_cubemap_path)))
            self.cubemap_selector.blockSignals(False)
            return False
        self.active_cubemap_path = str(path) if path else None
        self.cubemap_selector.blockSignals(True)
        index = self.cubemap_selector.findData(self.active_cubemap_path)
        if index < 0 and path:
            self.cubemap_selector.addItem(Path(path).name, str(path))
            index = self.cubemap_selector.count() - 1
        self.cubemap_selector.setCurrentIndex(max(0, index))
        self.cubemap_selector.blockSignals(False)
        self.cubemap_faces = faces
        self.cubemap_status_label.setText(f"{Path(path).name} · 6 faces · {faces[0].width} × {faces[0].height}" if path else "No cubemap loaded")
        self.cubemap_status_label.setToolTip(str(path) if path else "Reflections are disabled.")
        if self.viewport_3d:
            self.viewport_3d.set_environment(faces)
        return True

    def _use_nif_material(self):
        if not self.nif_meshes:
            QMessageBox.information(self, "Load a NIF first", "Load a NIF before importing its material textures.")
            return
        index = max(0, self.mesh_target_selector.currentData() or 0)
        mesh = self.nif_meshes[index]
        loaded = {}
        try:
            for slot, attribute in ((0, 'albedo'), (1, 'normal'), (5, 'metallic_other')):
                path = resolve_texture(mesh.texture_paths.get(slot, ''), self.nif_roots)
                if path:
                    with Image.open(path) as image:
                        loaded[attribute] = (image.convert('L' if slot == 5 else 'RGBA'), str(path))
        except Exception as error:
            QMessageBox.critical(self, "Material loading error", str(error))
            return
        if not loaded:
            QMessageBox.warning(self, "Textures not found", "Add the mod or Data folder containing this material's textures first.")
            return
        for attribute in ('albedo', 'ao', 'normal', 'roughness', 'mask', 'metallic_diffuse', 'metallic_other', 'metallic_base', 'height', 'cavity_map', 'edge_map'):
            setattr(self, attribute, None)
            setattr(self, attribute + '_preview_np', None)
            getattr(self, attribute + '_line').clear()
        for attribute, (image, path) in loaded.items():
            setattr(self, attribute, image)
            setattr(self, attribute + '_preview_np', pil_to_numpy(image.resize(self.preview_size, Image.LANCZOS)))
            getattr(self, attribute + '_line').setText(path)
        self.normal_type_selector.setCurrentText("DirectX")
        self.process_and_preview()

    def _material_overrides(self):
        overrides = {}
        if self.albedo_preview_np is not None:
            diffuse = self.albedo_preview_np.copy()
            if self.ao_preview_np is not None:
                diffuse = apply_ao_to_albedo_np(diffuse, self.ao_preview_np, self.ao_slider.value()/100.)
            if self.metallic_diffuse_preview_np is not None:
                diffuse = darken_albedo_metallic_np(diffuse, self.metallic_diffuse_preview_np, self.darken_metallic_slider.value()/100.)
            if self.cavity_map_preview_np is not None:
                diffuse = apply_custom_cavity_map(diffuse, self.cavity_map_preview_np, self.cavity_mode_selector.currentText(), self.cavity_strength_slider.value()/100.)
            if self.edge_map_preview_np is not None:
                diffuse = apply_custom_edge_map(diffuse, self.edge_map_preview_np, self.edge_mode_selector.currentText(), self.edge_strength_slider.value()/100.)
            if self.mask_preview_np is not None:
                diffuse = apply_opacity_mask_np(diffuse, self.mask_preview_np, self.opacity_slider.value()/100.)
            overrides[0] = numpy_to_pil(diffuse)
        if self.normal_preview_np is not None:
            normal = self.normal_preview_np.copy()
            if self.normal_type_selector.currentText() == "OpenGL":
                normal = convert_normal_to_directx_np(normal)
            normal = apply_roughness_to_normal_alpha(normal, self.roughness_preview_np,
                self.roughness_slider.value()/100., self.min_specular_slider.value()/100., self.max_specular_slider.value()/100.)
            overrides[1] = numpy_to_pil(normal)
        if self.metallic_other_preview_np is not None:
            metallic = darken_metallic_np(self.metallic_other_preview_np.copy(), self.metallic_darken_slider.value()/100.)
            if self.metallic_base_preview_np is not None:
                metallic = apply_base_to_metallic_np(metallic, self.metallic_base_preview_np, self.metallic_base_slider.value()/100.)
            overrides[5] = numpy_to_pil(metallic)
        return overrides

    def _update_3d_preview(self, *args):
        if self.viewport_3d is None:
            return
        target = self.mesh_target_selector.currentData()
        target = -1 if target is None else target
        overrides = self._material_overrides() if self.use_processed_checkbox.isChecked() else {}
        self.viewport_3d.isolate = target if self.isolate_checkbox.isChecked() else -1
        self.viewport_3d.set_scene(self.nif_meshes, overrides, target)

    def _update_3d_lighting(self, *args):
        if self.viewport_3d:
            self.viewport_3d.reflection_strength = self.reflection_slider.value()/100.
            self.viewport_3d.light_strength = self.preview_light_slider.value()/100.
            self.viewport_3d.environment_rotation = self.environment_rotation_slider.value()
            self.viewport_3d.update()

    @staticmethod
    def _scroll_panel(content):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _update_responsive_layout(self):
        compact = self.width() < 1100
        if compact != self._compact:
            self._compact = compact
            self.workspace_splitter.setOrientation(Qt.Vertical if compact else Qt.Horizontal)
            self.workspace_splitter.widget(0).setMinimumHeight(220 if compact else 0)
            self.workspace_splitter.setSizes([220, 460] if compact else [550, 650])
        self.controls_scroll.setMinimumHeight(200 if compact else 0)

    def _apply_theme(self):
        self.setStyleSheet("""
            QWidget { background: #171c24; color: #e5eaf1; font-family: 'Segoe UI'; font-size: 9.5pt; }
            QLabel { background: transparent; }
            QLabel#appTitle { font-size: 16pt; font-weight: 600; color: #f1f5f9; }
            QLabel#subtitle { color: #b6c2d1; font-size: 9pt; }
            QLabel#sectionTitle { color: #b6c2d1; font-size: 9pt; font-weight: 600; }
            QLabel#badge { color: #d4dfef; background: #293545; border: 1px solid #435269;
                          border-radius: 7px; padding: 8px 12px; font-size: 9pt; }
            QFrame#previewPanel { background: #202733; border: 1px solid #354050; border-radius: 12px; }
            QLabel#previewCanvas { background: #11161d; color: #aebdd0; border: 1px solid #354050; border-radius: 8px; }
            QGroupBox { background: #202733; border: 1px solid #3a4657; border-radius: 9px;
                        margin-top: 14px; padding: 16px 12px 14px 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; color: #e5eaf1; }
            QLineEdit, QComboBox { background: #141b25; border: 1px solid #49586d;
                                  border-radius: 6px; padding: 5px 8px; min-height: 17px; selection-background-color: #496383; }
            QLineEdit:focus, QComboBox:focus { border: 1px solid #91aacb; }
            QLineEdit:read-only { color: #c2cede; }
            QComboBox { padding-right: 26px; }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox QAbstractItemView { background: #263243; color: #f1f5f9; selection-background-color: #415976; padding: 5px; }
            QPushButton { background: #303d4f; color: #edf2f7; border: 1px solid #53647b;
                          border-radius: 6px; padding: 5px 10px; min-height: 17px; }
            QPushButton:hover { background: #40536b; border-color: #8da4c2; }
            QPushButton:pressed { background: #253246; }
            QPushButton:focus { border: 2px solid #abc2df; }
            QPushButton[primary="true"] { background: #415a77; border-color: #6d87a8; font-weight: 600; }
            QPushButton[primary="true"]:hover { background: #516e90; }
            QTabWidget::pane { border: 1px solid #354050; border-radius: 6px; padding: 6px; }
            QTabBar::tab { background: #232c39; color: #bfcbdc; padding: 8px 18px; margin-right: 4px;
                           border-top-left-radius: 6px; border-top-right-radius: 6px; }
            QTabBar::tab:selected { background: #3b4d65; color: #ffffff; }
            QTabBar::tab:hover { background: #34445a; }
            QSlider::groove:horizontal { height: 6px; background: #435168; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #91aacb; border-radius: 3px; }
            QSlider::handle:horizontal { background: #e1e9f5; border: 1px solid #91aacb;
                                        width: 16px; margin: -6px 0; border-radius: 8px; }
            QSlider::handle:horizontal:hover { background: #ffffff; }
            QCheckBox { spacing: 8px; background: transparent; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 1px solid #8296b1; background: #141b25; }
            QCheckBox::indicator:checked { background: #91aacb; border: 2px solid #e1e9f5; }
            QCheckBox::indicator:hover { border-color: #e1e9f5; }
            QScrollArea { border: none; background: #171c24; }
            QScrollBar:vertical { background: #1a222e; width: 12px; margin: 0; border-radius: 6px; }
            QScrollBar::handle:vertical { background: #657892; min-height: 32px; border-radius: 6px; }
            QScrollBar::handle:vertical:hover { background: #91aacb; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
            QSplitter::handle { background: #171c24; width: 10px; height: 10px; }
            QSplitter::handle:hover { background: #435875; }
            QToolTip { background: #edf2f8; color: #182332; border: 1px solid #9aaec8;
                       padding: 10px; font-size: 9.5pt; }
        """)

    def _configure_help(self):
        help_text = {
            'preview_selector': 'Choose the result to preview: diffuse, transparency, normal, specular alpha, or metallic. This selection does not change exported files.',
            'preview_label': 'Preview of the current settings. Normal preview brightness helps show the alpha channel; it does not simulate in-game lighting.',
            'ao_slider': 'Darken albedo using ambient occlusion. 0% disables AO; 100% applies the full map, emphasizing shadows in cracks and seams.',
            'darken_metallic_slider': 'Darken diffuse where the metallic map is bright. Higher values remove more base color in those areas to adjust their appearance alongside a cubemap.',
            'opacity_slider': 'Multiply the opacity mask: white is visible and black is transparent. 100% preserves the mask; 0% makes the entire texture transparent.',
            'normal_type_selector': 'Choose the source normal convention. OpenGL flips the green channel on export to produce DirectX. An incorrect choice can invert the appearance of surface relief.',
            'normal_autodetect_checkbox': 'Recognize OpenGL or DirectX from filenames such as normal_gl and normal_dx when loading a normal. If the name gives no hint, your manual selection is kept.',
            'normal_format_selector': 'BC7 offers higher quality and requires a compatible game engine. BC3 is the legacy alternative. Both preserve the alpha used for specular information.',
            'roughness_slider': 'Scale the specular alpha calculated from inverted roughness. 0% produces black alpha; 100% uses the full minimum/maximum range. Requires a roughness map.',
            'min_specular_slider': 'Specular level assigned to the roughest areas. Increasing it brightens their alpha and makes them shinier. Keep this value below the maximum.',
            'max_specular_slider': 'Specular level assigned to the smoothest areas. Lowering it limits their shine. Keep this value above the minimum.',
            'metallic_darken_slider': 'Reduce the separate metallic output: 0% preserves the map; 100% turns it black. This does not change the metallic map used to darken diffuse.',
            'metallic_base_slider': 'Modulate metallic using base-map brightness. 0% ignores the base map; 100% multiplies metallic by its grayscale brightness.',
            'height_strength_slider': 'Control the slope of normals generated from height. Higher values create stronger relief. Click the conversion button again after changing this setting.',
            'cavity_gen_strength_slider': 'Strength of cavities generated from the normal. Lower values move the map toward white; higher values emphasize dark recesses.',
            'cavity_gen_blur_slider': 'Smooth the generated cavity map. 0 preserves fine detail; higher values create wider, softer shading.',
            'edge_gen_strength_slider': 'Strength of edges detected from the normal. Higher values make edges brighter while the background stays black.',
            'edge_gen_blur_slider': 'Soften and widen the generated edge lines. 0 keeps sharper, more defined edges.',
            'cavity_mode_selector': 'Multiply darkens using the map. Overlay increases contrast. Soft Light adds gentler contrast. These modes affect diffuse.',
            'cavity_strength_slider': 'Amount of cavity blending applied to diffuse. 0% has no effect; 100% fully applies the selected blend mode.',
            'edge_mode_selector': 'Add adds brightness. Screen brightens while preserving more detail. Overlay adjusts contrast based on the map. These modes affect diffuse.',
            'edge_strength_slider': 'Amount of edge blending applied to diffuse. 0% has no effect; higher values strengthen the selected blend mode.',
            'output_folder_line': 'Folder where DDS textures and generated PNG detail maps are saved. Choose it with Browse Folder.',
            'browse_folder_btn': 'Choose where exported files will be saved.',
            'prefix_line': 'Base filename without an extension or folder path. For example, stone produces stone_d.dds, stone_n.dds, and stone_m.dds.',
            'diffuse_resolution_selector': 'Square size of the exported diffuse. Higher resolutions preserve more detail and use more memory. Enlarging an image cannot recover missing detail.',
            'normal_resolution_selector': 'Square size of the exported normal, independent of diffuse. Higher resolutions increase detail and memory use.',
            'metallic_resolution_selector': 'Square size of the exported metallic. A lower resolution may be enough for maps with smooth variations.',
            'save_diffuse_btn': 'Export albedo with AO, metallic darkening, detail maps, and opacity as BC3: prefix_d.dds. An existing file with this name is replaced after a successful conversion.',
            'save_normal_btn': 'Export the normal in DirectX using the selected format. Loaded roughness becomes specular alpha: prefix_n.dds. An existing file with this name is replaced.',
            'save_metallic_btn': 'Export grayscale metallic with darkening and base-map modulation as BC3: prefix_m.dds. An existing file with this name is replaced.',
        }
        textures = {
            'albedo': 'Base material color, combined with AO, metallic darkening, detail, and opacity to create diffuse.',
            'ao': 'Grayscale ambient occlusion: white preserves color and black darkens it. Adjust its effect with AO Strength.',
            'normal': 'Surface directions used to simulate relief. Check whether the source uses DirectX or OpenGL before exporting.',
            'roughness': 'Grayscale roughness: white is rough and black is smooth. It is inverted and remapped to the specular alpha of the normal.',
            'mask': 'Diffuse transparency mask: white is visible and black is transparent. It replaces the original alpha on export.',
            'metallic_diffuse': 'Metallic map used only to darken diffuse in white areas.',
            'metallic_other': 'Metallic map processed and exported as a separate file with the _m suffix.',
            'metallic_base': 'Image whose brightness modulates metallic output. Dark areas reduce its intensity.',
            'height': 'Grayscale height map used to generate a normal. Conversion replaces the loaded normal without changing the original image file.',
            'cavity_map': 'Cracks or dirt map blended into diffuse. With Multiply, black darkens and white preserves color.',
            'edge_map': 'Edges or wear map blended into diffuse. With Add and Screen, black is neutral and white brightens.',
        }
        for attribute, description in textures.items():
            help_text[attribute + '_line'] = description
        for attribute, description in help_text.items():
            widget = getattr(self, attribute)
            widget.setToolTip('<qt>' + description + '</qt>')
            widget.setToolTipDuration(20000)
            widget.setAccessibleDescription(description)
        button_help = {
            'Browse': 'Load an image for this setting. If the file is invalid, the previously loaded texture is preserved.',
            'Clear': 'Remove this texture from the current project. The original image file is not changed.',
            'Convert height to normal': 'Generate a normal from height using the selected intensity, replacing the loaded normal. Export it afterward to save the result.',
            'Generate cavity PNG': 'Generate prefix_cavity.png from the normal using the selected strength and blur. To apply it, load the result in Cavity/Curvature.',
            'Generate edge PNG': 'Generate prefix_edge.png from the normal. To apply it to diffuse, load the result as a custom edge map.',
        }
        for button in self.findChildren(QPushButton):
            if button.text() in button_help:
                button.setToolTip('<qt>' + button_help[button.text()] + '</qt>')
                button.setToolTipDuration(20000)
        for label in self.findChildren(QLabel):
            label.setWordWrap(True)
        for slider in self.findChildren(QSlider):
            slider.setMinimumWidth(65)
            slider.setAccessibleName(slider.toolTip())
        for edit in self.findChildren(QLineEdit):
            edit.setMinimumWidth(0)
        for combo in self.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(2 if combo is self.preview_mode_selector else 8)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        def help_for_labels(layout):
            controls = [layout.itemAt(i).widget() for i in range(layout.count())]
            descriptions = [widget.toolTip() for widget in controls
                            if widget is not None and not isinstance(widget, QLabel) and widget.toolTip()]
            if descriptions:
                for index, widget in enumerate(controls):
                    if isinstance(widget, QLabel) and not widget.toolTip():
                        description = descriptions[0]
                        if isinstance(layout, QGridLayout):
                            row, column, _, _ = layout.getItemPosition(index)
                            neighbor = layout.itemAtPosition(row, column + 1)
                            if neighbor is not None and neighbor.widget() is not None:
                                description = neighbor.widget().toolTip() or description
                        widget.setToolTip(description)
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.layout() is not None:
                    help_for_labels(item.layout())
                elif item.widget() is not None and item.widget().layout() is not None:
                    help_for_labels(item.widget().layout())
        for widget in self.findChildren(QGroupBox):
            help_for_labels(widget.layout())
        self.tab_widget.setTabToolTip(0, 'Load and adjust diffuse, normal, and metallic textures.')
        self.tab_widget.setTabToolTip(2, "Load a NIF, select a material target, and adjust cubemap reflections and preview lighting.")
        self.tab_widget.setTabToolTip(1, 'Convert height maps and generate or blend surface detail maps.')

    def resizeEvent(self, event):
        """Handle window resize to update preview"""
        super().resizeEvent(event)
        if hasattr(self, 'workspace_splitter'):
            self._update_responsive_layout()
            QTimer.singleShot(0, self._refresh_preview)

    def create_texture_row(self, name, grid_layout, row, load_func, remove_func):
        label = QLabel(name)
        file_line = QLineEdit()
        file_line.setReadOnly(True)
        file_line.setMinimumWidth(0)
        file_line.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        file_line.setPlaceholderText("No texture loaded")
        label.setWordWrap(True)
        label.setMinimumWidth(136)
        label.setMaximumWidth(150)
        browse_btn = QPushButton("Browse")
        
        browse_btn.clicked.connect(load_func)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(remove_func)
        
        grid_layout.setVerticalSpacing(10)
        grid_layout.setHorizontalSpacing(10)
        grid_layout.setColumnStretch(1, 1)
        grid_layout.addWidget(label, row, 0)
        grid_layout.addWidget(file_line, row, 1)
        grid_layout.addWidget(browse_btn, row, 2)
        grid_layout.addWidget(clear_btn, row, 3)
        
        return file_line

    def _load_texture(self, attribute, title, mode):
        path, _ = QFileDialog.getOpenFileName(self, title, "", "Images (*.png *.dds *.tga *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)")
        if not path:
            return
        try:
            with Image.open(path) as source:
                image = source.convert(mode) if mode else source.copy()
                if image.mode not in ('L', 'RGB', 'RGBA'):
                    image = image.convert('RGB')
            preview = pil_to_numpy(image.resize(self.preview_size, Image.LANCZOS))
        except Exception as error:
            QMessageBox.critical(self, "Image loading error", f"Could not load {path}:\n{error}")
            return
        setattr(self, attribute, image)
        setattr(self, attribute + '_preview_np', preview)
        getattr(self, attribute + '_line').setText(path)
        if attribute == 'normal' and self.normal_autodetect_checkbox.isChecked():
            self.on_autodetect_changed(Qt.Checked)
        self.process_and_preview()

    # ---------- Load / Remove Functions ----------
    def load_albedo(self):
        self._load_texture('albedo', 'Open albedo', 'RGBA')

    def remove_albedo(self):
        self.albedo = None
        self.albedo_preview_np = None
        self.albedo_line.clear()
        self.process_and_preview()

    def load_ao(self):
        self._load_texture('ao', 'Open ao', 'L')

    def remove_ao(self):
        self.ao = None
        self.ao_preview_np = None
        self.ao_line.clear()
        self.process_and_preview()
        
    def load_metallic_diffuse(self):
        self._load_texture('metallic_diffuse', 'Open metallic diffuse', 'L')

    def remove_metallic_diffuse(self):
        self.metallic_diffuse = None
        self.metallic_diffuse_preview_np = None
        self.metallic_diffuse_line.clear()
        self.process_and_preview()

    def load_metallic_other(self):
        self._load_texture('metallic_other', 'Open metallic other', 'L')

    def remove_metallic_other(self):
        self.metallic_other = None
        self.metallic_other_preview_np = None
        self.metallic_other_line.clear()
        self.process_and_preview()

    def load_metallic_base(self):
        self._load_texture('metallic_base', 'Open metallic base', None)

    def remove_metallic_base(self):
        self.metallic_base = None
        self.metallic_base_preview_np = None
        self.metallic_base_line.clear()
        self.process_and_preview()

    def load_normal(self):
        self._load_texture('normal', 'Open normal', 'RGBA')

    def remove_normal(self):
        self.normal = None
        self.normal_preview_np = None
        self.normal_line.clear()
        self.process_and_preview()
        
    def load_roughness(self):
        self._load_texture('roughness', 'Open roughness', 'L')

    def remove_roughness(self):
        self.roughness = None
        self.roughness_preview_np = None
        self.roughness_line.clear()
        self.process_and_preview()
        
    def load_mask(self):
        self._load_texture('mask', 'Open mask', 'L')

    def remove_mask(self):
        self.mask = None
        self.mask_preview_np = None
        self.mask_line.clear()
        self.process_and_preview()

    def load_height(self):
        self._load_texture('height', 'Open height', 'L')

    def remove_height(self):
        self.height = None
        self.height_preview_np = None
        self.height_line.clear()

    def load_cavity_map(self):
        self._load_texture('cavity_map', 'Open cavity map', 'L')

    def remove_cavity_map(self):
        self.cavity_map = None
        self.cavity_map_preview_np = None
        self.cavity_map_line.clear()
        self.process_and_preview()

    def load_edge_map(self):
        self._load_texture('edge_map', 'Open edge map', 'L')

    def remove_edge_map(self):
        self.edge_map = None
        self.edge_map_preview_np = None
        self.edge_map_line.clear()
        self.process_and_preview()

    def on_autodetect_changed(self, state):
        # Pixel averages cannot determine the normal-map convention reliably.
        self.normal_type_selector.setEnabled(True)
        if state == Qt.Checked and self.normal is not None:
            detected = self.detect_normal_type()
            if detected:
                self.normal_type_selector.setCurrentText(detected)
        self.process_and_preview()

    def detect_normal_type(self):
        filename = Path(self.normal_line.text()).stem.lower()
        if any(token in filename for token in ('opengl', 'normalgl', 'normal_gl', '_ogl')):
            return "OpenGL"
        if any(token in filename for token in ('directx', 'normaldx', 'normal_dx', '_dx')):
            return "DirectX"
        return None

    def convert_height_to_normal(self):
        """Convert height map to normal map using Sobel filter"""
        if self.height is None:
            QMessageBox.warning(self, "Error", "No height map loaded.")
            return
        
        # Get strength
        strength = self.height_strength_slider.value()
        
        # Convert height to numpy
        height_np = pil_to_numpy(self.height)
        if height_np.ndim == 3:
            height_np = height_np[..., 0]
        
        # Apply Sobel filter to get gradients
        from scipy import ndimage
        
        # Calculate gradients
        dx = ndimage.sobel(height_np, axis=1) * strength / 10.0
        dy = ndimage.sobel(height_np, axis=0) * strength / 10.0
        
        # Create normal map
        # Z component (pointing out of surface)
        dz = np.ones_like(dx)
        
        # Normalize
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        dx = dx / norm
        dy = dy / norm
        dz = dz / norm
        
        # Convert to 0-1 range (from -1 to 1)
        normal_r = (dx + 1.0) * 0.5
        normal_g = (dy + 1.0) * 0.5
        normal_b = (dz + 1.0) * 0.5
        
        # Create RGBA normal map
        normal_rgb = np.stack([normal_r, normal_g, normal_b], axis=-1)
        alpha = np.ones_like(normal_r)
        normal_rgba = np.dstack([normal_rgb, alpha])
        
        # Convert to PIL and set as current normal
        self.normal = numpy_to_pil(normal_rgba)
        self.normal_preview_np = pil_to_numpy(self.normal.resize(self.preview_size, Image.LANCZOS))
        self.normal_line.setText("Generated from Height Map")
        
        # Update preview
        self.process_and_preview()
        
        QMessageBox.information(self, "Success", "Height map converted to normal map!\nThe normal map has been loaded and is ready to use.")

    def generate_and_save_cavity(self):
        """Generate cavity map from normal and save as PNG"""
        if self.normal is None:
            QMessageBox.warning(self, "Error", "No normal map loaded. Load a normal map first.")
            return
        
        if not self.output_folder_path:
            QMessageBox.warning(self, "Error", "Please select an output folder first.")
            return
        
        # Get parameters
        strength = self.cavity_gen_strength_slider.value() / 100.0
        blur_radius = self.cavity_gen_blur_slider.value()
        
        # Prepare normal
        normal_np = pil_to_numpy(self.normal)
        if self.normal_type_selector.currentText() == "OpenGL":
            normal_np = convert_normal_to_directx_np(normal_np)
        
        # Generate cavity
        cavity_np = generate_cavity_from_normal(normal_np, strength=strength, blur_radius=blur_radius)
        
        # Convert to PIL
        cavity_pil = numpy_to_pil(cavity_np)
        cavity_pil = cavity_pil.convert('L')  # Grayscale
        
        # Get filename
        prefix = self.prefix_line.text().strip()
        if not prefix:
            prefix = "texture"
        
        filename = f"{prefix}_cavity.png"
        full_path = os.path.join(self.output_folder_path, filename)
        
        # Save
        cavity_pil.save(full_path)
        
        QMessageBox.information(self, "Success", f"Cavity map saved to:\n{full_path}\n\nYou can now:\n1. Edit it in Photoshop/GIMP\n2. Load it back using 'Browse' in Custom Detail Maps")

    def generate_and_save_edge(self):
        """Generate edge map from normal and save as PNG"""
        if self.normal is None:
            QMessageBox.warning(self, "Error", "No normal map loaded. Load a normal map first.")
            return
        
        if not self.output_folder_path:
            QMessageBox.warning(self, "Error", "Please select an output folder first.")
            return
        
        # Get parameters
        strength = self.edge_gen_strength_slider.value() / 100.0
        blur_radius = self.edge_gen_blur_slider.value()
        
        # Prepare normal
        normal_np = pil_to_numpy(self.normal)
        if self.normal_type_selector.currentText() == "OpenGL":
            normal_np = convert_normal_to_directx_np(normal_np)
        
        # Generate edge
        edge_np = generate_edge_from_normal(normal_np, strength=strength, blur_radius=blur_radius)
        
        # Convert to PIL
        edge_pil = numpy_to_pil(edge_np)
        edge_pil = edge_pil.convert('L')  # Grayscale
        
        # Get filename
        prefix = self.prefix_line.text().strip()
        if not prefix:
            prefix = "texture"
        
        filename = f"{prefix}_edge.png"
        full_path = os.path.join(self.output_folder_path, filename)
        
        # Save
        edge_pil.save(full_path)
        
        QMessageBox.information(self, "Success", f"Edge map saved to:\n{full_path}\n\nYou can now:\n1. Edit it in Photoshop/GIMP\n2. Load it back using 'Browse' in Custom Detail Maps")

    # ---------- Folder Selection ----------
    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.output_folder_path = folder_path
            self.output_folder_line.setText(folder_path)

    # ---------- Resolution Helper ----------
    def _get_resolution_from_selector(self, selector):
        """Extract resolution value from combo box text (e.g., '2048x2048 (2K)' -> 2048)"""
        text = selector.currentText()
        # Extract first number from string
        resolution_str = text.split('x')[0]
        return int(resolution_str)

    # ---------- Process and Preview ----------
    def process_and_preview(self):
        if self.nif_meshes:
            self._preview_timer.start()
        view_mode = self.preview_selector.currentText()
        preview_np = None
        
        if view_mode == "Diffuse Output":
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
            
            # Apply custom cavity map if loaded
            if self.cavity_map_preview_np is not None:
                cavity_strength = self.cavity_strength_slider.value() / 100.0
                cavity_mode = self.cavity_mode_selector.currentText()
                preview_np = apply_custom_cavity_map(preview_np, self.cavity_map_preview_np, cavity_mode, cavity_strength)
            
            # Apply custom edge map if loaded
            if self.edge_map_preview_np is not None:
                edge_strength = self.edge_strength_slider.value() / 100.0
                edge_mode = self.edge_mode_selector.currentText()
                preview_np = apply_custom_edge_map(preview_np, self.edge_map_preview_np, edge_mode, edge_strength)

        elif view_mode == "Diffuse with Opacity":
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
            
            # Apply custom cavity map if loaded
            if self.cavity_map_preview_np is not None:
                cavity_strength = self.cavity_strength_slider.value() / 100.0
                cavity_mode = self.cavity_mode_selector.currentText()
                preview_np = apply_custom_cavity_map(preview_np, self.cavity_map_preview_np, cavity_mode, cavity_strength)
            
            # Apply custom edge map if loaded
            if self.edge_map_preview_np is not None:
                edge_strength = self.edge_strength_slider.value() / 100.0
                edge_mode = self.edge_mode_selector.currentText()
                preview_np = apply_custom_edge_map(preview_np, self.edge_map_preview_np, edge_mode, edge_strength)

            if self.mask_preview_np is not None:
                opacity_strength = self.opacity_slider.value() / 100.0
                preview_np = apply_opacity_mask_np(preview_np, self.mask_preview_np, opacity_strength)
        
        elif view_mode == "Normal Output (RGB + Alpha)":
            if self.normal_preview_np is None:
                self.clear_preview()
                return
            preview_np = self.normal_preview_np.copy()

            if self.normal_type_selector.currentText() == "OpenGL":
                preview_np = convert_normal_to_directx_np(preview_np)
            
            roughness_strength = self.roughness_slider.value() / 100.0
            min_specular = self.min_specular_slider.value() / 100.0
            max_specular = self.max_specular_slider.value() / 100.0
            
            normal_with_alpha = apply_roughness_to_normal_alpha(
                preview_np, 
                self.roughness_preview_np,
                roughness_strength,
                min_specular,
                max_specular
            )
            
            # Visualizar el normal con el alpha aplicado como brillo
            preview_np = visualize_normal_with_alpha(normal_with_alpha)
            
        elif view_mode == "Normal Alpha Only (Specular)":
            if self.normal_preview_np is None:
                self.clear_preview()
                return
            
            normal_temp = self.normal_preview_np.copy()
            
            if self.normal_type_selector.currentText() == "OpenGL":
                normal_temp = convert_normal_to_directx_np(normal_temp)
            
            roughness_strength = self.roughness_slider.value() / 100.0
            min_specular = self.min_specular_slider.value() / 100.0
            max_specular = self.max_specular_slider.value() / 100.0
            
            normal_with_alpha = apply_roughness_to_normal_alpha(
                normal_temp, 
                self.roughness_preview_np,
                roughness_strength,
                min_specular,
                max_specular
            )
            
            # Mostrar el canal alpha como grayscale
            if normal_with_alpha.shape[2] == 4:
                alpha_channel = normal_with_alpha[..., 3]
                preview_np = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=-1)
            else:
                preview_np = normal_with_alpha[..., :3]

        elif view_mode == "Metallic Output":
            if self.metallic_other_preview_np is None:
                self.clear_preview()
                return
                
            preview_np = self.metallic_other_preview_np.copy()
            
            # Asegurar que sea 2D (grayscale)
            if preview_np.ndim == 3:
                preview_np = preview_np[..., 0]
            
            darken_strength = self.metallic_darken_slider.value() / 100.0
            preview_np = darken_metallic_np(preview_np, darken_strength)
            
            if self.metallic_base_preview_np is not None:
                base_influence = self.metallic_base_slider.value() / 100.0
                if base_influence > 0:
                    preview_np = apply_base_to_metallic_np(preview_np, self.metallic_base_preview_np, base_influence)
            
            # CRÍTICO: Asegurar que el resultado sea 2D para mostrarlo como grayscale
            if preview_np.ndim == 3:
                preview_np = preview_np[..., 0]

        if preview_np is not None:
            if preview_np.ndim == 2 or (preview_np.ndim == 3 and preview_np.shape[2] == 1):
                preview_np = np.dstack((preview_np, preview_np, preview_np))
            self.show_preview(numpy_to_pil(preview_np))
        else:
            self.clear_preview()

    # ---------- Texconv Helper Function ----------
    def _save_with_texconv(self, pil_image, file_path, dds_format):
        resource_root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
        texconv_path = resource_root / 'Resources' / 'texconv.exe'
        try:
            if not texconv_path.is_file():
                raise FileNotFoundError(f"Could not find {texconv_path}")
            destination = Path(file_path).resolve()
            # Stage on the same volume, then replace atomically after verification.
            with tempfile.TemporaryDirectory(prefix='skyrim_', dir=destination.parent) as staging:
                source = Path(staging) / 'texture.png'
                pil_image.save(source)
                command = [str(texconv_path), '-f', dds_format, '-ft', 'dds', str(source),
                           '-o', staging, '-y', '-nologo']
                result = subprocess.run(command, check=True, capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                generated = Path(staging) / 'texture.dds'
                with generated.open('rb') as stream:
                    if stream.read(4) != b'DDS ':
                        raise ValueError('Texconv did not produce a valid DDS file.')
                os.replace(generated, destination)
            QMessageBox.information(self, "Success", f"Texture saved as {dds_format} to {destination}")
            return True
        except subprocess.CalledProcessError as error:
            QMessageBox.critical(self, "Conversion error", error.stderr or error.stdout or str(error))
        except Exception as error:
            QMessageBox.critical(self, "Save error", str(error))
        return False

    # ---------- Output Functions ----------
    def _get_output_path(self, suffix):
        """Helper function to validate and get the full output path."""
        if not self.output_folder_path:
            QMessageBox.warning(self, "Error", "Please select an output folder first.")
            return None, None

        prefix = self.prefix_line.text().strip()
        if not prefix:
            QMessageBox.warning(self, "Error", "Please enter a file prefix.")
            return None, None
        
        if any(char in prefix for char in '<>:"/\\|?*') or prefix.endswith(('.', ' ')):
            QMessageBox.warning(self, "Error", "The prefix must be a valid filename without a folder path.")
            return None, None
        filename = f"{prefix}_{suffix}.dds"
        full_path = os.path.join(self.output_folder_path, filename)
        return full_path, prefix

    def save_diffuse_output(self):
        full_path, prefix = self._get_output_path("d")
        if not full_path:
            return

        if self.albedo is None:
            QMessageBox.warning(self, "Error", "No Albedo/Basecolor texture loaded.")
            return
        
        # Get target resolution
        target_resolution = self._get_resolution_from_selector(self.diffuse_resolution_selector)
        target_size = (target_resolution, target_resolution)
        
        # Resize albedo to target resolution
        albedo_resized = self.albedo.resize(target_size, Image.LANCZOS)
        albedo_out_np = pil_to_numpy(albedo_resized)
        
        if self.ao is not None:
            ao_full_np = pil_to_numpy(self.ao.resize(target_size, Image.LANCZOS))
            ao_strength = self.ao_slider.value() / 100.0
            albedo_out_np = apply_ao_to_albedo_np(albedo_out_np, ao_full_np, ao_strength)
        
        if self.metallic_diffuse is not None:
            metallic_full_np = pil_to_numpy(self.metallic_diffuse.resize(target_size, Image.LANCZOS))
            metallic_strength = self.darken_metallic_slider.value() / 100.0
            albedo_out_np = darken_albedo_metallic_np(albedo_out_np, metallic_full_np, metallic_strength)
        
        # Apply custom cavity map if loaded
        if self.cavity_map is not None:
            cavity_full_np = pil_to_numpy(self.cavity_map.resize(target_size, Image.LANCZOS))
            cavity_strength = self.cavity_strength_slider.value() / 100.0
            cavity_mode = self.cavity_mode_selector.currentText()
            albedo_out_np = apply_custom_cavity_map(albedo_out_np, cavity_full_np, cavity_mode, cavity_strength)
        
        # Apply custom edge map if loaded
        if self.edge_map is not None:
            edge_full_np = pil_to_numpy(self.edge_map.resize(target_size, Image.LANCZOS))
            edge_strength = self.edge_strength_slider.value() / 100.0
            edge_mode = self.edge_mode_selector.currentText()
            albedo_out_np = apply_custom_edge_map(albedo_out_np, edge_full_np, edge_mode, edge_strength)
            
        if self.mask is not None:
            mask_full_np = pil_to_numpy(self.mask.resize(target_size, Image.LANCZOS))
            opacity_strength = self.opacity_slider.value() / 100.0
            albedo_out_np = apply_opacity_mask_np(albedo_out_np, mask_full_np, opacity_strength)
            
        out_pil = numpy_to_pil(albedo_out_np)
        
        self._save_with_texconv(out_pil, full_path, "BC3_UNORM")

    def save_normal_output(self):
        full_path, prefix = self._get_output_path("n")
        if not full_path:
            return

        if self.normal is None:
            QMessageBox.warning(self, "Error", "No Normal texture loaded.")
            return
        
        # Get target resolution
        target_resolution = self._get_resolution_from_selector(self.normal_resolution_selector)
        target_size = (target_resolution, target_resolution)
        
        # Resize normal to target resolution
        normal_resized = self.normal.resize(target_size, Image.LANCZOS)
        normal_out_np = pil_to_numpy(normal_resized)
        
        if self.normal_type_selector.currentText() == "OpenGL":
            normal_out_np = convert_normal_to_directx_np(normal_out_np)
        
        # Aplicar roughness invertido al alpha con remapping
        roughness_strength = self.roughness_slider.value() / 100.0
        min_specular = self.min_specular_slider.value() / 100.0
        max_specular = self.max_specular_slider.value() / 100.0
        
        roughness_full_np = None
        if self.roughness is not None:
            roughness_full_np = pil_to_numpy(self.roughness.resize(target_size, Image.LANCZOS))
        
        normal_out_np = apply_roughness_to_normal_alpha(
            normal_out_np, 
            roughness_full_np,
            roughness_strength,
            min_specular,
            max_specular
        )
        
        out_pil = numpy_to_pil(normal_out_np)
        
        # Seleccionar formato basado en el selector
        format_text = self.normal_format_selector.currentText()
        if "BC7" in format_text:
            dds_format = "BC7_UNORM"
        else:
            dds_format = "BC3_UNORM"
        
        self._save_with_texconv(out_pil, full_path, dds_format)
            
    def save_metallic_output(self):
        full_path, prefix = self._get_output_path("m")
        if not full_path:
            return

        if self.metallic_other is None:
            QMessageBox.warning(self, "Error", "No Metallic texture loaded.")
            return

        # Get target resolution
        target_resolution = self._get_resolution_from_selector(self.metallic_resolution_selector)
        target_size = (target_resolution, target_resolution)
        
        # Resize metallic to target resolution
        metallic_resized = self.metallic_other.resize(target_size, Image.LANCZOS)
        metallic_out_np = pil_to_numpy(metallic_resized)
        
        # Asegurar que sea 2D (grayscale)
        if metallic_out_np.ndim == 3:
            metallic_out_np = metallic_out_np[..., 0]
        
        darken_strength = self.metallic_darken_slider.value() / 100.0
        metallic_out_np = darken_metallic_np(metallic_out_np, darken_strength)
        
        if self.metallic_base is not None:
            base_full_np = pil_to_numpy(self.metallic_base.resize(target_size, Image.LANCZOS))
            base_influence = self.metallic_base_slider.value() / 100.0
            if base_influence > 0:
                metallic_out_np = apply_base_to_metallic_np(metallic_out_np, base_full_np, base_influence)
        
        # CRÍTICO: Asegurar que el output sea grayscale
        if metallic_out_np.ndim == 3:
            metallic_out_np = metallic_out_np[..., 0]
        
        out_pil = numpy_to_pil(metallic_out_np)
        
        # Convertir a modo 'L' (grayscale) antes de guardar
        if out_pil.mode != 'L':
            out_pil = out_pil.convert('L')

        self._save_with_texconv(out_pil, full_path, "BC3_UNORM")

    # ---------- Preview Helper ----------
    def _refresh_preview(self):
        if self._last_preview is not None:
            self.show_preview(self._last_preview)

    def show_preview(self, pil_img):
        self._last_preview = pil_img
        if pil_img.mode == 'L' or pil_img.mode == 'P':
            pil_img = pil_img.convert('RGB')
        elif pil_img.mode not in ['RGB', 'RGBA']:
            pil_img = pil_img.convert('RGBA')
        
        # Get preview label size
        label_width = max(1, self.preview_label.width() - 4)
        label_height = max(1, self.preview_label.height() - 4)
        
        # Calculate scaled size maintaining aspect ratio
        img_width, img_height = pil_img.size
        aspect_ratio = img_width / img_height
        
        if label_width / label_height > aspect_ratio:
            # Label is wider, fit to height
            new_height = label_height
            new_width = max(1, int(label_height * aspect_ratio))
        else:
            # Label is taller, fit to width
            new_width = label_width
            new_height = max(1, int(label_width / aspect_ratio))
        
        # Resize image
        pil_img_resized = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to QImage
        if pil_img_resized.mode == 'RGB':
            arr = np.array(pil_img_resized)
            # Add alpha channel
            alpha = np.ones((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype) * 255
            arr = np.concatenate([arr, alpha], axis=2)
        else:
            arr = np.array(pil_img_resized)
        
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGBA8888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    def clear_preview(self):
        self._last_preview = None
        self.preview_label.setText("Load a texture to get started")
        self.preview_label.setPixmap(QPixmap())

# ---------- Run ----------
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    if len(sys.argv) == 3 and sys.argv[1] == '--preview-check':
        from preview_check import schedule_preview_check
        schedule_preview_check(app, w, sys.argv[2])
    sys.exit(app.exec_())
