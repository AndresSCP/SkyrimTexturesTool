import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QSlider, QComboBox, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QPoint
import skyrim_pbr_converter as converter

class ConverterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = converter.MainWindow()
        self.messages = patch.object(converter.QMessageBox, 'information').start()
        self.errors = patch.object(converter.QMessageBox, 'critical').start()
        self.addCleanup(patch.stopall)
        self.addCleanup(self.window.close)

    def test_responsive_layout_and_export_access(self):
        w = self.window
        w.show()
        for width, height in ((1400, 900), (680, 800), (620, 580)):
            w.resize(width, height)
            for _ in range(6):
                self.app.processEvents()
            self.assertEqual(w.width(), width)
            expected = Qt.Vertical if width < 1100 else Qt.Horizontal
            self.assertEqual(w.workspace_splitter.orientation(), expected)
            for index in range(3):
                w.tab_widget.setCurrentIndex(index)
                self.app.processEvents()
                scroll = w.tab_widget.widget(index)
                self.assertLessEqual(scroll.widget().width(), scroll.viewport().width())
            panel_layout = w.workspace_splitter.widget(0).layout()
            self.assertLess(panel_layout.itemAt(1).geometry().bottom(), panel_layout.itemAt(2).geometry().top())
            scroll = w.controls_scroll
            scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
            self.app.processEvents()
            position = w.save_metallic_btn.mapTo(scroll.viewport(), QPoint(0, 0))
            self.assertGreaterEqual(position.y(), 0)
            self.assertLessEqual(position.y() + w.save_metallic_btn.height(), scroll.viewport().height())
            scroll.verticalScrollBar().setValue(0)

    def test_all_settings_have_hover_help(self):
        for control_type in (QSlider, QComboBox, QLineEdit, QPushButton):
            for control in self.window.findChildren(control_type):
                self.assertTrue(control.toolTip(), type(control).__name__)

    def test_bad_image_preserves_loaded_texture(self):
        image = Image.new('RGBA', (4, 4))
        self.window.albedo = image
        with patch.object(converter.QFileDialog, 'getOpenFileName', return_value=(str(Path(__file__)), '')):
            self.window.load_albedo()
        self.assertIs(self.window.albedo, image)
        self.errors.assert_called_once()

    def test_autodetect_updates_when_loading_another_normal(self):
        self.window.normal_autodetect_checkbox.setChecked(True)
        with tempfile.TemporaryDirectory() as folder:
            for name, expected in [('normal_gl.png', 'OpenGL'), ('normal_dx.png', 'DirectX')]:
                path = Path(folder) / name
                Image.new('RGBA', (8, 8), (128, 128, 255, 255)).save(path)
                with patch.object(converter.QFileDialog, 'getOpenFileName', return_value=(str(path), '')):
                    self.window.load_normal()
                self.assertEqual(self.window.normal_type_selector.currentText(), expected)

    def test_zero_edge_strength_is_black(self):
        result = converter.generate_edge_from_normal(np.full((8, 8, 3), .5), strength=0)
        np.testing.assert_array_equal(result, 0)

    def test_failed_conversion_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'texture.dds'
            destination.write_bytes(b'previous texture')
            with patch.object(converter.subprocess, 'run'):
                self.assertFalse(self.window._save_with_texconv(Image.new('RGBA', (8, 8)), destination, 'BC3_UNORM'))
            self.assertEqual(destination.read_bytes(), b'previous texture')
            self.messages.assert_not_called()

    def test_real_dds_export(self):
        with tempfile.TemporaryDirectory() as folder:
            for fmt in ('BC3_UNORM', 'BC7_UNORM'):
                destination = Path(folder) / (fmt + '.dds')
                self.assertTrue(self.window._save_with_texconv(Image.new('RGBA', (8, 8), (128, 128, 255, 127)), destination, fmt))
                with Image.open(destination) as image:
                    self.assertEqual(image.size, (8, 8))
                    self.assertAlmostEqual(image.convert('RGBA').getpixel((0, 0))[3], 127, delta=4)
            self.errors.assert_not_called()

if __name__ == '__main__':
    unittest.main()
