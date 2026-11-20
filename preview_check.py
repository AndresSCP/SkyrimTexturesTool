"""Opt-in local rendering check for verifying source and frozen builds."""
import json
from pathlib import Path
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox


def schedule_preview_check(app, window, config_path):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    report_path = Path(config['report'])
    report = {'ok': False, 'errors': []}
    QMessageBox.critical = lambda *args: report['errors'].append(str(args[-1]))
    QMessageBox.warning = lambda *args: report['errors'].append(str(args[-1]))
    def finish():
        try:
            viewport = window.viewport_3d
            if viewport is None or viewport.renderer is None or viewport.error:
                raise RuntimeError(viewport.error if viewport else '3D viewport was not created.')
            image = viewport.grabFramebuffer()
            if image.isNull():
                raise RuntimeError('The 3D framebuffer is empty.')
            image.save(config['image'])
            report.update(ok=not report['errors'], shapes=len(window.nif_meshes),
                triangles=sum(len(mesh.triangles) for mesh in window.nif_meshes),
                cubemap=window.active_cubemap_path, renderer=viewport.gl_context.info['GL_RENDERER'])
        except Exception as error:
            report['errors'].append(str(error))
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        window.close()
        app.exit(0 if report['ok'] else 1)
    def load():
        try:
            if not window._load_nif_file(config['nif']):
                raise RuntimeError('Could not load the sample NIF.')
            if config.get('cubemap') and not window._load_cubemap_file(config['cubemap']):
                raise RuntimeError('Could not load the cubemap.')
        except Exception as error:
            report['errors'].append(str(error))
        QTimer.singleShot(1500, finish)
    QTimer.singleShot(200, load)
