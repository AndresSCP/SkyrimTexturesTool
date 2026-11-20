"""Interactive read-only material preview using the current Qt OpenGL context."""
import math
import numpy as np
import moderngl
from PIL import Image
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QSurfaceFormat, QPainter, QColor
from PyQt5.QtWidgets import QOpenGLWidget, QSizePolicy

VERTEX_SHADER = """#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
uniform mat4 mvp;
out vec3 position;
out vec3 normal;
out vec2 uv;
void main() {
    position = in_position;
    normal = in_normal;
    uv = in_uv;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""
FRAGMENT_SHADER = """#version 330
in vec3 position;
in vec3 normal;
in vec2 uv;
uniform sampler2D diffuse_map;
uniform sampler2D normal_map;
uniform sampler2D mask_map;
uniform samplerCube environment_map;
uniform vec3 eye;
uniform vec3 light_direction;
uniform float light_strength;
uniform float reflection_strength;
uniform float environment_rotation;
uniform float environment_lod;
uniform float glossiness;
uniform float specular_strength;
uniform float env_scale;
uniform bool has_environment;
out vec4 color;
void main() {
    vec4 base = texture(diffuse_map, uv);
    if (base.a < 0.02) discard;
    vec3 N = normalize(normal);
    if (!gl_FrontFacing) N = -N;
    vec4 normal_sample = texture(normal_map, uv);
    vec3 dpdx = dFdx(position), dpdy = dFdy(position);
    vec2 uvdx = dFdx(uv), uvdy = dFdy(uv);
    float det = uvdx.x * uvdy.y - uvdx.y * uvdy.x;
    if (abs(det) > 0.0000001) {
        vec3 T = normalize((dpdx * uvdy.y - dpdy * uvdx.y) / det);
        vec3 B = normalize((-dpdx * uvdy.x + dpdy * uvdx.x) / det);
        vec3 mapped = normal_sample.xyz * 2.0 - 1.0;
        N = normalize(mat3(T, B, N) * mapped);
    }
    vec3 V = normalize(eye - position);
    vec3 L = normalize(light_direction);
    float diffuse = max(dot(N, L), 0.0);
    vec3 linear_base = pow(max(base.rgb, vec3(0.0)), vec3(2.2));
    vec3 result = linear_base * (0.22 + 0.78 * diffuse * light_strength);
    vec3 halfway = normalize(V + L);
    float spec = pow(max(dot(N, halfway), 0.0), max(1.0, glossiness));
    result += vec3(spec * normal_sample.a * specular_strength * light_strength * 0.45);
    if (has_environment) {
        vec3 R = reflect(-V, N);
        float c = cos(environment_rotation), s = sin(environment_rotation);
        R.xy = mat2(c, -s, s, c) * R.xy;
        vec3 reflection = pow(max(textureLod(environment_map, R, environment_lod).rgb, vec3(0.0)), vec3(2.2));
        float mask = texture(mask_map, uv).r;
        float fresnel = 0.35 + 0.65 * pow(1.0 - max(dot(N,V), 0.0), 5.0);
        result += reflection * mask * reflection_strength * env_scale * fresnel;
    }
    color = vec4(pow(clamp(result, 0.0, 1.0), vec3(1.0 / 2.2)), base.a);
}
"""


def normalize(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def camera_matrix(eye, aspect):
    forward = normalize(-eye)
    side = normalize(np.cross(forward, np.array([0., 0., 1.])))
    up = np.cross(side, forward)
    view = np.eye(4, dtype='f4')
    view[:3, :3] = [side, up, -forward]
    view[:3, 3] = -view[:3, :3] @ eye
    f = 1.0 / math.tan(math.radians(42) / 2)
    near, far = .03, 100.
    projection = np.zeros((4, 4), dtype='f4')
    projection[0, 0], projection[1, 1] = f / max(aspect, .01), f
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = 2 * far * near / (near - far)
    projection[3, 2] = -1
    return projection @ view


class MaterialRenderer:
    def __init__(self, context):
        self.ctx = context
        self.program = context.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.drawables, self.resources = [], []
        self.environment = context.texture_cube((1, 1), 4, bytes([0, 0, 0, 255]) * 6)
        self.has_environment = False
        self.cube_levels = 0
        self.meshes = []

    def set_environment(self, faces):
        old = self.environment
        if faces:
            cube = self.ctx.texture_cube(faces[0].size, 4)
            for index, face in enumerate(faces):
                cube.write(index, face.convert('RGBA').tobytes())
            cube.build_mipmaps()
            cube.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.cube_levels = int(math.log2(faces[0].width))
        else:
            cube = self.ctx.texture_cube((1, 1), 4, bytes([0, 0, 0, 255]) * 6)
            self.cube_levels = 0
        self.environment, self.has_environment = cube, bool(faces)
        old.release()

    def set_scene(self, meshes, overrides=None, target=-1):
        for resource in reversed(self.resources):
            resource.release()
        self.drawables, self.resources = [], []
        self.meshes = meshes
        if not meshes:
            return
        points = np.concatenate([m.positions for m in meshes])
        center = (points.max(0) + points.min(0)) / 2
        scale = 2.0 / max(float(np.max(points.max(0) - points.min(0))), 1e-5)
        cache = {}
        defaults = {0: Image.new('RGBA', (1,1), (170,176,185,255)),
                    1: Image.new('RGBA', (1,1), (128,128,255,128)),
                    5: Image.new('RGBA', (1,1), (0,0,0,255))}
        for index, mesh in enumerate(meshes):
            interleaved = np.column_stack(((mesh.positions - center) * scale, mesh.normals, mesh.uvs)).astype('f4')
            vertices = self.ctx.buffer(interleaved.tobytes())
            triangles = self.ctx.buffer(mesh.triangles.astype('u2').tobytes())
            vao = self.ctx.vertex_array(self.program, [(vertices, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')],
                                        triangles, index_element_size=2)
            self.resources.extend([vertices, triangles, vao])
            textures = []
            for slot in (0, 1, 5):
                image = (overrides or {}).get(slot) if target in (-1, index) else None
                if image is None:
                    image = mesh.textures.get(slot, defaults[slot])
                key = id(image)
                if key not in cache:
                    texture = self.ctx.texture(image.size, 4, image.convert('RGBA').tobytes())
                    texture.build_mipmaps()
                    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                    texture.anisotropy = min(8., self.ctx.max_anisotropy)
                    cache[key] = texture
                    self.resources.append(texture)
                textures.append(cache[key])
            self.drawables.append((vao, textures, mesh, index))

    def render(self, width, height, framebuffer, yaw=20., pitch=8., distance=3.6,
               light_strength=1., reflection_strength=1., environment_rotation=0., isolate=-1):
        framebuffer.use()
        self.ctx.viewport = (0, 0, max(1, width), max(1, height))
        self.ctx.clear(.066, .086, .114, 1., depth=1.)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        yaw, pitch = math.radians(yaw), math.radians(pitch)
        eye = np.array([math.sin(yaw)*math.cos(pitch), -math.cos(yaw)*math.cos(pitch), math.sin(pitch)]) * distance
        program = self.program
        program['mvp'].write(camera_matrix(eye, width / max(1, height)).astype('f4').T.tobytes())
        program['eye'].value = tuple(eye)
        program['light_direction'].value = tuple(normalize(eye + np.array([-1., 0., 2.])))
        program['light_strength'].value = light_strength
        program['reflection_strength'].value = reflection_strength
        program['environment_rotation'].value = math.radians(environment_rotation)
        program['has_environment'].value = self.has_environment
        for name, unit in [('diffuse_map', 0), ('normal_map', 1), ('mask_map', 2), ('environment_map', 3)]:
            program[name].value = unit
        self.environment.use(3)
        for vao, textures, mesh, index in self.drawables:
            if isolate not in (-1, index):
                continue
            for unit, texture in enumerate(textures):
                texture.use(unit)
            program['env_scale'].value = max(0., mesh.env_scale)
            program['glossiness'].value = max(1., mesh.glossiness)
            program['specular_strength'].value = max(0., mesh.specular_strength)
            program['environment_lod'].value = min(float(self.cube_levels), math.sqrt(2. / (max(1., mesh.glossiness) + 2.)) * self.cube_levels)
            vao.render(moderngl.TRIANGLES)
        self.ctx.finish()

    def release(self):
        for resource in reversed(self.resources):
            resource.release()
        self.resources, self.drawables = [], []
        self.environment.release()
        self.program.release()


class NifViewport(QOpenGLWidget):
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)
        self.setMinimumSize(180, 80)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setFocusPolicy(Qt.StrongFocus)
        self.renderer, self.gl_context = None, None
        self.meshes, self.overrides, self.target, self.faces = [], {}, -1, None
        self.scene_dirty = self.cube_dirty = True
        self.yaw, self.pitch, self.distance = 20., 8., 3.6
        self.light_strength = self.reflection_strength = 1.
        self.environment_rotation = 0.
        self.isolate = -1
        self.error = ''
        self._mouse = None
        QTimer.singleShot(1500, self._check_context)
        self.setToolTip('Drag to orbit. Scroll to zoom. Double-click to reset the camera. Changes affect the preview only.')

    def _check_context(self):
        if self.isVisible() and not self.isValid():
            self.error = '3D preview is unavailable. OpenGL 3.3 is required; the 2D editor remains available.'
            self.failed.emit(self.error)

    def initializeGL(self):
        try:
            self.gl_context = moderngl.create_context(require=330)
            self.renderer = MaterialRenderer(self.gl_context)
            self.gl_context.extra = None
            self.gl_context_cleanup = super().context()
            self.gl_context_cleanup.aboutToBeDestroyed.connect(self._release)
        except Exception as error:
            self.error = '3D preview requires OpenGL 3.3. ' + str(error)
            self.failed.emit(self.error)

    def paintGL(self):
        if not self.renderer:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor('#11161d'))
            painter.setPen(QColor('#c2cede'))
            painter.drawText(self.rect().adjusted(16,16,-16,-16), Qt.AlignCenter | Qt.TextWordWrap,
                             self.error or 'Load a NIF to preview its materials.')
            painter.end()
            return
        try:
            if self.scene_dirty:
                self.renderer.set_scene(self.meshes, self.overrides, self.target)
                self.scene_dirty = False
            if self.cube_dirty:
                self.renderer.set_environment(self.faces)
                self.cube_dirty = False
            framebuffer = self.gl_context.detect_framebuffer(self.defaultFramebufferObject())
            ratio = self.devicePixelRatioF()
            self.renderer.render(int(self.width()*ratio), int(self.height()*ratio), framebuffer,
                self.yaw, self.pitch, self.distance, self.light_strength, self.reflection_strength,
                self.environment_rotation, self.isolate)
            if not self.meshes:
                painter = QPainter(self)
                painter.setPen(QColor('#c2cede'))
                painter.drawText(self.rect(), Qt.AlignCenter, 'Load a NIF to preview its materials.')
                painter.end()
        except Exception as error:
            self.error = 'Could not render the 3D preview: ' + str(error)
            self.failed.emit(self.error)

    def set_scene(self, meshes, overrides=None, target=-1):
        self.meshes, self.overrides, self.target = meshes, overrides or {}, target
        self.scene_dirty = True
        self.update()

    def set_environment(self, faces):
        self.faces = faces
        self.cube_dirty = True
        self.update()

    def reset_camera(self):
        self.yaw, self.pitch, self.distance = 20., 8., 3.6
        self.update()

    def mousePressEvent(self, event):
        self._mouse = event.pos()
        self.setFocus()

    def mouseMoveEvent(self, event):
        if self._mouse is not None and event.buttons() & Qt.LeftButton:
            delta = event.pos() - self._mouse
            self.yaw += delta.x() * .5
            self.pitch = max(-85., min(85., self.pitch + delta.y() * .5))
            self._mouse = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self._mouse = None

    def wheelEvent(self, event):
        self.distance = max(1.3, min(20., self.distance * math.exp(-event.angleDelta().y() / 1200.)))
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.reset_camera()

    def _release(self):
        if self.renderer:
            self.makeCurrent()
            self.renderer.release()
            self.renderer = None
            self.doneCurrent()
