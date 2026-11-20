import io
import os
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from nif_preview import Mesh, load_cubemap, load_nif, load_mesh_textures, texture_roots, resolve_texture
from viewer3d import MaterialRenderer

SAMPLE = Path(os.environ['SKYRIM_TEST_NIF']) if os.environ.get('SKYRIM_TEST_NIF') else None
CUBES = Path(os.environ['SKYRIM_TEST_CUBEMAPS']) if os.environ.get('SKYRIM_TEST_CUBEMAPS') else None

class CubemapTests(unittest.TestCase):
    def test_six_faces_keep_their_order(self):
        colors = [(255,0,0,255),(0,255,0,255),(0,0,255,255),(255,255,0,255),(0,255,255,255),(255,0,255,255)]
        faces = []
        for color in colors:
            stream = io.BytesIO()
            Image.new('RGBA', (4,4), color).save(stream, format='DDS')
            faces.append(stream.getvalue())
        header = bytearray(faces[0][:128])
        struct.pack_into('<I', header, 112, 0xFE00)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cube.dds'
            path.write_bytes(header + b''.join(face[128:] for face in faces))
            decoded = load_cubemap(path)
            self.assertEqual([face.getpixel((0,0)) for face in decoded], colors)
            path.write_bytes(path.read_bytes()[:-8])
            with self.assertRaisesRegex(ValueError, 'truncated'):
                load_cubemap(path)

    def test_rejects_plain_2d_dds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'plain.dds'
            Image.new('RGBA', (4,4)).save(path)
            with self.assertRaisesRegex(ValueError, 'six faces'):
                load_cubemap(path)

    def test_texture_search_prevents_parent_paths(self):
        self.assertIsNone(resolve_texture('../outside.dds', [Path.cwd()]))
        self.assertIsNone(resolve_texture('C:/outside.dds', [Path.cwd()]))

    def test_rejects_unsupported_nif(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'broken.nif'
            path.write_bytes(b'not a nif')
            with self.assertRaises(ValueError):
                load_nif(path)

    @unittest.skipUnless(SAMPLE and SAMPLE.is_file(), 'Local sample NIF not available')
    def test_real_skyrim_model_and_materials(self):
        meshes = load_nif(SAMPLE)
        self.assertGreater(len(meshes), 0)
        self.assertFalse(load_mesh_textures(meshes, texture_roots(SAMPLE)))
        for mesh in meshes:
            self.assertTrue(np.isfinite(mesh.positions).all())
            self.assertLess(int(mesh.triangles.max()), len(mesh.positions))
            self.assertEqual(len(mesh.uvs), len(mesh.positions))
            self.assertEqual(set(mesh.textures), {0,1,5})

    @unittest.skipUnless(CUBES and CUBES.is_dir(), 'Local cubemaps not available')
    def test_unpacked_skyrim_cubemaps(self):
        count = 0
        for path in CUBES.rglob('*.dds'):
            header = path.read_bytes()[:128]
            if len(header) < 128 or struct.unpack_from('<I',header,112)[0] & 0xFE00 != 0xFE00:
                continue
            faces = load_cubemap(path)
            self.assertEqual(len(faces), 6)
            self.assertTrue(all(face.size == faces[0].size for face in faces))
            count += 1
        self.assertGreater(count, 0)

class RenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import moderngl
        try:
            cls.ctx = moderngl.create_standalone_context(require=330)
        except Exception as error:
            raise unittest.SkipTest(f'OpenGL 3.3 unavailable: {error}')

    @classmethod
    def tearDownClass(cls):
        cls.ctx.release()

    def setUp(self):
        self.renderer = MaterialRenderer(self.ctx)
        self.frame = self.ctx.simple_framebuffer((128,128))
        self.addCleanup(self.renderer.release)
        self.addCleanup(self.frame.release)

    def mesh(self, offset=0):
        positions = np.array([[-1,0,-1],[1,0,-1],[1,0,1],[-1,0,1]],dtype='f4')
        positions[:,0] += offset
        return Mesh('quad',positions,np.tile([0,-1,0],(4,1)).astype('f4'),
            np.array([[0,0],[1,0],[1,1],[0,1]],dtype='f4'),np.array([[0,1,2],[0,2,3]],dtype='u2'),
            textures={0:Image.new('RGBA',(2,2),(0,0,0,255)),1:Image.new('RGBA',(2,2),(128,128,255,0)),5:Image.new('RGBA',(2,2),'white')})

    def render(self, **kwargs):
        self.renderer.render(128,128,self.frame,yaw=0,pitch=0,**kwargs)
        return np.frombuffer(self.frame.read(components=3),dtype=np.uint8).reshape(128,128,3).copy()

    def test_cubemap_color_and_reflection_strength_change_the_material(self):
        self.renderer.set_scene([self.mesh()])
        self.renderer.set_environment([Image.new('RGBA',(4,4),'red') for _ in range(6)])
        red = self.render()[64,64]
        self.assertGreater(int(red[0]),50)
        self.assertLess(int(red[2]),5)
        disabled = self.render(reflection_strength=0)[64,64]
        self.assertLess(int(disabled.max()),5)
        self.renderer.set_environment([Image.new('RGBA',(4,4),'blue') for _ in range(6)])
        blue = self.render()[64,64]
        self.assertGreater(int(blue[2]),50)
        self.assertLess(int(blue[0]),5)

    def test_black_mask_blocks_reflections(self):
        mesh = self.mesh()
        mesh.textures[5] = Image.new('RGBA',(2,2),'black')
        self.renderer.set_scene([mesh])
        self.renderer.set_environment([Image.new('RGBA',(4,4),'white') for _ in range(6)])
        self.assertLess(int(self.render()[64,64].max()),5)

    def test_edits_only_affect_the_selected_shape(self):
        meshes = [self.mesh(-1.2),self.mesh(1.2)]
        self.renderer.set_scene(meshes)
        original = self.render()
        self.renderer.set_scene(meshes,{0:Image.new('RGBA',(2,2),'red')},target=0)
        edited = self.render()
        difference = np.abs(edited.astype(float)-original.astype(float))
        self.assertGreater(float(difference[:,:64].mean()),1)
        self.assertLess(float(difference[:,64:].mean()),.01)

if __name__ == '__main__':
    unittest.main()
