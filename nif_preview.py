"""Read-only Skyrim NIF geometry and DDS cubemap loading."""
import ctypes as ct
import io
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
import numpy as np
from PIL import Image


def resource_path(name):
    return Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent)) / 'Resources' / name


# Buffer layouts adapted from PyNifly V30.2.0 NiflyDefs.hpp (GPL-3.0).
# See Resources/licenses and THIRD_PARTY_NOTICES.md.
class NiShapeBuf(ct.Structure):
    _fields_ = [
        ('bufSize', ct.c_uint16),
        ('bufType', ct.c_uint16),
        ('ID', ct.c_uint32),
        ('nameID', ct.c_uint32),
        ('controllerID', ct.c_uint32),
        ('extraDataCount', ct.c_uint16),
        ('flags', ct.c_uint32),
        ('translation', (ct.c_float * 3)),
        ('rotation', ((ct.c_float * 3) * 3)),
        ('scale', ct.c_float),
        ('propertyCount', ct.c_uint16),
        ('collisionID', ct.c_uint32),
        ('hasVertices', ct.c_uint8),
        ('hasNormals', ct.c_uint8),
        ('hasVertexColors', ct.c_uint8),
        ('hasUV', ct.c_uint8),
        ('hasFullPrecision', ct.c_uint8),
        ('boundingSphereCenter', (ct.c_float * 3)),
        ('boundingSphereRadius', ct.c_float),
        ('vertexCount', ct.c_uint32),
        ('triangleCount', ct.c_uint32),
        ('skinInstanceID', ct.c_uint32),
        ('shaderPropertyID', ct.c_uint32),
        ('alphaPropertyID', ct.c_uint32),
        ('vertexDesc', ct.c_uint64),
    ]

class NiShaderBuf(ct.Structure):
    _fields_ = [
        ('bufSize', ct.c_uint16),
        ('bufType', ct.c_uint16),
        ('nameID', ct.c_uint32),
        ('bBSLightingShaderProperty', ct.c_char),
        ('bslspShaderType', ct.c_uint32),
        ('controllerID', ct.c_uint32),
        ('extraDataCount', ct.c_uint16),
        ('shaderFlags', ct.c_uint16),
        ('Shader_Type', ct.c_uint32),
        ('Shader_Flags_1', ct.c_uint32),
        ('Shader_Flags_2', ct.c_uint32),
        ('Env_Map_Scale', ct.c_float),
        ('numSF1', ct.c_uint32),
        ('numSF2', ct.c_uint32),
        ('UV_Offset_U', ct.c_float),
        ('UV_Offset_V', ct.c_float),
        ('UV_Scale_U', ct.c_float),
        ('UV_Scale_V', ct.c_float),
        ('textureSetID', ct.c_uint32),
        ('Emissive_Color', (ct.c_float * 4)),
        ('Emissive_Mult', ct.c_float),
        ('rootMaterialNameID', ct.c_uint32),
        ('textureClampMode', ct.c_uint32),
        ('Alpha', ct.c_float),
        ('Refraction_Str', ct.c_float),
        ('Glossiness', ct.c_float),
        ('specularColor', (ct.c_float * 3)),
        ('Spec_Str', ct.c_float),
        ('Soft_Lighting', ct.c_float),
        ('Rim_Light_Power', ct.c_float),
        ('subsurfaceRolloff', ct.c_float),
        ('rimlightPower2', ct.c_float),
        ('backlightPower', ct.c_float),
        ('grayscaleToPaletteScale', ct.c_float),
        ('fresnelPower', ct.c_float),
        ('wetnessSpecScale', ct.c_float),
        ('wetnessSpecPower', ct.c_float),
        ('wetnessMinVar', ct.c_float),
        ('wetnessEnvmapScale', ct.c_float),
        ('wetnessFresnelPower', ct.c_float),
        ('wetnessMetalness', ct.c_float),
        ('wetnessUnknown1', ct.c_float),
        ('wetnessUnknown2', ct.c_float),
        ('lumEmittance', ct.c_float),
        ('exposureOffset', ct.c_float),
        ('finalExposureMin', ct.c_float),
        ('finalExposureMax', ct.c_float),
        ('doTranslucency', ct.c_char),
        ('subsurfaceColor', (ct.c_float * 3)),
        ('transmissiveScale', ct.c_float),
        ('turbulence', ct.c_float),
        ('thickObject', ct.c_char),
        ('mixAlbedo', ct.c_char),
        ('hasTextureArrays', ct.c_char),
        ('numTextureArrays', ct.c_uint32),
        ('useSSR', ct.c_char),
        ('wetnessUseSSR', ct.c_char),
        ('skinTintColor', (ct.c_float * 3)),
        ('Skin_Tint_Alpha', ct.c_float),
        ('hairTintColor', (ct.c_float * 3)),
        ('maxPasses', ct.c_float),
        ('scale', ct.c_float),
        ('parallaxInnerLayerThickness', ct.c_float),
        ('parallaxRefractionScale', ct.c_float),
        ('parallaxInnerLayerTextureScale', (ct.c_float * 2)),
        ('parallaxEnvmapStrength', ct.c_float),
        ('sparkleParameters', (ct.c_float * 4)),
        ('eyeCubemapScale', ct.c_float),
        ('eyeLeftReflectionCenter', (ct.c_float * 3)),
        ('eyeRightReflectionCenter', (ct.c_float * 3)),
        ('sourceTexture', (ct.c_char * 256)),
        ('lightingInfluence', ct.c_char),
        ('envMapMinLOD', ct.c_char),
        ('falloffStartAngle', ct.c_float),
        ('falloffStopAngle', ct.c_float),
        ('falloffStartOpacity', ct.c_float),
        ('falloffStopOpacity', ct.c_float),
        ('refractionPower', ct.c_float),
        ('baseColor', (ct.c_float * 4)),
        ('baseColorScale', ct.c_float),
        ('softFalloffDepth', ct.c_float),
        ('greyscaleTexture', (ct.c_char * 256)),
        ('envMapTexture', (ct.c_char * 256)),
        ('normalTexture', (ct.c_char * 256)),
        ('envMaskTexture', (ct.c_char * 256)),
        ('envMapScale', ct.c_float),
        ('emittanceColor', (ct.c_float * 3)),
        ('emitGradientTexture', (ct.c_char * 256)),
        ('refractionStrength', ct.c_float),
        ('refractionFirePeriod', ct.c_uint32),
        ('parallaxMaxPasses', ct.c_float),
        ('parallaxScale', ct.c_float),
        ('emissiveColor', (ct.c_float * 4)),
    ]


@dataclass
class Mesh:
    name: str
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    triangles: np.ndarray
    texture_paths: dict = field(default_factory=dict)
    textures: dict = field(default_factory=dict)
    shader_type: int = 0
    shader_flags: int = 0
    env_scale: float = 1.0
    glossiness: float = 30.0
    specular_strength: float = 1.0


def _library():
    dll = ct.CDLL(str(resource_path('NiflyDLL.dll')))
    pointer, integer, floats = ct.c_void_p, ct.c_int, ct.POINTER(ct.c_float)
    signatures = {
        'load': ([ct.c_char_p], pointer), 'destroy': ([pointer], None),
        'getShapes': ([pointer, ct.POINTER(pointer), integer, integer], integer),
        'getShapeName': ([pointer, ct.c_char_p, integer], integer),
        'getVertsForShape': ([pointer, pointer, floats, integer, integer], integer),
        'getNormalsForShape': ([pointer, pointer, floats, integer, integer], integer),
        'getUVs': ([pointer, pointer, floats, integer, integer], integer),
        'getTriangles': ([pointer, pointer, ct.POINTER(ct.c_uint16), integer, integer], integer),
        'getShaderTextureSlot': ([pointer, pointer, integer, ct.c_char_p, integer], integer),
        'getNodeTransform': ([pointer, floats], None),
        'getBlockID': ([pointer, pointer], ct.c_uint32),
        'getBlock': ([pointer, ct.c_uint32, pointer], integer),
        'getShapeGlobalToSkin': ([pointer, pointer, floats], ct.c_bool),
        'getNodeParent': ([pointer, pointer], pointer),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(dll, name)
        function.argtypes, function.restype = arguments, result
    return dll


def _matrix(values):
    values = np.asarray(values, dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = values[3:12].reshape(3, 3) * values[12]
    matrix[:3, 3] = values[:3]
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError('The NIF contains an invalid transform.')
    return matrix


def resolve_texture(relative, roots):
    parts = PureWindowsPath(relative).parts
    if not parts or PureWindowsPath(relative).is_absolute() or '..' in parts:
        return None
    for root in roots:
        candidate = Path(root).joinpath(*parts)
        if candidate.is_file():
            return candidate
    return None


def texture_roots(nif_path, extra_roots=()):
    roots = []
    for parent in Path(nif_path).resolve().parents:
        if parent.name.lower() == 'meshes':
            roots.append(parent.parent)
            break
    roots.extend(Path(root) for root in extra_roots)
    return roots


def load_nif(path, extra_roots=()):
    """Return bind-pose meshes; never write to the source NIF or textures."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    # Reject unsupported games before handing the file to the native library.
    with path.open('rb') as stream:
        header = stream.readline(128)
        metadata = stream.read(17)
    if not header.startswith(b'Gamebryo File Format, Version 20.2.0.7') or len(metadata) < 17:
        raise ValueError('This viewer supports Skyrim SE/AE NIF files (20.2.0.7).')
    if struct.unpack_from('<I', metadata, 5)[0] != 12 or struct.unpack_from('<I', metadata, 13)[0] != 100:
        raise ValueError('This viewer supports Skyrim SE/AE NIF files only.')
    dll = _library()
    handle = dll.load(os.fsencode(str(path)))
    if not handle:
        raise ValueError('The NIF could not be read. It may be unsupported or damaged.')
    result = []
    try:
        count = dll.getShapes(handle, None, 0, 0)
        if not 0 < count <= 10000:
            raise ValueError('The NIF has no supported renderable shapes.')
        shapes = (ct.c_void_p * count)()
        dll.getShapes(handle, shapes, count, 0)
        for shape in shapes:
            def array(function, width, dtype=np.float32):
                size = function(handle, shape, None, 0, 0)
                if not 0 <= size <= 5000000:
                    raise ValueError('Invalid NIF geometry size.')
                data = np.zeros((size, width), dtype=dtype)
                pointer_type = ct.c_uint16 if dtype == np.uint16 else ct.c_float
                if size:
                    function(handle, shape, data.ctypes.data_as(ct.POINTER(pointer_type)), data.size, 0)
                return data
            positions = array(dll.getVertsForShape, 3)
            triangles = array(dll.getTriangles, 3, np.uint16)
            if not len(positions) or not len(triangles):
                continue
            if triangles.max() >= len(positions) or not np.isfinite(positions).all():
                raise ValueError('The NIF contains invalid vertex or triangle data.')
            normals = array(dll.getNormalsForShape, 3)
            if len(normals) != len(positions):
                normals = np.zeros_like(positions)
                a, b, c = positions[triangles[:, 0]], positions[triangles[:, 1]], positions[triangles[:, 2]]
                face_normals = np.cross(b - a, c - a)
                for column in range(3):
                    np.add.at(normals, triangles[:, column], face_normals)
            uvs = array(dll.getUVs, 2)
            if len(uvs) != len(positions):
                uvs = np.zeros((len(positions), 2), dtype=np.float32)
            transform = (ct.c_float * 13)()
            if dll.getShapeGlobalToSkin(handle, shape, transform):
                matrix = np.linalg.inv(_matrix(transform))
            else:
                dll.getNodeTransform(shape, transform)
                matrix = _matrix(transform)
                parent = dll.getNodeParent(handle, shape)
                seen = set()
                while parent and parent not in seen:
                    seen.add(parent)
                    # NiNode and NiShape share the NiAVObject transform accessor.
                    dll.getNodeTransform(parent, transform)
                    matrix = _matrix(transform) @ matrix
                    parent = dll.getNodeParent(handle, parent)
            positions = (positions @ matrix[:3, :3].T + matrix[:3, 3]).astype('f4')
            normals = (normals @ np.linalg.inv(matrix[:3, :3])).astype('f4')
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
            name = ct.create_string_buffer(2048)
            dll.getShapeName(shape, name, len(name))
            textures = {}
            for slot in (0, 1, 4, 5):
                buffer = ct.create_string_buffer(8192)
                dll.getShaderTextureSlot(handle, shape, slot, buffer, len(buffer))
                textures[slot] = buffer.value.decode('utf-8', errors='replace')
            shape_buffer = NiShapeBuf()
            shape_buffer.bufSize, shape_buffer.bufType = ct.sizeof(shape_buffer), 1
            if dll.getBlock(handle, dll.getBlockID(handle, shape), ct.byref(shape_buffer)) != 0:
                raise ValueError('Could not read NIF shape properties.')
            shader = NiShaderBuf()
            shader.bufSize, shader.bufType = ct.sizeof(shader), 28
            shader.Env_Map_Scale, shader.Glossiness, shader.Spec_Str = 1.0, 30.0, 1.0
            if shape_buffer.shaderPropertyID != 0xffffffff:
                if dll.getBlock(handle, shape_buffer.shaderPropertyID, ct.byref(shader)) != 0:
                    raise ValueError('Could not read NIF shader properties.')
            result.append(Mesh(name.value.decode('utf-8', errors='replace'), positions, normals,
                               uvs, triangles, textures, {}, shader.Shader_Type, shader.Shader_Flags_1,
                               shader.Env_Map_Scale, shader.Glossiness, shader.Spec_Str))
    finally:
        dll.destroy(handle)
    if not result:
        raise ValueError('No supported triangle geometry was found in this NIF.')
    return result


def load_mesh_textures(meshes, roots, max_size=1024):
    cache, missing = {}, set()
    for mesh in meshes:
        for slot in (0, 1, 5):
            relative = mesh.texture_paths.get(slot, '')
            if not relative:
                continue
            key = relative.casefold()
            if key not in cache:
                source = resolve_texture(relative, roots)
                if source is None:
                    missing.add(relative)
                    cache[key] = None
                else:
                    try:
                        with Image.open(source) as image:
                            image = image.convert('RGBA')
                            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                            cache[key] = image.copy()
                    except (OSError, ValueError):
                        missing.add(relative)
                        cache[key] = None
            if cache[key] is not None:
                mesh.textures[slot] = cache[key]
    return sorted(missing)


def load_cubemap(path, max_size=512):
    """Decode all six DDS faces, respecting mip-chain offsets and DX10 headers."""
    data = Path(path).read_bytes()
    if len(data) < 128 or data[:4] != b'DDS ' or struct.unpack_from('<I', data, 4)[0] != 124:
        raise ValueError('Choose a valid DDS cubemap.')
    height, width = struct.unpack_from('<II', data, 12)
    mip_count = max(1, struct.unpack_from('<I', data, 28)[0])
    caps2 = struct.unpack_from('<I', data, 112)[0]
    if not width or width != height or width > 16384 or mip_count > 16 or caps2 & 0xFE00 != 0xFE00:
        raise ValueError('A complete square DDS cubemap with all six faces is required.')
    fourcc = data[84:88]
    header_size, block_bytes, bytes_per_pixel = 128, None, None
    if fourcc == b'DX10':
        if len(data) < 148:
            raise ValueError('Incomplete DX10 DDS header.')
        dxgi, dimension, flags, array_size, _ = struct.unpack_from('<5I', data, 128)
        if dimension != 3 or not flags & 4 or array_size != 1:
            raise ValueError('Cubemap arrays and volume textures are not supported.')
        header_size = 148
        if dxgi in (71, 72, 80, 81): block_bytes = 8
        elif dxgi in (74, 75, 77, 78, 83, 84, 95, 96, 98, 99): block_bytes = 16
        elif dxgi in (28, 29, 87, 91): bytes_per_pixel = 4
        else: raise ValueError(f'Unsupported DDS cubemap format: DXGI {dxgi}.')
    elif fourcc in (b'DXT1', b'ATI1', b'BC4U', b'BC4S'): block_bytes = 8
    elif fourcc in (b'DXT3', b'DXT5', b'ATI2', b'BC5U', b'BC5S'): block_bytes = 16
    elif fourcc == bytes(4):
        bits = struct.unpack_from('<I', data, 88)[0]
        if bits not in (8, 16, 24, 32): raise ValueError('Unsupported uncompressed DDS pixel format.')
        bytes_per_pixel = bits // 8
    else: raise ValueError(f'Unsupported DDS cubemap compression: {fourcc!r}.')
    stride = 0
    for level in range(mip_count):
        w, h = max(1, width >> level), max(1, height >> level)
        stride += max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * block_bytes if block_bytes else w * h * bytes_per_pixel
    if len(data) < header_size + 6 * stride:
        raise ValueError('The cubemap is truncated or its mip chain is incomplete.')
    header = bytearray(data[:header_size])
    struct.pack_into('<I', header, 112, 0)
    if header_size == 148:
        struct.pack_into('<I', header, 136, 0)
    faces = []
    for face in range(6):
        start = header_size + face * stride
        with Image.open(io.BytesIO(header + data[start:start + stride])) as image:
            image = image.convert('RGBA')
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            faces.append(image.copy())
    return faces
