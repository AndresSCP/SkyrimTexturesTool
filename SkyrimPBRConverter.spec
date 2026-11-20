# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['skyrim_pbr_converter.py'],
    pathex=[],
    binaries=[('Resources/texconv.exe', 'Resources'), ('Resources/NiflyDLL.dll', 'Resources')],
    datas=[('Resources/viewer_defaults.json', 'Resources'), ('Resources/licenses', 'Resources/licenses'), ('THIRD_PARTY_NOTICES.md', '.')],
    hiddenimports=['glcontext', 'glcontext.wgl'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SkyrimPBRConverter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
