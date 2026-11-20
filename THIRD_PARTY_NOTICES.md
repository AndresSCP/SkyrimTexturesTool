# Third-party components

The NIF viewer uses NiflyDLL from PyNifly V30.2.0 by Bad Dog, based on nifly by ousnius and its contributors. The ctypes buffer declarations in nif_preview.py are adapted from the same release's NiflyDefs.hpp. These components are licensed under GPL-3.0; license texts are in Resources/licenses.

- PyNifly release and corresponding source: https://github.com/BadDogSkyrim/PyNifly/tree/V30.2.0
- Release binary: https://github.com/BadDogSkyrim/PyNifly/releases/tag/V30.2.0
- nifly source: https://github.com/ousnius/nifly

The viewer uses ModernGL and glcontext (MIT); see their bundled licenses. Other dependencies retain their own licenses. The PyInstaller bundle includes dependency metadata where provided by its hooks.

Skyrim NIF files, material textures, and cubemaps are external user assets and are not included in the executable. Resources/viewer_defaults.json contains optional local paths only.
