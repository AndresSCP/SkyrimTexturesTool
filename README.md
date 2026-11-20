# Skyrim PBR Converter

[English](#english) · [Español](#español)

## English

A Windows tool for preparing diffuse, normal, and metallic textures for Skyrim. Combine ambient occlusion (AO) and detail maps, preview materials in 2D or on a 3D NIF model, and export DDS textures with texconv.

### Download and run

Download [SkyrimPBRConverter.exe](https://github.com/AndresSCP/SkyrimTexturesTool/releases/latest/download/SkyrimPBRConverter.exe) and double-click it on 64-bit Windows. The portable executable includes Python, its dependencies, and texconv. The first launch may take a moment while it extracts the bundled components.

### Usage

1. Use **Browse** in the **Standard** and **Advanced** tabs to load your images.
2. Adjust the settings and check the preview.
3. Choose **DirectX** or **OpenGL** to match your source normal map. **Detect from filename** recognizes markers such as `normal_gl`, `normal_dx`, `OpenGL`, and `DirectX`. It keeps your manual selection if the filename has no recognizable marker.
4. Choose an output folder, filename prefix, and resolution. Output textures are square.
5. Use **Save Diffuse Output**, **Save Normal Output**, or **Save Metallic Output** to export the textures you need.

The output files are `<prefix>_d.dds`, `<prefix>_n.dds`, and `<prefix>_m.dds`. Diffuse and metallic use BC3. Normal maps support BC3 or BC7. The tool validates the DDS header before replacing an existing file. Large exports can temporarily keep the interface busy. Invalid inputs preserve the previously loaded texture, and setting edge intensity to zero produces a black edge map.

The final appearance in Skyrim depends on the material and shader you configure.

### Adaptive layout and tooltips

The layout places the preview and settings side by side in wide windows and stacks them below 1,100 pixels. Drag the divider to resize the panels and scroll to reach settings and export controls. The interface supports Windows display scaling and has a minimum window size of 620 × 580. Hover over a control for an English tooltip.

### 3D NIF and cubemap preview

Select **3D Preview** above the preview area, then click **Load NIF** to open a Skyrim SE/AE model in its base pose. The viewer searches for textures relative to the mod's `meshes` directory. Use **Texture search folder** to add a mod folder or the game's `Data` directory.

- Drag with the left mouse button to orbit, scroll to zoom, and double-click to reset the camera.
- **Cubemap folder** lists six-face DDS cubemaps and selects the model's cubemap by name when available. Use **Load DDS** to pick another one or **No cubemap** to disable reflections.
- **Reflection intensity**, **Environment rotation**, and **Light intensity** affect the preview.
- **Apply edits to** chooses which shapes receive the processed maps. **Show selected shape only** isolates a shape.
- **Use material textures** loads the selected material's diffuse map, DirectX normal map, and reflection mask into the editor and clears the other input maps. The original files stay intact.
- **Use processed textures** lets you compare the edited maps with the model's original textures. The preview refreshes shortly after you adjust a control.

The viewer supports geometry, UVs, normals, base-pose transforms, diffuse maps, normal maps with specular alpha, and reflection masks. It requires OpenGL 3.3. The 2D preview works independently. Animation, skinning, collision visualization, parallax, and exact reproduction of Skyrim, ENB, or Community Shaders rendering are outside the viewer's scope. Export writes DDS textures only.

Provide your own models, textures, and cubemaps. You can set optional starting paths with `nif_file` and `cubemap_folder` in [viewer_defaults.json](Resources/viewer_defaults.json).

### Develop and build

From the repository root on Windows with Python installed:

```powershell
python -m pip install -r requirements.txt
python skyrim_pbr_converter.py
python -m unittest test test_viewer -v
```

Set `SKYRIM_TEST_NIF` and `SKYRIM_TEST_CUBEMAPS` to run integration tests with a local model and cubemap folder.

To build the portable executable, run `.\build_exe.ps1` in PowerShell. The script installs the pinned dependencies from `requirements-build-lock.txt` and writes `dist/SkyrimPBRConverter.exe`.

See [third-party notices](THIRD_PARTY_NOTICES.md) and [licenses](Resources/licenses) for bundled components.

## Español

Aplicación de Windows para preparar texturas diffuse, normal y metallic, combinar AO y mapas de detalle, y exportar DDS mediante texconv.

### Abrir la aplicación

Descarga `SkyrimPBRConverter.exe` desde [GitHub Releases](https://github.com/AndresSCP/SkyrimTexturesTool/releases) y ábrelo con doble clic. Puedes copiar ese archivo a otra carpeta: incluye Python, las bibliotecas y `Resources/texconv.exe`. No necesitas instalar Python ni usar comandos para abrirlo. La primera apertura puede tardar mientras se extraen los componentes.

### Uso

1. Carga tus imágenes con Browse en las pestañas Standard y Advanced.
2. Ajusta las intensidades y revisa la vista previa.
3. Selecciona DirectX u OpenGL según el origen del normal. Detect from filename reconoce nombres con normal_gl, normal_dx, OpenGL o DirectX; si no encuentra una pista, conserva la selección manual. No puede determinar la convención a partir de los píxeles.
4. Selecciona una carpeta de salida, el prefijo y las resoluciones.
5. Guarda las texturas con Save Diffuse / Normal / Metallic Output.

Las exportaciones se llaman `<prefijo>_d.dds`, `<prefijo>_n.dds` y `<prefijo>_m.dds`. Guardar nuevamente el mismo nombre reemplaza la versión anterior cuando la conversión termina correctamente. Diffuse y metallic usan BC3; normal permite BC7 o BC3. La conversión se ejecuta en el hilo de la interfaz y puede dejarla ocupada durante exportaciones grandes.

El procesamiento aproxima materiales PBR a las salidas ofrecidas por esta herramienta; el resultado en el juego también depende del material y shader configurados. La resolución de salida siempre es cuadrada.

### Desarrollo

Dependencias: `requirements.txt`. Para ejecutar desde el código: `python skyrim_pbr_converter.py`. Para comprobar las pruebas: `python -m unittest test test_viewer -v`.

Para reconstruir el ejecutable ejecuta `build_exe.ps1` desde PowerShell. Requiere Python y acceso a internet para instalar las dependencias. La salida queda en `dist/SkyrimPBRConverter.exe`.

### Mejoras aplicadas

- Se eliminó la segunda copia del programa.
- Las imágenes inválidas muestran un error y conservan la textura anterior.
- La conversión DDS se prepara temporalmente y verifica la cabecera antes de reemplazar el archivo final.
- Los recursos se encuentran tanto al ejecutar el código como el ejecutable.
- La detección por nombre se actualiza al cargar un nuevo normal.
- La intensidad cero de los bordes genera un mapa negro.
- Se añadieron pruebas de carga, detección, conservación de archivos y exportación real BC3/BC7.

### Interfaz adaptable y ayuda

La interfaz usa tonos gris oscuro y azul apagado, texto de 9,5 puntos y adaptación al escalado de Windows. En ventanas anchas muestra vista previa y ajustes lado a lado; por debajo de 1100 píxeles los apila. Los paneles se pueden redimensionar arrastrando su separación y los ajustes y exportaciones permanecen accesibles mediante desplazamiento. Tamaño mínimo: 620 × 580.

Mantén el cursor sobre un ajuste, selector o botón para ver una explicación en inglés de su función y efecto.

### Vista 3D de NIF y cubemaps

Usa el selector 2D / 3D encima de la vista previa. En la pestaña **3D Preview**, pulsa **Load NIF** para abrir un modelo de Skyrim SE/AE. El visor muestra las piezas en pose base y busca sus texturas a partir de la carpeta meshes del mod. **Texture search folder** permite añadir una carpeta de mod o Data para resolver archivos faltantes.

- Arrastra con el botón izquierdo para girar, usa la rueda para acercarte y haz doble clic para reiniciar la cámara.
- **Cubemap folder** enumera los DDS de seis caras de la carpeta indicada. Al abrir el modelo se selecciona su cubemap por nombre si está disponible. **Load DDS** permite escoger otro archivo. **No cubemap** desactiva las reflexiones.
- Ajusta **Reflection intensity**, **Environment rotation** y **Light intensity** para inspeccionar el material. Estos ajustes solo afectan a la vista previa.
- **Apply edits to** selecciona qué piezas reciben las texturas del editor. Las otras conservan sus materiales originales. **Show selected shape only** permite aislar una pieza.
- **Use material textures** carga diffuse, normal DirectX y máscara de reflexión del material seleccionado en el editor y limpia los demás mapas de entrada. No altera los archivos originales.
- **Use processed textures** permite comparar los mapas ajustados con las texturas originales del NIF. Los cambios se actualizan tras una pequeña pausa al mover controles.

La vista utiliza geometría, coordenadas UV, normales, transformaciones de pose base y materiales con diffuse, normal/alpha especular y máscara de reflexión. No incluye animación, simulación de piel, colisiones visibles, parallax ni una reproducción exacta de los shaders de Skyrim, ENB o Community Shaders. La exportación continúa generando texturas DDS; el visor no escribe NIF ni inserta rutas de cubemap en ellos.

Requiere OpenGL 3.3 para el visor 3D. La vista 2D se mantiene independiente. Los modelos, texturas y cubemaps del juego no se empaquetan: deben seguir disponibles en sus carpetas externas. `Resources/viewer_defaults.json` permite configurar rutas iniciales opcionales con las claves `nif_file` y `cubemap_folder`. Las pruebas de integración permiten indicar recursos locales mediante `SKYRIM_TEST_NIF` y `SKYRIM_TEST_CUBEMAPS`.

Las dependencias del visor y sus licencias se describen en `THIRD_PARTY_NOTICES.md` y `Resources/licenses`.
