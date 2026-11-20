# Skyrim PBR Converter

Aplicación de Windows para preparar texturas diffuse, normal y metallic, combinar AO y mapas de detalle, y exportar DDS mediante texconv.

## Abrir la aplicación

Descarga `SkyrimPBRConverter.exe` desde [GitHub Releases](https://github.com/AndresSCP/SkyrimTexturesTool/releases) y ábrelo con doble clic. Puedes copiar ese archivo a otra carpeta: incluye Python, las bibliotecas y `Resources/texconv.exe`. No necesitas instalar Python ni usar comandos para abrirlo. La primera apertura puede tardar mientras se extraen los componentes.

## Uso

1. Carga tus imágenes con Browse en las pestañas Standard y Advanced.
2. Ajusta las intensidades y revisa la vista previa.
3. Selecciona DirectX u OpenGL según el origen del normal. Detect from filename reconoce nombres con normal_gl, normal_dx, OpenGL o DirectX; si no encuentra una pista, conserva la selección manual. No puede determinar la convención a partir de los píxeles.
4. Selecciona una carpeta de salida, el prefijo y las resoluciones.
5. Guarda las texturas con Save Diffuse / Normal / Metallic Output.

Las exportaciones se llaman `<prefijo>_d.dds`, `<prefijo>_n.dds` y `<prefijo>_m.dds`. Guardar nuevamente el mismo nombre reemplaza la versión anterior cuando la conversión termina correctamente. Diffuse y metallic usan BC3; normal permite BC7 o BC3. La conversión se ejecuta en el hilo de la interfaz y puede dejarla ocupada durante exportaciones grandes.

El procesamiento aproxima materiales PBR a las salidas ofrecidas por esta herramienta; el resultado en el juego también depende del material y shader configurados. La resolución de salida siempre es cuadrada.

## Desarrollo

Dependencias: `requirements.txt`. Para ejecutar desde el código: `python skyrim_pbr_converter.py`. Para comprobar las pruebas: `python -m unittest test test_viewer -v`.

Para reconstruir el ejecutable ejecuta `build_exe.ps1` desde PowerShell. Requiere Python y acceso a internet para instalar las dependencias. La salida queda en `dist/SkyrimPBRConverter.exe`.

## Mejoras aplicadas

- Se eliminó la segunda copia del programa.
- Las imágenes inválidas muestran un error y conservan la textura anterior.
- La conversión DDS se prepara temporalmente y verifica la cabecera antes de reemplazar el archivo final.
- Los recursos se encuentran tanto al ejecutar el código como el ejecutable.
- La detección por nombre se actualiza al cargar un nuevo normal.
- La intensidad cero de los bordes genera un mapa negro.
- Se añadieron pruebas de carga, detección, conservación de archivos y exportación real BC3/BC7.

## Interfaz adaptable y ayuda

La interfaz usa tonos gris oscuro y azul apagado, texto de 9,5 puntos y adaptación al escalado de Windows. En ventanas anchas muestra vista previa y ajustes lado a lado; por debajo de 1100 píxeles los apila. Los paneles se pueden redimensionar arrastrando su separación y los ajustes y exportaciones permanecen accesibles mediante desplazamiento. Tamaño mínimo: 620 × 580.

Mantén el cursor sobre un ajuste, selector o botón para ver una explicación en inglés de su función y efecto.

## Vista 3D de NIF y cubemaps

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
