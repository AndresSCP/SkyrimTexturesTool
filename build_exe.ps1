$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (!(Test-Path .venv/Scripts/python.exe)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el entorno" }
}
& .venv/Scripts/python.exe -m pip install -r requirements-build-lock.txt
if ($LASTEXITCODE -ne 0) { throw "No se pudieron instalar las dependencias" }
& .venv/Scripts/python.exe -m PyInstaller --noconfirm --clean SkyrimPBRConverter.spec
if ($LASTEXITCODE -ne 0) { throw "No se pudo generar el ejecutable" }

