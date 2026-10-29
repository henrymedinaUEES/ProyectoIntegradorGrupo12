$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "No existe .venv. Ejecute: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

Set-Location -LiteralPath $PSScriptRoot
& $python -m streamlit run app\app.py
