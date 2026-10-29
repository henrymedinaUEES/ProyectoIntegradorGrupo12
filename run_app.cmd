@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: No existe el entorno virtual .venv.
  echo Ejecute: python -m venv .venv
  echo Luego: .venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app\app.py
