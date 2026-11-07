"""
Script de inicio para el servidor RAG Doc Assistant.
Ejecuta:
    python run.py
"""

import sys
from pathlib import Path
import uvicorn

# Asegurar que el directorio raíz esté en sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    print("🚀 Iniciando RAG Doc Assistant...")
    print("📍 Frontend Web: http://127.0.0.1:8000")
    print("📑 Swagger API Docs: http://127.0.0.1:8000/docs")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
