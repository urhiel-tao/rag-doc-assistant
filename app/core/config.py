import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory: rag-doc-assistant root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

class Settings:
    PROJECT_NAME: str = "RAG Doc Assistant"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Directories
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"
    VECTORSTORE_DIR: Path = BASE_DIR / "data" / "vectorstore"

    # Chunking settings
    CHUNK_SIZE: int = 500  # palabras / caracteres aproximados
    CHUNK_OVERLAP: int = 60  # solapamiento entre fragmentos

    # RAG settings
    DEFAULT_TOP_K: int = 3

    # LLM Settings (Google Gemini)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

settings = Settings()

# Ensure directories exist
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
