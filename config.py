import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-key-change-me")
    DATABASE = os.getenv("DATABASE_PATH", str(BASE_DIR / "pdf_insights.sqlite3"))
    PDF_DATASET = os.getenv("PDF_DATASET", str(BASE_DIR / "Pdf_Dataset"))
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL", "all-MiniLM-L6-v2")
    SEMANTIC_INDEX_PATH = os.getenv("SEMANTIC_INDEX_PATH", str(BASE_DIR / "storage" / "index" / "documents.faiss"))
    SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.20"))
