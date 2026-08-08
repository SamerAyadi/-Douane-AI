import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(ROOT_DIR / ".env")


DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
OFFICIAL_RAW_DATA_DIR = RAW_DATA_DIR / "official"
OFFICIAL_AR_DATA_DIR = OFFICIAL_RAW_DATA_DIR / "ar"
OFFICIAL_FR_DATA_DIR = OFFICIAL_RAW_DATA_DIR / "fr"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
DOUANE_METADATA_FILE = METADATA_DIR / "douane_documents.csv"
CHROMA_DB_DIR = ROOT_DIR / "chroma_db"


LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large"
)


COLLECTION_NAME = os.getenv("COLLECTION_NAME", "douane_documents")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

TOP_K = int(os.getenv("TOP_K", "2"))

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))
MAX_CONTEXT_CHUNK_CHARS = int(os.getenv("MAX_CONTEXT_CHUNK_CHARS", "1800"))
MAX_ANSWER_WORDS = int(os.getenv("MAX_ANSWER_WORDS", "120"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "220"))

OCR_ENABLED = os.getenv("OCR_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "ara+fra")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()


SUPPORTED_EXTENSIONS = [".pdf", ".docx"]


def create_required_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OFFICIAL_AR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OFFICIAL_FR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)