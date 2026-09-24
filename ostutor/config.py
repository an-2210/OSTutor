"""
Centralized Configuration Module for OSTutorLLM RAG Pipeline.

Provides paths, parameters, and environment overrides for document ingestion,
chunking, embedding generation, vector indexing, and retrieval.
"""

import os
from pathlib import Path

# Base Repository Directory
BASE_DIR = Path(__file__).resolve().parent

# Data Paths
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", BASE_DIR / "data" / "rag" / "raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", BASE_DIR / "data" / "rag" / "processed"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "data" / "rag" / "index"))
METADATA_CSV_PATH = Path(os.getenv("METADATA_CSV_PATH", BASE_DIR / "data" / "rag" / "metadata.csv"))
TAXONOMY_JSON_PATH = Path(os.getenv("TAXONOMY_JSON_PATH", BASE_DIR / "data" / "taxonomy.json"))

# Processed Artifact Paths
CHUNKS_JSONL_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_METADATA_PATH = INDEX_DIR / "chunks.json"

# RAG Hyperparameters
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP", "100"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Supported Document Extensions
SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".txt", ".docx"}


def ensure_directories_exist() -> None:
    """Create data directories if they do not exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories_exist()
    print("OSTutorLLM configuration loaded.")
    print(f"Base Directory       : {BASE_DIR}")
    print(f"Raw Data Directory   : {RAW_DATA_DIR}")
    print(f"Embedding Model      : {EMBEDDING_MODEL_NAME}")
    print(f"Chunk Target Size    : {CHUNK_SIZE_TOKENS} tokens (overlap: {CHUNK_OVERLAP_TOKENS})")
