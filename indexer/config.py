from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class IngestionConfig:
    documents_dir: Path = Path(os.getenv("RAG_DOCUMENTS_DIR", "documents"))
    chroma_dir: Path = Path(os.getenv("RAG_CHROMA_DIR", "data/chroma"))
    collection_name: str = os.getenv("RAG_CHROMA_COLLECTION", "rag_documents")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1024"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "128"))


def load_config() -> IngestionConfig:
    return IngestionConfig()