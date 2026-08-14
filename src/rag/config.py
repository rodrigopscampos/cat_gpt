from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

@dataclass(frozen=True)
class RuntimeConfig:
    documents_dir: Path = Path("documents")
    collection_name: str = "rag_documents"
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "gemma3:1b"
    ollama_base_url: str = "http://localhost:11434"
    top_k: int = 5
    max_context_chars: int = 12000
    conversation_turns: int = 4
    temperature: float = 0.2
    request_timeout: float = 120.0
    chunk_size: int = 1024
    chunk_overlap: int = 128

    def _chroma_dir_name(self) -> str:
        raw_name = f"{self.embedding_model}-{self.chunk_size}-{self.chunk_overlap}"
        return re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name).strip("-")

    @property
    def chroma_dir(self):
        return Path("data/chroma") / self._chroma_dir_name()

Config = RuntimeConfig()
