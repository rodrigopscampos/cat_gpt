from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class RuntimeConfig:
    chroma_dir: Path
    collection_name: str
    embedding_model: str
    llm_model: str
    ollama_base_url: str
    top_k: int
    max_context_chars: int
    conversation_turns: int
    temperature: float
    request_timeout: float

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            chroma_dir=Path(os.getenv("RAG_CHROMA_DIR", "data/chroma")),
            collection_name=os.getenv("RAG_CHROMA_COLLECTION", "rag_documents"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
            llm_model=os.getenv("RAG_LLM_MODEL", "gemma3:1b"),
            ollama_base_url=os.getenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434"),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000")),
            conversation_turns=int(os.getenv("RAG_CONVERSATION_TURNS", "4")),
            temperature=float(os.getenv("RAG_TEMPERATURE", "0.2")),
            request_timeout=float(os.getenv("RAG_OLLAMA_TIMEOUT", "120")),
        )


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig.from_env()