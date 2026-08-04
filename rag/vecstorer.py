from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore

from indexer.embeddings import build_embedding_model
from rag.config import RuntimeConfig, load_runtime_config


@dataclass(frozen=True)
class RetrievedChunk:
    node_id: str
    text: str
    score: float | None
    metadata: dict[str, Any]
    citation: str


def format_citation(metadata: dict[str, Any]) -> str:
    parts: list[str] = []

    document_name = metadata.get("document_name") or metadata.get("relative_path")
    if document_name:
        parts.append(str(document_name))

    page = metadata.get("page")
    if page is not None:
        parts.append(f"p. {page}")

    section = metadata.get("section")
    if section:
        parts.append(f"section {section}")

    chunk_id = metadata.get("chunk_id")
    if chunk_id and not parts:
        parts.append(str(chunk_id))

    return ", ".join(parts) if parts else "source unavailable"


class VecStoreRetriever:
    def __init__(
        self,
        config: RuntimeConfig
    ) -> None:
        self.config = config or load_runtime_config()
        self.embed_model = build_embedding_model(self.config.embedding_model)
        self.collection = self._load_collection(self.config.chroma_dir, self.config.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.index = VectorStoreIndex.from_vector_store(vector_store, embed_model=self.embed_model)
        self.retriever = self.index.as_retriever(similarity_top_k=self.config.top_k)

    def _load_collection(self, chroma_dir: Path, collection_name: str):
        if not chroma_dir.exists():
            raise FileNotFoundError(f"Chroma directory not found: {chroma_dir}")

        client = PersistentClient(path=str(chroma_dir))
        return client.get_collection(collection_name)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        nodes = self.retriever.retrieve(query)
        return [self._convert_node(node) for node in nodes]

    def _convert_node(self, node: NodeWithScore) -> RetrievedChunk:
        metadata = dict(node.node.metadata or {})
        return RetrievedChunk(
            node_id=str(node.node.node_id),
            text=node.node.get_content(),
            score=node.score,
            metadata=metadata,
            citation=format_citation(metadata),
        )
