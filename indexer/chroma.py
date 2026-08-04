from __future__ import annotations

from pathlib import Path

from chromadb import PersistentClient
from llama_index.vector_stores.chroma import ChromaVectorStore


def clear_collection(persist_dir: Path, collection_name: str) -> None:
    client = PersistentClient(path=str(persist_dir))
    existing_names = {collection.name for collection in client.list_collections()}
    if collection_name in existing_names:
        client.delete_collection(collection_name)


def build_vector_store(persist_dir: Path, collection_name: str) -> tuple[ChromaVectorStore, object]:
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    return ChromaVectorStore(chroma_collection=collection), collection