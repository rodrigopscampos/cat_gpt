from __future__ import annotations
from pathlib import Path
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from chromadb import Collection, PersistentClient
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from rag.config import Config
import rag.loaders as loaders
import rag.embedding as embedding

def _chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int):
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    for index, node in enumerate(nodes):
        node.metadata["chunk_index"] = index
        node.metadata["chunk_id"] = node.metadata.get(
            "chunk_id",
            f"{node.metadata.get('relative_path', 'document')}::chunk-{index:05d}",
        )

    print(f"Split {len(documents)} documents into {len(nodes)} chunks")
    return nodes

def _clear_collection(client: PersistentClient, persist_dir: Path, collection_name: str) -> None:
    existing_names = {collection.name for collection in client.list_collections()}
    if collection_name in existing_names:
        client.delete_collection(collection_name)
        print(f"Cleared existing collection '{collection_name}' in {persist_dir}")


def _build_vector_store(client: PersistentClient, persist_dir: Path, collection_name: str) -> tuple[ChromaVectorStore, Collection]:
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    print(f"Created or loaded collection '{collection_name}' in {persist_dir}")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    print(f"Initialized ChromaVectorStore with collection '{collection_name}'")
    return vector_store, collection


def main() -> None:
    print(f"Loading documents from {Config.documents_dir}")
    
    documents = loaders.load_documents(Config.documents_dir)

    if not documents:
        print(f"No supported documents found in {Config.documents_dir}")
        return

    nodes = _chunk_documents(documents, Config.chunk_size, Config.chunk_overlap)

    Config.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(Config.chroma_dir))

    _clear_collection(client, Config.chroma_dir, Config.collection_name)

    vector_store, collection = _build_vector_store(client, Config.chroma_dir, Config.collection_name)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    Settings.embed_model = embedding.Embedding

    # Build index and persist vectors
    print(f"Seeding vector store in {Config.chroma_dir} with collection '{Config.collection_name}'")
    print(f"Indexing {len(nodes)} chunks into vector store...")
    VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embedding.Embedding, show_progress=True)

    print(
        f"Indexed {len(documents)} source documents into {len(nodes)} chunks "
        f"and persisted {collection.count()} vectors in {Config.chroma_dir}"
    )


if __name__ == "__main__":
    main()
