from __future__ import annotations

import argparse
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex

from indexer.chunking import chunk_documents
from indexer.chroma import build_vector_store, clear_collection
from indexer.config import IngestionConfig, load_config
from indexer.embeddings import build_embedding_model
from indexer.loaders import load_documents


def ingest_documents(config: IngestionConfig) -> int:
    documents = load_documents(config.documents_dir)
    if not documents:
        print(f"No supported documents found in {config.documents_dir}")
        return 0

    nodes = chunk_documents(documents, config.chunk_size, config.chunk_overlap)

    clear_collection(config.chroma_dir, config.collection_name)
    vector_store, collection = build_vector_store(config.chroma_dir, config.collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = build_embedding_model(config.embedding_model)
    Settings.embed_model = embed_model

    VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)

    print(
        f"Indexed {len(documents)} source documents into {len(nodes)} chunks "
        f"and persisted {collection.count()} vectors in {config.chroma_dir}"
    )
    return len(nodes)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the offline Chroma index from local documents.")
    parser.add_argument("--documents", type=Path, default=None, help="Document root directory")
    parser.add_argument("--chroma-dir", type=Path, default=None, help="Persistent Chroma directory")
    parser.add_argument("--collection", type=str, default=None, help="Chroma collection name")
    parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=None, help="Sentence chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Sentence chunk overlap")
    return parser


def resolve_config(args: argparse.Namespace) -> IngestionConfig:
    config = load_config()
    return IngestionConfig(
        documents_dir=args.documents or config.documents_dir,
        chroma_dir=args.chroma_dir or config.chroma_dir,
        collection_name=args.collection or config.collection_name,
        embedding_model=args.embedding_model or config.embedding_model,
        chunk_size=args.chunk_size or config.chunk_size,
        chunk_overlap=args.chunk_overlap or config.chunk_overlap,
    )


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = resolve_config(args)
    ingest_documents(config)


if __name__ == "__main__":
    main()