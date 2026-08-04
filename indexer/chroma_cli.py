from __future__ import annotations

import argparse
from pathlib import Path

from chromadb import PersistentClient

from indexer.config import load_config
from indexer.embeddings import build_embedding_model


def build_argument_parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(description="Simple CLI for inspecting and querying a local Chroma database.")
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=config.chroma_dir,
        help=f"Persistent Chroma directory (default: {config.chroma_dir})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Collection name (default: {config.collection_name})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=config.embedding_model,
        help=f"Embedding model for query text (default: {config.embedding_model})",
    )
    parser.add_argument("--list", action="store_true", help="List collections")
    parser.add_argument("--query", type=str, default=None, help="Query text")
    parser.add_argument("--k", type=int, default=3, help="Top-k results to return (default: 3)")
    return parser


def _collection_name(item: object) -> str:
    if isinstance(item, str):
        return item
    name = getattr(item, "name", None)
    return str(name) if name else str(item)


def list_collections(client: PersistentClient) -> int:
    collections = client.list_collections()
    if not collections:
        print("No collections found.")
        return 0

    print("Collections:")
    count = 0
    for item in collections:
        name = _collection_name(item)
        vector_count = "?"
        try:
            vector_count = str(client.get_collection(name).count())
        except Exception:
            pass

        print(f"- {name}: {vector_count} vectors")
        count += 1
    return count


def query_collection(
    client: PersistentClient,
    collection_name: str,
    query_text: str,
    k: int,
    embedding_model_name: str,
) -> int:
    collection = client.get_collection(collection_name)
    embedding_model = build_embedding_model(embedding_model_name)
    query_embedding = embedding_model.get_text_embedding(query_text)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    if not ids:
        print("No matches found.")
        return 0

    print(f"Top {len(ids)} results in '{collection_name}':")
    for idx, item_id in enumerate(ids, start=1):
        document = documents[idx - 1] if idx - 1 < len(documents) else ""
        metadata = metadatas[idx - 1] if idx - 1 < len(metadatas) else {}
        distance = distances[idx - 1] if idx - 1 < len(distances) else None

        print(f"\n[{idx}] id={item_id}")
        if distance is not None:
            print(f"distance={distance}")
        print(f"metadata={metadata}")
        print(f"document={document}")

    return len(ids)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be greater than 0")

    if not args.list and not args.query:
        parser.print_help()
        print("\nChoose at least one action: --list and/or --query \"your text\"")
        return

    if not args.chroma_dir.exists():
        print(f"Chroma directory not found: {args.chroma_dir}")
        return

    client = PersistentClient(path=str(args.chroma_dir))

    if args.list:
        list_collections(client)

    if args.query:
        try:
            query_collection(client, args.collection, args.query, args.k, args.embedding_model)
        except Exception as exc:
            print(f"Query failed: {exc}")
            print(
                "Hint: ensure --embedding-model matches the model used during indexing for this collection."
            )


if __name__ == "__main__":
    main()