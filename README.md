# Local RAG Question & Answer System

This repository is being built as a fully local RAG system with two phases:

1. offline document ingestion and indexing
2. online question answering through Open WebUI

## Current MVP focus

The first implementation milestone is the offline ingestion pipeline.

### Supported inputs

- PDF files with extractable text
- TXT files
- Markdown files

### Deferred for MVP

- HTML
- OCR / scanned PDFs
- incremental indexing

## Tech stack

- Python
- Poetry
- LlamaIndex
- ChromaDB
- Ollama
- Open WebUI
- `BAAI/bge-m3`

## Install

```bash
poetry install
```

## Index documents

Place files under `documents/` and run:

```bash
poetry run rag-index
```

By default, the index is written to `data/chroma/` using a single Chroma collection.

## Check the ChromaDB index

After indexing, you can inspect the local Chroma database in a few ways.

### CLI: list collections and vector counts

Use a small Python snippet to connect to the persistent database and print the available collections:

```bash
poetry run python - <<'PY'
from chromadb import PersistentClient

client = PersistentClient(path="data/chroma")
for collection in client.list_collections():
    print(f"{collection.name}: {collection.count()} vectors")
PY
```

By default, this project uses Chroma's built-in values: `default_tenant` and `default_database`.
You do not need to pass them for the local embedded database shown above unless you have changed
the client configuration.

If you ever connect to a custom or remote Chroma deployment, the Python client also accepts
`tenant` and `database` arguments.

You can also inspect the first records in the main collection:

```bash
poetry run python - <<'PY'
from chromadb import PersistentClient

client = PersistentClient(path="data/chroma")
collection = client.get_collection("rag_documents")
print(collection.peek(5))
PY
```

### GUI: open the SQLite file

Chroma persists its local metadata in `data/chroma/chroma.sqlite3`. You can open that file with any SQLite GUI, such as:

- DB Browser for SQLite
- SQLiteStudio
- the SQLite viewer extension in VS Code

This is useful when you want to verify that the database exists, check tables, or confirm that indexing wrote data successfully.

### Optional: run a local Chroma server

If you prefer to inspect Chroma through a running local service, start the built-in server:

```bash
poetry run chroma run --path data/chroma --host localhost --port 8000
```

Then open `http://localhost:8000/docs` in your browser for the API documentation.

## Configuration

The ingestion pipeline reads these environment variables when present:

- `RAG_DOCUMENTS_DIR`
- `RAG_CHROMA_DIR`
- `RAG_CHROMA_COLLECTION`
- `RAG_EMBEDDING_MODEL`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`