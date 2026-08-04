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

### CLI: default-first Chroma commands

Use the built-in Chroma CLI from this repository:

```bash
poetry run rag-chroma --list
```

The command above reads defaults from project config, so in the common case you don't need
to pass extra parameters:

- Chroma directory: `data/chroma`
- Collection: `rag_documents`
- Embedding model for query text: `BAAI/bge-m3`

Run a sample top-k query (using defaults):

```bash
poetry run rag-chroma --query "cat nutrition by age"
```

Override defaults when needed:

```bash
poetry run rag-chroma --chroma-dir data/chroma --collection rag_documents --query "kitten feeding" --k 5
```

If your collection was indexed with a different embedding model, override it explicitly:

```bash
poetry run rag-chroma --query "kitten feeding" --embedding-model BAAI/bge-small-en-v1.5
```

You can also run it as a module:

```bash
poetry run python -m indexer.chroma_cli --list
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