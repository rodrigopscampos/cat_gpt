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

## Configuration

The ingestion pipeline reads these environment variables when present:

- `RAG_DOCUMENTS_DIR`
- `RAG_CHROMA_DIR`
- `RAG_CHROMA_COLLECTION`
- `RAG_EMBEDDING_MODEL`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`