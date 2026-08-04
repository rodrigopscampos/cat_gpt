# Local RAG Question & Answer System

## Objective

Build a **100% local Retrieval-Augmented Generation (RAG)** system capable of answering questions about a collection of documents through a ChatGPT-like interface.

### Requirements

* Run locally without requiring cloud services.
* Index PDF, HTML, TXT and Markdown documents.
* Use Python as the primary implementation language.
* Provide a ChatGPT-like user experience through Open WebUI.
* Keep the architecture simple, modular and easily extensible.
* Allow future replacement of individual components (LLM, embeddings, vector database, etc.).

---

# Architecture Overview

The solution is divided into two independent parts:

1. **Offline Data Ingestion**
2. **Online Question & Answer**

The only shared artifact between them is the persistent vector database.

```text
                 OFFLINE                              ONLINE

          Data Ingestion                     Question & Answer

     Documents (PDF/HTML/TXT)             User Question (Open WebUI)
                │                                    │
                ▼                                    ▼
         Python Indexer                    Custom Open WebUI Pipeline
                │                                    │
                ▼                                    ▼
      Extraction / Chunking                 Load Chroma Database
                │                                    │
                ▼                                    ▼
      Generate Embeddings                Retrieve Relevant Chunks
                │                                    │
                ▼                                    ▼
      Persist Chroma Database             Prompt Construction
                │                                    │
                └────────────────────────►───────────┘
                                   Ollama (gemma3:1b)
```

---

# Technology Stack

| Layer                | Technology                      |
| -------------------- | ------------------------------- |
| Programming Language | Python                          |
| UI                   | Open WebUI                      |
| LLM Runtime          | Ollama                          |
| Initial LLM          | gemma3:1b                       |
| RAG Framework        | LlamaIndex                      |
| Vector Database      | ChromaDB (embedded, persistent) |
| Storage              | Local filesystem                |

No dedicated backend (FastAPI) will be used. The custom RAG pipeline will be loaded directly by Open WebUI.

---

# Component Responsibilities

## Open WebUI

Provides the user interface:

* ChatGPT-like experience
* Conversation history
* Markdown rendering
* Streaming responses
* Loads and executes the custom RAG pipeline

---

## Custom RAG Pipeline

Responsible for:

* Loading the vector database
* Executing similarity search
* Building prompts
* Calling the LLM
* Returning answers

---

## LlamaIndex

Responsible for:

* Document loading
* Chunking
* Embedding generation
* Retrieval
* Query engine

---

## ChromaDB

Responsible for storing:

* Embeddings
* Chunk metadata
* Persistent vector index

Runs embedded in the Python process.

---

## Ollama

Responsible only for local inference.

Initial model:

* gemma3:1b

The model can be upgraded later without affecting the rest of the architecture.

---

# Front 1 – Offline Data Ingestion

## Goal

Build the vector database from the document collection.

This process runs only when documents are added or updated.

After indexing completes, the process terminates.

## Input

```
documents/
```

## Output

```
data/
    chroma/
```

## Workflow

```text
Load Documents
        ↓
Extract Text
        ↓
Normalize
        ↓
Chunk
        ↓
Generate Embeddings
        ↓
Persist Chroma Database
```

## Supported Documents

Initial scope:

* PDF
* HTML
* TXT
* Markdown

Future support:

* DOCX
* CSV
* Excel
* OCR
* Images

---

# Front 2 – Online Question & Answer

## Goal

Answer user questions using the pre-built vector database.

No indexing occurs during runtime.

## Workflow

```text
Question
    ↓
Generate Question Embedding
    ↓
Load Chroma Database
    ↓
Similarity Search
    ↓
Retrieve Top-K Chunks
    ↓
Build Prompt
    ↓
Call Ollama
    ↓
Answer
```

---

# Suggested Repository Structure

```text
rag-system/

├── documents/
│
├── data/
│   └── chroma/
│
├── indexer/
│   ├── main.py
│   ├── loaders/
│   ├── chunking.py
│   ├── embeddings.py
│   └── chroma.py
│
├── rag/
│   ├── pipeline.py
│   ├── retriever.py
│   ├── prompt.py
│   └── llm.py
│
├── openwebui/
│   └── pipeline.py
│
├── requirements.txt
│
└── README.md
```

---

# Design Principles

* Local-first execution
* No cloud dependency
* Minimal infrastructure
* Pure Python implementation
* Persistent indexes
* Modular components
* Separation between indexing and querying
* Easy replacement of LLMs, embedding models and vector databases

---

# Pending Technical Decisions

## 1. Embedding Model

Candidates:

* BAAI/bge-m3 (recommended)
* nomic-embed-text
* bge-small

Current recommendation:

> **BAAI/bge-m3**

---

## 2. Chunking Strategy

Options:

* Fixed-size chunks
* Sentence-aware chunking
* Semantic chunking

Current recommendation:

> **Sentence-aware chunking using LlamaIndex**

---

## 3. Metadata Schema

Each chunk should contain metadata such as:

* document
* page
* section
* chunk_id
* source_type

---

## 4. Retrieval Strategy

Initial implementation:

* Top-K vector similarity

Suggested default:

* Top-K = 5

Future improvements:

* Hybrid search (BM25 + vectors)
* Reranking

---

## 5. Conversation Memory

Current recommendation:

* Sliding conversation window only

The vector database should contain only document knowledge.

---

## 6. Knowledge Base Organization

Current recommendation:

Support multiple Chroma collections, allowing one knowledge base per project.

---

## 7. Citation Strategy

Answers should include references to the original source whenever possible.

Example:

* Document name
* Page number
* Section

---

## 8. Configuration

Externalize configurable parameters:

* LLM model
* Embedding model
* Chunk size
* Chunk overlap
* Top-K
* Chroma database path

Using either:

* YAML
* Environment variables

---

# Future Enhancements

Outside the MVP scope:

* Hybrid retrieval
* Cross-encoder reranking
* OCR
* Incremental indexing
* Document versioning
* Evaluation metrics
* Larger local LLMs
* Optional cloud LLM providers

---

# Current Project Status

## Architecture

* ✅ Overall architecture defined
* ✅ Technology stack selected
* ✅ Offline indexing separated from online inference
* ✅ FastAPI removed from the design
* ✅ Open WebUI selected as the application host
* ✅ ChromaDB selected as the persistent embedded vector database

## Remaining Design Decisions

* ⏳ Select embedding model
* ⏳ Finalize chunking configuration
* ⏳ Define metadata schema
* ⏳ Define knowledge base organization
* ⏳ Define citation behavior

## Implementation

No implementation has started.

The first milestone is to build the offline indexing pipeline capable of reading documents, generating embeddings and producing a persistent Chroma database that can later be consumed by the Open WebUI RAG pipeline.
