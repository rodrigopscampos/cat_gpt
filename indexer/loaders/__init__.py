from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import fitz
from llama_index.core import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def discover_source_files(documents_dir: Path) -> list[Path]:
    if not documents_dir.exists():
        return []

    return sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _build_metadata(path: Path, root: Path, source_type: str, page_number: int | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "document_name": path.name,
        "relative_path": _relative_path(path, root),
        "source_type": source_type,
    }
    if page_number is not None:
        metadata["page"] = page_number
    return metadata


def _load_pdf(path: Path, root: Path) -> Iterable[Document]:
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = normalize_text(page.get_text("text"))
            if not text:
                continue

            yield Document(text=text, metadata=_build_metadata(path, root, "pdf", page_number))


def _load_text(path: Path, root: Path) -> Document | None:
    text = normalize_text(path.read_text(encoding="utf-8", errors="replace"))
    if not text:
        return None

    source_type = "markdown" if path.suffix.lower() in {".md", ".markdown"} else "txt"
    return Document(text=text, metadata=_build_metadata(path, root, source_type))


def load_documents(documents_dir: Path) -> list[Document]:
    documents: list[Document] = []

    for path in discover_source_files(documents_dir):
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path, documents_dir))
            continue

        text_document = _load_text(path, documents_dir)
        if text_document is not None:
            documents.append(text_document)

    return documents