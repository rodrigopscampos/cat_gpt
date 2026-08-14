from __future__ import annotations
import json
import re
from pathlib import Path
from llama_index.core import Document

SUPPORTED_EXTENSIONS = {".txt", ".json"}

def _discover_source_files(documents_dir: Path) -> list[Path]:
    if not documents_dir.exists():
        return []

    return sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _normalize_text(text: str) -> str:
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


def _load_text(path: Path, root: Path) -> Document | None:
    text = _normalize_text(path.read_text(encoding="utf-8", errors="replace"))
    if not text:
        return None

    source_type = "markdown" if path.suffix.lower() in {".md", ".markdown"} else "txt"
    return Document(text=text, metadata=_build_metadata(path, root, source_type))


def _coerce_json_records(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]

    if isinstance(payload, dict):
        for key in ("items", "documents", "chunks", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
        return [payload]

    return []


def _extract_json_text(record: dict[str, object]) -> str:
    preferred_keys = ("texto_embedding", "texto", "content", "body", "description")
    for key in preferred_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_text(value)

    parts: list[str] = []
    for key in ("titulo", "secao", "fonte", "url"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}: {value.strip()}")

    for key, value in record.items():
        if key in {"id", "titulo", "secao", "fonte", "url", "tipo", "texto", "texto_embedding"}:
            continue
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}: {value.strip()}")

    return _normalize_text("\n".join(parts))


def _load_json(path: Path, root: Path) -> list[Document]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        print(f"Skipped invalid JSON file: {path} ({exc})")
        return []

    documents: list[Document] = []
    for index, record in enumerate(_coerce_json_records(payload)):
        text = _extract_json_text(record)

        if not text:
            continue

        metadata = _build_metadata(path, root, "json")
        metadata["json_record_index"] = index

        for key in ("id", "titulo", "secao", "fonte", "url", "tipo"):
            value = record.get(key, '')
            metadata[key] = str(value).strip()

        document = Document(text=text, metadata=metadata)
        documents.append(document)
        
        print(f"Loaded JSON record: {path} [{index}] ({len(document.text)} chars)")

    return documents


def load_documents(documents_dir: Path) -> list[Document]:
    documents: list[Document] = []

    for path in _discover_source_files(documents_dir):
        if path.suffix.lower() == ".json":
            documents.extend(_load_json(path, documents_dir))
            continue

        text_document = _load_text(path, documents_dir)
        if text_document is not None:
            documents.append(text_document)
            print(f"Loaded document: {path} ({len(text_document.text)} chars)")

    return documents
