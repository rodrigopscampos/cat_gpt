from __future__ import annotations

from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int):
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    for index, node in enumerate(nodes):
        node.metadata["chunk_index"] = index
        node.metadata["chunk_id"] = node.metadata.get(
            "chunk_id",
            f"{node.metadata.get('relative_path', 'document')}::chunk-{index:05d}",
        )

    return nodes