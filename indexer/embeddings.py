from __future__ import annotations

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def build_embedding_model(model_name: str) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=model_name)