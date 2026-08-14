from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag.config import Config
Embedding = HuggingFaceEmbedding(Config.embedding_model)