"""Online RAG components for the local question-answering pipeline."""

from rag.config import RuntimeConfig, load_runtime_config
from rag.pipeline import RAGPipeline, RAGResult, ask_question, format_history
from rag.prompt import PromptBundle, build_messages, format_sources_markdown, render_answer
from rag.retriever import RagRetriever, RetrievedChunk, format_citation

__all__ = [
	"PromptBundle",
	"RAGPipeline",
	"RAGResult",
	"RagRetriever",
	"RetrievedChunk",
	"RuntimeConfig",
	"ask_question",
	"build_messages",
	"format_citation",
	"format_history",
	"format_sources_markdown",
	"load_runtime_config",
	"render_answer",
]
