from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rag.config import RuntimeConfig
from rag.pipeline import RAGPipeline, format_history


class Valves(BaseModel):
	chroma_dir: str | None = None
	collection_name: str | None = None
	embedding_model: str | None = None
	llm_model: str | None = None
	ollama_base_url: str | None = None
	top_k: int | None = None
	max_context_chars: int | None = None
	conversation_turns: int | None = None
	temperature: float | None = None
	request_timeout: float | None = None


class Pipeline:
	id = "cat-gpt-rag"
	name = "Cat GPT RAG"

	def __init__(self) -> None:
		self.valves = Valves()
		self._pipeline: RAGPipeline | None = None

	def _runtime_config(self) -> RuntimeConfig:
		config = RuntimeConfig.from_env()
		valves = self.valves
		return RuntimeConfig(
			chroma_dir=config.chroma_dir if valves.chroma_dir is None else Path(valves.chroma_dir),
			collection_name=valves.collection_name or config.collection_name,
			embedding_model=valves.embedding_model or config.embedding_model,
			llm_model=valves.llm_model or config.llm_model,
			ollama_base_url=valves.ollama_base_url or config.ollama_base_url,
			top_k=valves.top_k or config.top_k,
			max_context_chars=valves.max_context_chars or config.max_context_chars,
			conversation_turns=valves.conversation_turns or config.conversation_turns,
			temperature=valves.temperature or config.temperature,
			request_timeout=valves.request_timeout or config.request_timeout,
		)

	def _get_pipeline(self) -> RAGPipeline:
		if self._pipeline is None:
			self._pipeline = RAGPipeline(config=self._runtime_config())
		return self._pipeline

	def _extract_question(self, body: dict[str, Any]) -> str:
		messages = body.get("messages") or []
		for message in reversed(messages):
			if message.get("role") == "user":
				content = message.get("content")
				if isinstance(content, str) and content.strip():
					return content.strip()
		raise ValueError("No user question was provided in the Open WebUI payload.")

	def _extract_history(self, body: dict[str, Any]) -> list[dict[str, str]]:
		messages = body.get("messages") or []
		history = format_history(messages, turn_limit=self._get_pipeline().config.conversation_turns)
		if history and history[-1].get("role") == "user":
			return history[:-1]
		return history

	def pipe(self, body: dict[str, Any], user: dict[str, Any] | None = None) -> str:
		pipeline = self._get_pipeline()
		question = self._extract_question(body)
		history = self._extract_history(body)

		try:
			result = pipeline.answer(question, history=history)
			return result.answer
		except Exception as exc:  # pragma: no cover - depends on local services
			return (
				"I couldn't complete the local RAG request. "
				f"Please check that Chroma and Ollama are running. Details: {exc}"
			)


class Pipe(Pipeline):
	pass
