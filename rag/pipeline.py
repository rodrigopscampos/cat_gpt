from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag.config import RuntimeConfig, load_runtime_config
from rag.llm import OllamaClient, OllamaConfig
from rag.prompt import build_messages, render_answer
from rag.retriever import RagRetriever, RetrievedChunk


@dataclass(frozen=True)
class RAGResult:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    context: str


class RAGPipeline:
    def __init__(
        self,
        config: RuntimeConfig | None = None,
        retriever: RagRetriever | None = None,
        llm: OllamaClient | None = None,
    ) -> None:
        self.config = config or load_runtime_config()
        self.retriever = retriever or RagRetriever(config=self.config)
        self.llm = llm or OllamaClient(config=OllamaConfig.from_runtime(self.config))

    def answer(self, question: str, history: list[dict[str, str]] | None = None) -> RAGResult:
        sources = self.retriever.retrieve(question)
        prompt = build_messages(question=question, sources=sources, history=history, config=self.config)
        answer = self.llm.chat(prompt.messages)
        return RAGResult(question=question, answer=render_answer(answer, sources), sources=sources, context=prompt.context)

    def answer_stream(self, question: str, history: list[dict[str, str]] | None = None):
        sources = self.retriever.retrieve(question)
        prompt = build_messages(question=question, sources=sources, history=history, config=self.config)
        yield from self.llm.stream_chat(prompt.messages)


def format_history(messages: list[dict[str, Any]], turn_limit: int) -> list[dict[str, str]]:
    if turn_limit <= 0:
        return []

    history: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        history.append({"role": role, "content": content})
    return history[-turn_limit * 2 :]


def ask_question(question: str, history: list[dict[str, str]] | None = None) -> str:
    pipeline = RAGPipeline()
    return pipeline.answer(question, history=history).answer
