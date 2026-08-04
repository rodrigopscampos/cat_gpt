from __future__ import annotations

from dataclasses import dataclass

from rag.config import RuntimeConfig
from rag.retriever import RetrievedChunk


@dataclass(frozen=True)
class PromptBundle:
    messages: list[dict[str, str]]
    context: str


SYSTEM_PROMPT = (
    "You are a local retrieval-augmented assistant. Answer the user's question using only the provided "
    "context when possible. If the context does not contain enough information, say so clearly and avoid "
    "inventing details. Prefer concise, direct answers. Always mention useful citations from the sources section "
    "when available."
)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _format_history(history: list[dict[str, str]], turn_limit: int) -> list[dict[str, str]]:
    if turn_limit <= 0:
        return []

    filtered = [message for message in history if message.get("role") in {"user", "assistant"}]
    return filtered[-turn_limit * 2 :]


def build_context_block(sources: list[RetrievedChunk], max_context_chars: int) -> str:
    if not sources:
        return ""

    sections: list[str] = []
    remaining = max_context_chars

    for index, source in enumerate(sources, start=1):
        header = f"[Source {index}] {source.citation}"
        body = _truncate(source.text.strip(), max(0, remaining - len(header) - 3))
        section = f"{header}\n{body}".strip()
        if not section:
            continue

        section_length = len(section)
        if sections and section_length > remaining:
            break

        sections.append(section)
        remaining -= section_length
        if remaining <= 0:
            break

    return "\n\n".join(sections)


def build_messages(
    question: str,
    sources: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
    config: RuntimeConfig | None = None,
) -> PromptBundle:
    config = config or RuntimeConfig.from_env()
    context = build_context_block(sources, config.max_context_chars)

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(_format_history(history, config.conversation_turns))

    if context:
        user_prompt = (
            "Use the sources below to answer the question. Keep the answer grounded in the sources and cite them "
            "briefly when possible.\n\n"
            f"Sources:\n{context}\n\n"
            f"Question: {question}"
        )
    else:
        user_prompt = (
            "No retrieved sources were available. If you answer, be explicit that the collection did not return "
            "supporting context.\n\n"
            f"Question: {question}"
        )

    messages.append({"role": "user", "content": user_prompt})
    return PromptBundle(messages=messages, context=context)


def format_sources_markdown(sources: list[RetrievedChunk]) -> str:
    if not sources:
        return "No supporting sources were retrieved."

    lines = ["## Sources"]
    for index, source in enumerate(sources, start=1):
        lines.append(f"- **Source {index}** — {source.citation}")
    return "\n".join(lines)


def render_answer(answer: str, sources: list[RetrievedChunk]) -> str:
    answer = answer.strip()
    sources_block = format_sources_markdown(sources)
    if answer:
        return f"{answer}\n\n{sources_block}"
    return sources_block
