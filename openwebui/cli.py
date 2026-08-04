from __future__ import annotations

import argparse
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.prompt import render_answer
from rag.pipeline import RAGPipeline, format_history


app = FastAPI(
    title="Cat GPT API",
    description="Local retrieval-augmented chat against the Chroma index in data/chroma.",
    version="1.0.0",
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)


class SourcePayload(BaseModel):
    citation: str
    score: float
    metadata: dict[str, Any]


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourcePayload]
    warning: str | None = None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the local FastAPI chat API.")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the local FastAPI chat API")
    serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind to")
    serve.add_argument("--port", type=int, default=8080, help="Port to bind to")
    serve.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser")

    return parser


def _build_pipeline() -> RAGPipeline:
    return RAGPipeline()


def serve(host: str, port: int, no_browser: bool = False) -> None:
    if no_browser:
        print("The API no longer serves a browser UI. Swagger docs are available at /docs.")

    import uvicorn

    uvicorn.run(app, host=host, port=port)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    pipeline = _build_pipeline()
    history = [message.model_dump() for message in request.history]

    try:
        result = pipeline.answer(question, history=format_history(history, pipeline.config.conversation_turns))
    except Exception as exc:  # pragma: no cover - local service/runtime dependent
        fallback_sources = pipeline.retriever.retrieve(question)
        fallback_answer = (
            "I could retrieve relevant local sources, but I couldn't reach Ollama to generate a full answer. "
            "Start Ollama and try again for a synthesized response."
        )
        return ChatResponse(
            question=question,
            answer=render_answer(fallback_answer, fallback_sources),
            sources=[
                SourcePayload(citation=source.citation, score=source.score, metadata=source.metadata)
                for source in fallback_sources
            ],
            warning=str(exc),
        )

    return ChatResponse(
        question=question,
        answer=result.answer,
        sources=[
            SourcePayload(citation=source.citation, score=source.score, metadata=source.metadata)
            for source in result.sources
        ],
    )


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command != "serve":
        parser.print_help()
        return

    serve(args.host, args.port, no_browser=args.no_browser)