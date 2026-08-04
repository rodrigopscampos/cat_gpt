from __future__ import annotations

# Suppress non-fatal Pydantic / User warnings during imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import time
import uuid
import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline

app = FastAPI(
    title="Cat GPT API",
    description="Local retrieval-augmented chat against the Chroma index in data/chroma.",
    version="1.0.0",
)


# --- Original Custom Models ---

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


# --- OpenAI Compatibility Models ---

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    stream: bool = False


# --- Core Logic ---

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the local FastAPI chat API.")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the local FastAPI chat API")
    serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind to")
    serve.add_argument("--port", type=int, default=8080, help="Port to bind to")
    serve.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser")

    return parser


def serve(host: str, port: int, no_browser: bool = False) -> None:
    if no_browser:
        print("The API no longer serves a browser UI. Swagger docs are available at /docs.")

    import uvicorn
    uvicorn.run(app, host=host, port=port)


def _format_history(messages: list[dict[str, Any]], turn_limit: int) -> list[dict[str, str]]:
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


# --- Original Route ---

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    pipeline = RAGPipeline()
    history = [message.model_dump() for message in request.history]

    result = pipeline.answer(question, history=_format_history(history, pipeline.config.conversation_turns))

    return ChatResponse(
        question=question,
        answer=result.answer,
        sources=[
            SourcePayload(citation=source.citation, score=source.score, metadata=source.metadata)
            for source in result.sources
        ],
    )


# --- OpenAI Compatible Routes for Open WebUI ---

@app.get("/v1/models")
def get_models():
    """Tells Open WebUI what models are available."""
    return {
        "object": "list",
        "data": [
            {
                "id": "cat-gpt-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    }

@app.post("/v1/chat/completions")
def chat_completions(request: OpenAIChatRequest):
    """Handles standard chat requests from Open WebUI."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Extract the full history to map to the RAGPipeline format
    history_msgs = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # The final message is treated as the current question
    question = history_msgs.pop()["content"] if history_msgs else ""

    pipeline = RAGPipeline()
    result = pipeline.answer(question, history=_format_history(history_msgs, pipeline.config.conversation_turns))

    # Open WebUI standard UI doesn't natively parse custom payload blocks, 
    # so we neatly format your RAG sources at the end of the text response.
    final_answer = result.answer
    if result.sources:
        final_answer += "\n\n---\n**Sources:**\n"
        for s in result.sources:
            final_answer += f"- {s.citation} (Score: {s.score:.2f})\n"

    response_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Handle Streaming (Open WebUI prefers streaming SSE format)
    if request.stream:
        def generate():
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": final_answer}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    # Handle standard non-streaming response
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_answer
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command != "serve":
        parser.print_help()
        return

    serve(args.host, args.port, no_browser=args.no_browser)


if __name__ == "__main__":
    main()