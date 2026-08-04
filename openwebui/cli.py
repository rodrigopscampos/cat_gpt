from __future__ import annotations

import argparse
import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import webbrowser

from rag.prompt import render_answer
from rag.pipeline import RAGPipeline, format_history


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cat GPT Q&A</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
    main {{ max-width: 960px; margin: 0 auto; padding: 24px; }}
    .card {{ background: #111827; border: 1px solid #334155; border-radius: 16px; padding: 16px; box-shadow: 0 20px 40px rgba(0,0,0,.2); }}
    h1 {{ margin: 0 0 12px; font-size: 1.6rem; }}
    p.subtle {{ color: #94a3b8; margin-top: 0; }}
    textarea {{ width: 100%; min-height: 120px; resize: vertical; border-radius: 12px; border: 1px solid #475569; background: #0b1220; color: #e2e8f0; padding: 12px; box-sizing: border-box; }}
    button {{ background: #38bdf8; color: #082f49; border: none; border-radius: 999px; padding: 10px 18px; font-weight: 700; cursor: pointer; margin-top: 12px; }}
    button:disabled {{ opacity: .6; cursor: wait; }}
    .output {{ white-space: pre-wrap; line-height: 1.55; margin-top: 16px; background: #0b1220; border: 1px solid #334155; border-radius: 12px; padding: 14px; min-height: 120px; }}
    .row {{ display: grid; gap: 16px; }}
    .sources {{ margin-top: 12px; color: #cbd5e1; }}
    .tag {{ display: inline-block; margin-right: 8px; padding: 4px 10px; border-radius: 999px; background: #1d4ed8; color: white; font-size: .82rem; }}
  </style>
</head>
<body>
  <main>
    <div class="card">
      <h1>Cat GPT Q&A</h1>
      <p class="subtle">Local retrieval-augmented chat against the Chroma index in <code>data/chroma</code>.</p>
      <div class="row">
        <label>
          <span class="tag">Question</span>
          <textarea id="question" placeholder="Ask about the indexed documents..."></textarea>
        </label>
        <div>
          <button id="ask">Ask</button>
        </div>
        <div class="output" id="output">Your answer will appear here.</div>
      </div>
    </div>
  </main>
  <script>
    const button = document.getElementById('ask');
    const question = document.getElementById('question');
    const output = document.getElementById('output');

    async function ask() {{
      const text = question.value.trim();
      if (!text) {{
        output.textContent = 'Please enter a question first.';
        return;
      }}

      button.disabled = true;
      output.textContent = 'Thinking...';

      try {{
        const response = await fetch('/api/chat', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ question: text, history: [] }})
        }});
        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.error || 'Request failed');
        }}

        output.textContent = data.answer;
      }} catch (error) {{
        output.textContent = 'Error: ' + error.message;
      }} finally {{
        button.disabled = false;
      }}
    }}

    button.addEventListener('click', ask);
    question.addEventListener('keydown', (event) => {{
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {{
        ask();
      }}
    }});
  </script>
</body>
</html>
"""


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the local Q&A UI.")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the local browser-based Q&A UI")
    serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind to")
    serve.add_argument("--port", type=int, default=8080, help="Port to bind to")
    serve.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser")

    return parser


def _build_pipeline() -> RAGPipeline:
    return RAGPipeline()


class _Handler(BaseHTTPRequestHandler):
    server_version = "CatGPTQAServer/1.0"

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            body = HTML_TEMPLATE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        payload = self._read_json_body()
        question = str(payload.get("question") or "").strip()
        if not question:
            self._send_json({"error": "Missing question"}, status=HTTPStatus.BAD_REQUEST)
            return

        history = payload.get("history") or []
        if not isinstance(history, list):
            history = []

        pipeline = _build_pipeline()
        try:
            result = pipeline.answer(question, history=format_history(history, pipeline.config.conversation_turns))
        except Exception as exc:  # pragma: no cover - local service/runtime dependent
            fallback_sources = pipeline.retriever.retrieve(question)
            fallback_answer = (
                "I could retrieve relevant local sources, but I couldn't reach Ollama to generate a full answer. "
                "Start Ollama and try again for a synthesized response."
            )
            self._send_json(
                {
                    "question": question,
                    "answer": render_answer(fallback_answer, fallback_sources),
                    "sources": [
                        {
                            "citation": source.citation,
                            "score": source.score,
                            "metadata": source.metadata,
                        }
                        for source in fallback_sources
                    ],
                    "warning": str(exc),
                }
            )
            return

        self._send_json(
            {
                "question": question,
                "answer": result.answer,
                "sources": [
                    {
                        "citation": source.citation,
                        "score": source.score,
                        "metadata": source.metadata,
                    }
                    for source in result.sources
                ],
            }
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        message = format % args
        print(f"[{self.log_date_time_string()}] {self.address_string()} {message}")


def serve(host: str, port: int, no_browser: bool = False) -> None:
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    print(f"Cat GPT Q&A UI listening on {url}")
    if not no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command != "serve":
        parser.print_help()
        return

    serve(args.host, args.port, no_browser=args.no_browser)