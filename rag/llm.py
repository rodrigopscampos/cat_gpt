from __future__ import annotations

from ollama import Client

from dataclasses import dataclass
import json
from typing import Iterator
from urllib import error, request

from rag.config import RuntimeConfig, load_runtime_config


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    temperature: float
    timeout: float

    @classmethod
    def from_runtime(cls, config: RuntimeConfig) -> "OllamaConfig":
        return cls(
            base_url=config.ollama_base_url,
            model=config.llm_model,
            temperature=config.temperature,
            timeout=config.request_timeout,
        )


class OllamaClient:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        runtime_config = load_runtime_config()
        self.config = config or OllamaConfig.from_runtime(runtime_config)

    def chat(self, messages: list[dict[str, str]]) -> str:
        # payload = {
        #     "model": self.config.model,
        #     "messages": messages,
        #     "stream": False,
        #     "options": {"temperature": self.config.temperature},
        # }
        # data = self._post_json("/api/chat", payload)
        # message = data.get("message") or {}
        # content = message.get("content", "")
        # if not isinstance(content, str):
        #     return str(content)
        # return content
        
        client = Client(host="http://localhost:11434")

        try:
            response = client.chat(
                model=self.config.model,
                messages=messages,
                stream=False,
                options={"temperature": self.config.temperature},
            )

            response_text = str(response["message"]["content"])
            print(response_text)
            return response_text
    
        except Exception as e:
            print(e)
            raise e

    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[str]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.config.temperature},
        }
        yield from self._stream_json_lines("/api/chat", payload)

    def _post_json(self, endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        url = f"{self.config.base_url.rstrip('/')}{endpoint}"
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout) as response:
                body = response.read().decode("utf-8")
        except error.URLError as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                f"Unable to reach Ollama at {self.config.base_url}. Start Ollama or update RAG_OLLAMA_BASE_URL."
            ) from exc

        return json.loads(body)

    def _stream_json_lines(self, endpoint: str, payload: dict[str, object]) -> Iterator[str]:
        url = f"{self.config.base_url.rstrip('/')}{endpoint}"
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    message = payload.get("message") or {}
                    chunk = message.get("content")
                    if isinstance(chunk, str) and chunk:
                        yield chunk
                    if payload.get("done"):
                        break
        except error.URLError as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                f"Unable to reach Ollama at {self.config.base_url}. Start Ollama or update RAG_OLLAMA_BASE_URL."
            ) from exc
