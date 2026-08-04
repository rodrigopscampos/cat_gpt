from dataclasses import dataclass

from fastapi.testclient import TestClient

from openwebui import cli


@dataclass
class DummySource:
    citation: str
    score: float
    metadata: dict[str, object]


@dataclass
class DummyResult:
    answer: str
    sources: list[DummySource]


@dataclass
class DummyRetriever:
    sources: list[DummySource]

    def retrieve(self, query: str) -> list[DummySource]:
        return self.sources


@dataclass
class DummyConfig:
    conversation_turns: int = 2


class DummyPipeline:
    def __init__(self) -> None:
        self.config = DummyConfig()
        self.retriever = DummyRetriever(
            [DummySource(citation="guide.txt, p. 7", score=0.94, metadata={"document_name": "guide.txt"})]
        )
        self.last_question: str | None = None
        self.last_history: list[dict[str, str]] | None = None

    def answer(self, question: str, history: list[dict[str, str]]) -> DummyResult:
        self.last_question = question
        self.last_history = history
        return DummyResult(answer="Cats love routines.", sources=self.retriever.sources)


def test_chat_endpoint_returns_json_answer(monkeypatch) -> None:
    pipeline = DummyPipeline()
    monkeypatch.setattr(cli, "_build_pipeline", lambda: pipeline)
    client = TestClient(cli.app)

    response = client.post(
        "/api/chat",
        json={"question": "Do cats like routines?", "history": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == "Do cats like routines?"
    assert payload["answer"] == "Cats love routines."
    assert payload["sources"][0]["citation"] == "guide.txt, p. 7"
    assert pipeline.last_question == "Do cats like routines?"
    assert pipeline.last_history == [{"role": "user", "content": "Hi"}]


def test_root_does_not_serve_html_landing_page() -> None:
    client = TestClient(cli.app)

    response = client.get("/")

    assert response.status_code == 404
