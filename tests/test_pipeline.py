from dataclasses import dataclass

from rag.config import RuntimeConfig
from rag.pipeline import RAGPipeline
from rag.retriever import RetrievedChunk


@dataclass
class DummyRetriever:
    sources: list[RetrievedChunk]

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return self.sources


@dataclass
class DummyLLM:
    response: str
    last_messages: list[dict[str, str]] | None = None

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        return self.response


def test_pipeline_answers_with_retrieved_sources() -> None:
    source = RetrievedChunk(
        node_id="node-1",
        text="Cats thrive on predictable routines.",
        score=0.94,
        metadata={"document_name": "guide.txt", "page": 7},
        citation="guide.txt, p. 7",
    )
    config = RuntimeConfig(
        chroma_dir=RuntimeConfig.from_env().chroma_dir,
        collection_name="rag_documents",
        embedding_model="BAAI/bge-m3",
        llm_model="gemma3:1b",
        ollama_base_url="http://localhost:11434",
        top_k=5,
        max_context_chars=500,
        conversation_turns=2,
        temperature=0.2,
        request_timeout=30,
    )
    llm = DummyLLM(response="Cats benefit from routine.")
    pipeline = RAGPipeline(config=config, retriever=DummyRetriever([source]), llm=llm)

    result = pipeline.answer("Do cats like routines?")

    assert "Cats benefit from routine." in result.answer
    assert "## Sources" in result.answer
    assert result.sources[0].citation == "guide.txt, p. 7"
    assert llm.last_messages is not None
    assert llm.last_messages[-1]["role"] == "user"