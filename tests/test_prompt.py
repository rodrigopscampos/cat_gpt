from rag.config import RuntimeConfig
from rag.prompt import build_messages, render_answer
from rag.retriever import RetrievedChunk


def test_build_messages_includes_context_and_question() -> None:
    sources = [
        RetrievedChunk(
            node_id="node-1",
            text="Cats need regular wellness care.",
            score=0.91,
            metadata={"document_name": "guide.txt", "page": 2},
            citation="guide.txt, p. 2",
        )
    ]
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

    bundle = build_messages("What do cats need?", sources=sources, history=[{"role": "user", "content": "Hi"}], config=config)

    assert bundle.messages[0]["role"] == "system"
    assert "Cats need regular wellness care." in bundle.messages[-1]["content"]
    assert "What do cats need?" in bundle.messages[-1]["content"]
    assert bundle.context


def test_render_answer_appends_sources() -> None:
    sources = [
        RetrievedChunk(
            node_id="node-1",
            text="Cats need regular wellness care.",
            score=0.91,
            metadata={"document_name": "guide.txt", "page": 2},
            citation="guide.txt, p. 2",
        )
    ]

    rendered = render_answer("Yes, they do.", sources)

    assert "Yes, they do." in rendered
    assert "## Sources" in rendered
    assert "guide.txt, p. 2" in rendered