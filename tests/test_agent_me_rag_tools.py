from types import SimpleNamespace

from kk_utils.agents.adapters.agent_me import tools


def test_search_digital_me_deduplicates_and_preserves_scores(monkeypatch):
    duplicate = {
        "text": "same content",
        "relevance_score": 0.8,
        "distance": 0.4,
        "metadata": {"doc_id": "a", "source_file": "resume.md"},
    }
    unique = {
        "text": "different content",
        "relevance_score": 0.6,
        "distance": 0.8,
        "metadata": {"doc_id": "b", "source_file": "work.md"},
    }
    result = SimpleNamespace(
        chunks=[duplicate, dict(duplicate), unique],
        has_results=True,
        confidence=0.7,
        sources=["a", "b"],
        message="ok",
        retrieval_time_ms=12.5,
        chunks_searched=36,
        avg_distance=0.6,
    )

    class FakeRAGEngine:
        def __init__(self, collection_name):
            self.collection_name = collection_name

        def query(self, **kwargs):
            return result

    monkeypatch.setattr("kk_utils.rag.rag_engine.RAGEngine", FakeRAGEngine)
    output = tools.search_digital_me(
        "work experience",
        top_k=5,
        user_id="demo_user",
        persona_collection="persona_kengkoon",
    )

    assert output["collection_name"] == "persona_kengkoon"
    assert len(output["chunks"]) == 2
    assert output["confidence"] == 0.7
    assert output["chunks"][0]["relevance_score"] == 0.8
    assert output["chunks_searched"] == 36


def test_work_experience_preserves_rag_metrics(monkeypatch):
    monkeypatch.setattr(
        tools,
        "search_digital_me",
        lambda **kwargs: {
            "confidence": 0.7,
            "chunks": [{"content": "experience"}],
            "sources": ["doc-1"],
            "retrieval_time_ms": 15.0,
            "chunks_searched": 36,
            "avg_distance": 0.6,
            "collection_name": "persona_kengkoon",
        },
    )
    output = tools.get_work_experience(
        user_id="demo_user",
        persona_collection="persona_kengkoon",
    )
    assert output["source"] == "rag"
    assert output["retrieval_time_ms"] == 15.0
    assert output["chunks_searched"] == 36
    assert output["avg_distance"] == 0.6
    assert output["collection_name"] == "persona_kengkoon"
