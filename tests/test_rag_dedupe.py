from kk_utils.rag.dedupe import apply_cleanup_plan, build_cleanup_plan
from kk_utils.rag.rag_engine import RAGEngine


class FakeCollection:
    def __init__(self):
        self.ids = [
            "old_0", "old_1",
            "new_0", "new_1",
            "unique_0",
        ]
        self.documents = ["alpha", "beta", "alpha", "beta", "gamma"]
        self.metadatas = [
            {"doc_id": "old", "chunk_index": 0, "uploaded_at": "2026-05-19"},
            {"doc_id": "old", "chunk_index": 1, "uploaded_at": "2026-05-19"},
            {"doc_id": "new", "chunk_index": 0, "uploaded_at": "2026-05-25"},
            {"doc_id": "new", "chunk_index": 1, "uploaded_at": "2026-05-25"},
            {"doc_id": "unique", "chunk_index": 0, "uploaded_at": "2026-05-20"},
        ]
        self.deleted = []
        self.updated = []

    def get(self, **kwargs):
        return {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }

    def delete(self, ids):
        self.deleted.extend(ids)

    def update(self, ids, metadatas):
        self.updated.append((ids, metadatas))


def test_cleanup_plan_keeps_newest_complete_document():
    collection = FakeCollection()
    plan = build_cleanup_plan(collection)
    assert plan["documents_to_remove"] == 1
    assert plan["chunks_to_remove"] == 2
    assert plan["duplicate_groups"][0]["keep"]["doc_id"] == "new"
    assert plan["duplicate_groups"][0]["remove"][0]["doc_id"] == "old"

    result = apply_cleanup_plan(collection, plan)
    assert result["chunks_removed"] == 2
    assert collection.deleted == ["old_0", "old_1"]
    assert collection.updated[0][0] == ["new_0", "new_1"]
    assert all(
        metadata.get("chunk_fingerprint")
        for metadata in collection.updated[0][1]
    )


def test_add_document_rejects_existing_chunk_fingerprint():
    class DuplicateCollection:
        def get(self, where, **kwargs):
            if "chunk_fingerprint" in where:
                return {
                    "ids": ["existing_chunk_0"],
                    "metadatas": [{"doc_id": "existing_doc"}],
                }
            return {"ids": [], "metadatas": []}

        def count(self):
            return 3

    engine = RAGEngine.__new__(RAGEngine)
    engine.collection = DuplicateCollection()
    engine.chunker = type("Chunker", (), {"chunk": lambda self, text: ["one", "two"]})()

    result = engine.add_document("new_doc", "same text")
    assert result["duplicate"] is True
    assert result["existing_doc_id"] == "existing_doc"
    assert result["chunks_added"] == 0
