# `digital_me_rag`

**Status: ✅ Completed**

Internal ChromaDB RAG engine for the `digital_me` skill. This is **not a tool provider** — it has no `@agent_tool` functions and is not directly callable by agents. It is consumed exclusively by `digital_me`.

---

## Purpose

Provides a singleton `get_digital_me_rag()` that queries a ChromaDB vector collection with profile documents (resume, projects, skills). Returns ranked chunks with confidence scores.

---

## Internal Usage

```python
# Called internally by digital_me/tools.py
from kk_agent_skills.digital_me_rag import get_digital_me_rag

rag = get_digital_me_rag()
result = rag.query(
    question="What Python frameworks has KK worked with?",
    top_k=5,
    filter_metadata={"type": "skills"},
    min_confidence=0.1,
)

# result.has_results → bool
# result.chunks      → list of ranked text chunks
# result.confidence  → float (0.0–1.0)
# result.sources     → list of source document names
# result.message     → human-readable status
```

---

## Capability Declaration

```python
SkillManifest(
    name="digital_me_rag",
    capabilities=["rag_engine"],   # not "tool_provider"
)
```

Skills with `rag_engine` capability are excluded from tool registration and agent discovery. They are internal infrastructure only.

---

## ChromaDB Collection

| Setting | Value |
|---------|-------|
| Collection name | `digital_me` |
| Embedding model | Configured in `kk_utils.rag` |
| Similarity metric | Cosine |

---

## Dependencies

- `chromadb` — vector store
- `kk_utils.rag` — RAG base classes, embedding, query pipeline
