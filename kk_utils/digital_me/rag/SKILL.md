---
name: digital_me_rag
description: Digital Me RAG engine — extends kk_utils RAGEngine with user-scoped security filtering, document-type auto-detection, and resume/project search helpers. Provides get_digital_me_rag() singleton used by the digital_me skill tools.
version: 1.0.0
dependencies: kk-utils[rag], chromadb, sentence-transformers
capabilities: rag_engine
tags: digital_me, rag, chromadb, resume, profile
metadata:
  author: personal-assistant
  collection_name: digital_me
---

# digital_me_rag

RAG engine subclass for the Digital Me knowledge base.

Extends `kk_utils.rag.RAGEngine` with:
- **User-scoped security filtering** — injects `user_id` into every query and document
- **Document-type auto-detection** — tags uploaded docs as resume, cover_letter, or document
- **Convenience search methods** — `search_resume()`, `search_projects()`
- **Singleton factory** — `get_digital_me_rag()` for shared instance

## Usage

```python
from kk_agent_skills.digital_me_rag import get_digital_me_rag

rag = get_digital_me_rag()
result = rag.query("What is my work experience?", user_id="user_123")
```

## Config

Reads from `RAG_CONFIG_PATH` env var → `config/rag_config.yaml` → defaults.
Collection name: `digital_me`
