# RAG Service - Unified Architecture

**Version:** 1.0  
**Date:** 2026-03-05  
**Status:** Core Library Complete

---

## Overview

The **RAGService** in kk-utils provides a unified interface for all RAG operations, ensuring consistency across all interfaces.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  RAGService (kk-utils)                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  upload_document()                                       │
│  ├── PostgreSQL (metadata)                               │
│  └── ChromaDB (chunks + embeddings)                      │
│                                                          │
│  delete_document()                                       │
│  ├── PostgreSQL (remove metadata)                        │
│  └── ChromaDB (remove chunks)                            │
│                                                          │
│  search()                                                │
│  └── ChromaDB (vector search)                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Usage

### All Interfaces Use the Same Code

#### 1. Document Manager (Gradio)
```python
from kk_utils.rag import RAGService

service = RAGService()
result = service.upload_document(
    file_path="/path/to/file.pdf",
    user_id="demo_user",
    classification="PUBLIC"
)
```

#### 2. Bulk Import Script
```python
from kk_utils.rag import RAGService

service = RAGService()
result = service.upload_document(
    file_path="/path/to/file.md",
    user_id="bulk_import",
    classification="PUBLIC"
)
```

#### 3. RAG Dashboard (React)
```javascript
// Calls backend API which uses RAGService
await apiClient.post('/api/v1/documents/upload', formData);
```

#### 4. AgentMe Chat
```python
# Searches ChromaDB via RAGService
service = RAGService()
results = service.search("What is my experience?")
```

---

## Benefits

### ✅ Consistency
- All interfaces use the same code
- Same data structure
- Same validation rules

### ✅ Data Integrity
- PostgreSQL + ChromaDB always in sync
- Transaction-like behavior
- Rollback on failure

### ✅ Maintainability
- Single source of truth
- Easy to update logic
- Consistent error handling

### ✅ Visibility
- Document Manager shows ALL documents
- Bulk imports visible in UI
- Unified statistics

---

## Database Roles

### PostgreSQL
**Stores:**
- Document metadata (filename, size, type)
- User information
- Upload timestamps
- File hashes
- Classification levels
- Subscription data

**Purpose:**
- Fast structured queries
- User management
- Access control
- Audit logging

### ChromaDB
**Stores:**
- Document chunks
- Vector embeddings
- Chunk metadata
- Similarity indices

**Purpose:**
- Semantic search
- Similarity matching
- RAG retrieval

---

## Import Scripts

### Old Approach (Deprecated)
```python
# Direct ChromaDB only
rag = RAGEngine()
rag.add_document(doc_id, text, metadata)
# ❌ PostgreSQL not updated
# ❌ Document Manager can't see it
```

### New Approach (Recommended)
```python
# RAGService - Both databases
service = RAGService()
result = service.upload_document(file_path, user_id)
# ✅ PostgreSQL updated
# ✅ ChromaDB updated
# ✅ Document Manager shows it
```

---

## Files

### Core Library
- `kk-utils/kk_utils/rag/rag_service.py` - Unified service
- `kk-utils/kk_utils/rag/rag_engine.py` - ChromaDB engine
- `kk-utils/kk_utils/rag/__init__.py` - Exports

### Import Scripts
- `import_documents.py` - New script using RAGService ✅
- `import_to_rag_standalone.py` - Old script (ChromaDB only) ⚠️

### Backend API
- `backend/app/api/v1/documents.py` - Document upload
- `backend/app/api/v1/rag_collections.py` - RAG collections

### Frontend
- `rag-dashboard/` - React dashboard
- `gradio-apps/document_manager.py` - Document Manager UI
- `gradio-apps/agent_me_chat.py` - AgentMe Chat

---

## Migration Path

### Phase 1 (Current) ✅
- RAGService created
- Bulk import script updated
- All new uploads use RAGService

### Phase 2 (Next)
- Update existing document upload API to use RAGService
- Migrate old ChromaDB-only data to PostgreSQL
- Full integration complete

### Phase 3 (Future)
- Add update/reindex functionality
- Add batch operations
- Add caching layer

---

## Best Practices

### 1. Always Use RAGService
```python
# ✅ Good
from kk_utils.rag import RAGService
service = RAGService()
service.upload_document(...)

# ❌ Bad - bypasses PostgreSQL
from kk_utils.rag import RAGEngine
rag = RAGEngine()
rag.add_document(...)
```

### 2. Handle Errors Gracefully
```python
try:
    result = service.upload_document(file_path, user_id)
    if result.get('success'):
        logger.info(f"Uploaded: {result['doc_id']}")
    else:
        logger.error(f"Failed: {result.get('error')}")
except Exception as e:
    logger.error(f"Upload failed: {e}")
```

### 3. Close Resources
```python
service = RAGService()
try:
    # ... use service ...
finally:
    service.close()  # Cleanup connections
```

### 4. Use Consistent Metadata
```python
metadata = {
    "source": "bulk_import",  # or "document_manager", "api", etc.
    "upload_date": datetime.now().isoformat(),
    "user_id": user_id,
    "classification": "PUBLIC",
}
```

---

## Troubleshooting

### Issue: Document Manager doesn't show bulk-imported files

**Cause:** Used old import script (ChromaDB only)

**Solution:**
```bash
# Re-import using new script
cd rag-input-data/agent_me_chat/personas/chua_keng_koon
python import_documents.py
```

### Issue: PostgreSQL connection failed

**Cause:** DATABASE_URL not set or wrong

**Solution:**
```bash
# Set DATABASE_URL in .env
DATABASE_URL=postgresql://user:pass@localhost:5432/personal_assistant
# or for SQLite (development)
DATABASE_URL=sqlite:///./app.db
```

### Issue: ChromaDB not found

**Cause:** ChromaDB not installed

**Solution:**
```bash
pip install chromadb
```

---

## Testing

### Test Upload
```python
from kk_utils.rag import RAGService

service = RAGService()

# Upload test file
result = service.upload_document(
    file_path="test.md",
    user_id="test_user",
    classification="PUBLIC"
)

print(f"Uploaded: {result['doc_id']}")
print(f"Chunks: {result['chunks_added']}")

service.close()
```

### Test Search
```python
from kk_utils.rag import RAGService

service = RAGService()

# Search
results = service.search("test query", top_k=5)

for chunk in results:
    print(f"Score: {chunk.get('score', 0):.2f}")
    print(f"Text: {chunk['text'][:100]}...")

service.close()
```

### Test Statistics
```python
from kk_utils.rag import RAGService

service = RAGService()
stats = service.get_statistics()

print(f"Collections: {stats['total_collections']}")
print(f"Documents: {stats['total_documents']}")
print(f"Chunks: {stats['total_chunks']}")

service.close()
```

---

## Summary

**Before:**
```
Document Manager → PostgreSQL
Bulk Import → ChromaDB
❌ Not connected
```

**After:**
```
All Interfaces → RAGService → PostgreSQL + ChromaDB
✅ Unified
✅ Consistent
✅ Complete
```

---

**End of RAG Service Documentation**
