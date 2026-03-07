# KK-Utils RAG Module - Usage Examples

**Version:** 1.0  
**Package:** `kk_utils.rag`

---

## Overview

The RAG module in kk-utils provides a **reusable, standalone RAG engine** that works directly with ChromaDB - no backend API required.

---

## Installation

```bash
# Install kk-utils
cd kk-utils
pip install -e .

# Install ChromaDB
pip install chromadb
```

---

## Quick Start

### Basic Usage

```python
from kk_utils.rag import RAGEngine, RAGConfig

# Create RAG engine
rag = RAGEngine(collection_name="my_knowledge_base")

# Add a document
rag.add_document(
    doc_id="doc_001",
    text="Your document text here...",
    metadata={"type": "manual", "category": "guide"}
)

# Query the knowledge base
results = rag.query("How do I...?")

print(f"Confidence: {results.confidence:.2f}")
for chunk in results.chunks:
    print(f"- {chunk['text'][:100]}...")
```

---

## Import Documents (Bulk Import)

### Example: Import Markdown Files

```python
from kk_utils.rag import RAGEngine, RAGConfig, SentenceChunker
from pathlib import Path

# Create RAG engine with custom config
config = RAGConfig(
    chunking={
        "strategy": "sentence",
        "chunk_size": 500,
        "chunk_overlap": 50
    }
)

rag = RAGEngine(collection_name="portfolio", config=config)

# Import markdown files
docs_folder = Path("rag-documents")
for md_file in docs_folder.glob("*.md"):
    print(f"Importing: {md_file.name}")
    
    # Read file
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Chunk document
    chunker = SentenceChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk(content)
    
    # Add chunks to RAG
    for i, chunk in enumerate(chunks):
        doc_id = f"{md_file.stem}_chunk_{i}"
        metadata = {
            "source_file": md_file.name,
            "chunk_index": i,
            "type": "portfolio"
        }
        
        rag.add_document(doc_id=doc_id, text=chunk, metadata=metadata)
    
    print(f"  ✅ Added {len(chunks)} chunks")

# Show stats
stats = rag.get_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Total chunks: {stats['total_chunks']}")
```

---

## Chunking Strategies

### Word-based Chunking (Default)

```python
from kk_utils.rag import RAGEngine, RAGConfig

config = RAGConfig(
    chunking={
        "strategy": "word",
        "chunk_size": 500,      # words per chunk
        "chunk_overlap": 50     # overlapping words
    }
)

rag = RAGEngine(collection_name="my_kb", config=config)
```

**Best for:** General documents, fast processing

---

### Sentence-based Chunking

```python
from kk_utils.rag import RAGEngine, RAGConfig, SentenceChunker

config = RAGConfig(
    chunking={
        "strategy": "sentence",
        "chunk_size": 500,
        "chunk_overlap": 50
    }
)

rag = RAGEngine(collection_name="my_kb", config=config)

# Or use chunker directly
chunker = SentenceChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk(long_text)
```

**Best for:** Technical docs, better coherence

---

### Semantic Chunking (Advanced)

```python
from kk_utils.rag import RAGEngine, RAGConfig

config = RAGConfig(
    chunking={
        "strategy": "semantic"  # Uses AI to find topic boundaries
    }
)

rag = RAGEngine(collection_name="my_kb", config=config)
```

**Best for:** High-quality chunking, slower processing

---

## Embedding Providers

### Default (Local, Fast)

```python
from kk_utils.rag import RAGEngine, RAGConfig

config = RAGConfig(
    embedding={
        "provider": "default",
        "model": "all-MiniLM-L6-v2"
    }
)

rag = RAGEngine(collection_name="my_kb", config=config)
```

**No API key required** - runs locally

---

### OpenAI Embeddings

```python
from kk_utils.rag import RAGEngine, RAGConfig

# Set environment variable first
# export OPENAI_API_KEY=sk-...

config = RAGConfig(
    embedding={
        "provider": "openai",
        "model": "text-embedding-3-small"
    }
)

rag = RAGEngine(collection_name="my_kb", config=config)
```

**Best for:** High quality, production use

---

### HuggingFace Embeddings

```python
from kk_utils.rag import RAGEngine, RAGConfig

config = RAGConfig(
    embedding={
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
)

rag = RAGEngine(collection_name="my_kb", config=config)
```

**Best for:** Open source alternative

---

## Query with Filters

### Filter by Metadata

```python
from kk_utils.rag import RAGEngine

rag = RAGEngine(collection_name="my_kb")

# Filter by type
results = rag.query(
    question="How do I...?",
    filter_metadata={"type": "manual"}
)

# Filter by multiple fields
results = rag.query(
    question="What is...?",
    filter_metadata={
        "type": "resume",
        "category": "experience"
    }
)
```

---

### Query with Debug Info

```python
results = rag.query(
    question="What is my experience?",
    top_k=5,
    return_debug_info=True
)

# Access debug information
if results.debug:
    print(f"Retrieval time: {results.debug['retrieval_time_ms']}ms")
    
    for chunk_score in results.debug['chunk_scores']:
        print(f"Chunk {chunk_score['chunk_index']}: {chunk_score['similarity']:.3f}")
```

---

## Complete Example: Portfolio Import

```python
"""
Import portfolio documents to RAG
"""
from kk_utils.rag import RAGEngine, RAGConfig, SentenceChunker
from pathlib import Path
import time

def import_portfolio():
    # Configuration
    COLLECTION_NAME = "chua_keng_koon_portfolio"
    DOCS_FOLDER = Path("rag-documents")
    
    # Create RAG engine
    config = RAGConfig(
        chunking={
            "strategy": "sentence",
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        retrieval={
            "default_top_k": 5,
            "min_confidence": 0.15
        }
    )
    
    rag = RAGEngine(collection_name=COLLECTION_NAME, config=config)
    
    # Import documents
    for md_file in DOCS_FOLDER.glob("*.md"):
        if md_file.name in ["README.md", "CREATION_SUMMARY.md"]:
            continue  # Skip meta documents
        
        print(f"\n📄 Importing: {md_file.name}")
        
        # Read content
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Chunk document
        chunker = SentenceChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(content)
        
        # Add to RAG
        for i, chunk in enumerate(chunks):
            doc_id = f"{md_file.stem}_chunk_{i}"
            metadata = {
                "source_file": md_file.name,
                "chunk_index": i,
                "type": "portfolio",
                "import_date": time.strftime("%Y-%m-%d")
            }
            
            rag.add_document(doc_id=doc_id, text=chunk, metadata=metadata)
        
        print(f"  ✅ Added {len(chunks)} chunks")
    
    # Show stats
    stats = rag.get_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Collection: {stats['collection_name']}")

if __name__ == "__main__":
    import_portfolio()
```

---

## API Reference

### RAGEngine

```python
RAGEngine(
    collection_name: str = "default",
    persist_directory: Optional[str] = None,
    config: Optional[RAGConfig] = None,
)
```

**Methods:**
- `add_document(doc_id, text, metadata)` - Add document
- `query(question, top_k, filter_metadata, return_debug_info)` - Query
- `delete_document(doc_id)` - Delete document
- `get_stats()` - Get statistics
- `clear()` - Clear all data
- `get_config()` - Get configuration
- `update_config(updates)` - Update configuration

### RAGResult

```python
@dataclass
class RAGResult:
    query: str
    chunks: List[Dict[str, Any]]
    confidence: float
    sources: List[str]
    message: str
    debug: Optional[Dict[str, Any]]
    error: Optional[str]
    
    @property
    def has_results(self) -> bool: ...
```

---

## Best Practices

### 1. Choose Right Chunk Size

```python
# Small documents (< 10 pages)
chunk_size = 300

# Medium documents (10-50 pages)
chunk_size = 500

# Large documents (> 50 pages)
chunk_size = 800
```

### 2. Tune Confidence Thresholds

```python
# Permissive (small datasets)
min_confidence = 0.10

# Balanced (medium datasets)
min_confidence = 0.15

# Strict (large datasets)
min_confidence = 0.30
```

### 3. Use Metadata Filters

Always filter when applicable:

```python
results = rag.query(
    question="...",
    filter_metadata={
        "user_id": "user123",
        "type": "resume"
    }
)
```

### 4. Enable Debug for Testing

```python
config = RAGConfig(
    logging={
        "debug_mode": True,
        "log_queries": True
    }
)
```

---

## Troubleshooting

### Issue: ChromaDB not found

**Solution:**
```bash
pip install chromadb
```

### Issue: Low confidence scores

**Solution:**
```python
# Lower the threshold
config = RAGConfig(
    retrieval={"min_confidence": 0.05}
)
```

### Issue: Slow queries

**Solution:**
```python
# Reduce top_k
results = rag.query("...", top_k=3)

# Or use smaller chunks
config = RAGConfig(
    chunking={"chunk_size": 400}
)
```

---

## Related Documentation

- `kk_utils.rag_client` - API client for backend
- `personal-assistant/docs/20260305-core-rag-library-guide.md` - Full RAG guide
- `personal-assistant/docs/20260305-rag-configuration-guide.md` - Configuration reference

---

**End of RAG Module Usage Guide**
