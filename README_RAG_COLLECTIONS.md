# RAG Collections - Organizing Your Data

**Purpose:** How to organize RAG documents into logical groups/collections

---

## Overview

You can organize your RAG data in **two ways**:

1. **Multiple Collections** (Recommended) - Separate vector spaces
2. **Metadata Filters** - Single collection with categories

---

## Method 1: Multiple Collections (Recommended)

### Use Case: Different Domains

```
agent_me_chat      → Personal portfolio, resume, profile
agent_me_research  → Research papers, technical docs
hr_documents       → HR policies, employee handbooks
legal_documents    → Contracts, legal agreements
```

### Example Usage

```python
from kk_utils.rag import RAGCollectionManager

# Initialize manager
manager = RAGCollectionManager(persist_directory="./rag_data")

# Get or create collections
agent_me = manager.get_collection("agent_me_chat")
research = manager.get_collection("agent_me_research")
hr = manager.get_collection("hr_documents")
legal = manager.get_collection("legal_documents")

# Add documents to specific collections
agent_me.add_document(
    doc_id="profile_001",
    text="Chua Keng Koon is a software engineer...",
    metadata={"type": "profile"}
)

hr.add_document(
    doc_id="hr_policy_001",
    text="Employee handbook section 1...",
    metadata={"type": "policy"}
)

# Query specific collection
results = agent_me.query("What is my experience?")
print(f"Found {len(results.chunks)} results in agent_me_chat")

# Search across all collections
all_results = manager.search_all("Python programming")
for collection_name, results in all_results.items():
    print(f"\n{collection_name}:")
    for chunk in results['chunks']:
        print(f"  - {chunk['text'][:100]}...")
```

---

## Method 2: Metadata Filters

### Use Case: Related Content, Single Collection

```python
from kk_utils.rag import RAGEngine

# Single collection for all documents
rag = RAGEngine(collection_name="all_documents")

# Add documents with category metadata
rag.add_document(
    doc_id="doc_001",
    text="Profile content...",
    metadata={
        "category": "agent_me_chat",
        "type": "profile",
        "user_id": "demo"
    }
)

rag.add_document(
    doc_id="doc_002",
    text="Legal contract...",
    metadata={
        "category": "legal",
        "type": "contract",
        "user_id": "demo"
    }
)

# Query with filter
results = rag.query(
    question="What is my experience?",
    filter_metadata={"category": "agent_me_chat"}
)

# Search across categories
all_results = rag.query("Python skills")
```

---

## Comparison

| Feature | Multiple Collections | Metadata Filters |
|---------|---------------------|------------------|
| **Isolation** | Complete | Partial |
| **Performance** | Faster (targeted search) | Slower (filters after search) |
| **Flexibility** | Separate configs | Single config |
| **Cross-search** | Manual (search_all) | Automatic |
| **Best For** | Different domains | Related content |

---

## Recommended Structure for Your Use Case

### For Personal Portfolio + Multiple Domains

```python
from kk_utils.rag import create_rag_collections

# Create all collections at once
manager = create_rag_collections(
    collection_names=[
        "agent_me_chat",       # Personal portfolio
        "agent_me_research",   # Research & technical
        "hr_documents",        # HR & policies
        "legal_documents",     # Legal & contracts
        "general_knowledge",   # General info
    ],
    persist_directory="./rag_data"
)

# Import portfolio documents
portfolio_collection = manager.get_collection("agent_me_chat")

# Import your resume/profile
for md_file in Path("rag-documents").glob("*.md"):
    # ... read and chunk ...
    portfolio_collection.add_document(doc_id, chunk, metadata)

# Query portfolio only
results = portfolio_collection.query("What is my experience?")

# Or search everything
all_results = manager.search_all("Python programming")
```

---

## Advanced: Collection-Specific Configurations

```python
from kk_utils.rag import RAGCollectionManager, RAGConfig

manager = RAGCollectionManager()

# AgentMe Chat - Optimized for conversation
agent_me_config = RAGConfig(
    chunking={"strategy": "sentence", "chunk_size": 400},
    retrieval={"min_confidence": 0.15, "default_top_k": 5}
)

agent_me = manager.get_collection(
    "agent_me_chat",
    config=agent_me_config
)

# Legal - Optimized for precision
legal_config = RAGConfig(
    chunking={"strategy": "sentence", "chunk_size": 600},
    retrieval={"min_confidence": 0.30, "default_top_k": 3}
)

legal = manager.get_collection(
    "legal_documents",
    config=legal_config
)
```

---

## Complete Example: Multi-Domain Import

```python
"""
Import documents into multiple RAG collections
"""
from kk_utils.rag import RAGCollectionManager, RAGConfig
from pathlib import Path

# Initialize manager
manager = RAGCollectionManager(persist_directory="./rag_data")

# Define collection configurations
COLLECTIONS = {
    "agent_me_chat": {
        "folder": "chua_keng_koon",
        "config": RAGConfig(
            chunking={"strategy": "sentence", "chunk_size": 500}
        )
    },
    "hr_documents": {
        "folder": "hr_policies",
        "config": RAGConfig(
            chunking={"strategy": "sentence", "chunk_size": 600}
        )
    },
    "legal_documents": {
        "folder": "legal_contracts",
        "config": RAGConfig(
            chunking={"strategy": "sentence", "chunk_size": 700}
        )
    },
}

# Import documents for each collection
for collection_name, config in COLLECTIONS.items():
    print(f"\n📁 Importing: {collection_name}")
    
    # Get collection
    rag = manager.get_collection(collection_name, config=config)
    
    # Import documents from folder
    docs_folder = Path("rag-input-data") / config["folder"]
    
    for md_file in docs_folder.glob("*.md"):
        print(f"  📄 {md_file.name}")
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Chunk and add
        # ... chunking logic ...
        
        for i, chunk in enumerate(chunks):
            doc_id = f"{md_file.stem}_chunk_{i}"
            metadata = {
                "source_file": md_file.name,
                "category": collection_name,
                "chunk_index": i
            }
            
            rag.add_document(doc_id=doc_id, text=chunk, metadata=metadata)
    
    print(f"  ✅ Collection '{collection_name}' ready")

# Show all collections
print("\n📊 Collection Summary:")
stats = manager.get_stats()
print(f"Total collections: {stats['total_collections']}")

for name, collection_stats in stats['collections'].items():
    print(f"  - {name}: {collection_stats.get('total_chunks', 0)} chunks")
```

---

## Query Examples

### Query Single Collection

```python
# Get specific collection
agent_me = manager.get_collection("agent_me_chat")

# Query
results = agent_me.query("What is Chua Keng Koon's experience?")

print(f"Confidence: {results.confidence:.2f}")
for chunk in results.chunks:
    print(f"- {chunk['text'][:100]}...")
```

### Search Across All Collections

```python
# Search everything
all_results = manager.search_all(
    query="Python programming",
    top_k=3
)

# Display results by collection
for collection_name, data in all_results.items():
    if 'error' in data:
        print(f"❌ {collection_name}: {data['error']}")
        continue
    
    print(f"\n📚 {collection_name}:")
    print(f"   Confidence: {data['confidence']:.2f}")
    
    for chunk in data['chunks']:
        print(f"   - {chunk['text'][:80]}...")
```

### Search Specific Collections Only

```python
# Search only in agent_me and research
results = manager.search_all(
    query="AI experience",
    include_collections=["agent_me_chat", "agent_me_research"]
)

# Exclude HR and legal
results = manager.search_all(
    query="Python",
    exclude_collections=["hr_documents", "legal_documents"]
)
```

---

## Best Practices

### 1. Name Collections Clearly

```python
# ✅ Good
"agent_me_chat"
"hr_documents"
"legal_contracts"

# ❌ Bad
"collection1"
"test"
"docs"
```

### 2. Use Consistent Metadata

```python
metadata = {
    "category": "agent_me_chat",
    "type": "profile",
    "source": "resume",
    "user_id": "demo",
    "import_date": "2026-03-05"
}
```

### 3. Separate by Domain

```python
# ✅ Do: Separate collections for different domains
portfolio = manager.get_collection("portfolio")
legal = manager.get_collection("legal")

# ❌ Don't: Mix unrelated content
mixed = manager.get_collection("everything")
```

### 4. Use search_all() Sparingly

```python
# ✅ Use for broad queries
results = manager.search_all("Python programming")

# ❌ Don't use when you know the domain
# (query specific collection instead)
agent_me = manager.get_collection("agent_me_chat")
results = agent_me.query("My experience")
```

---

## Troubleshooting

### Issue: Collection not found

```python
# Collection doesn't exist yet
rag = manager.get_collection("new_collection", create_if_missing=True)
```

### Issue: Wrong results from collection

```python
# Check what's in the collection
info = manager.get_collection_info("agent_me_chat")
print(info['stats'])
```

### Issue: Need to reset collection

```python
# Delete and recreate
manager.delete_collection("agent_me_chat")
rag = manager.get_collection("agent_me_chat")  # Fresh start
```

---

## Summary

**For your use case (AgentMe Chat + HR + Legal):**

```python
from kk_utils.rag import RAGCollectionManager

manager = RAGCollectionManager(persist_directory="./rag_data")

# Get collections
agent_me = manager.get_collection("agent_me_chat")
hr = manager.get_collection("hr_documents")
legal = manager.get_collection("legal_documents")

# Import documents to respective collections
# ... import logic ...

# Query specific domain
agent_me.query("What is my experience?")

# Or search all
manager.search_all("Python skills")
```

This gives you **clean separation** + **flexible cross-search**! 🎯
