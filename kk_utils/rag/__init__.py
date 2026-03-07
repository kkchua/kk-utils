"""
Core RAG Library - Reusable Retrieval-Augmented Generation Engine

This module provides a generic RAG engine that can be used for any knowledge base:
- Document Q&A
- Manual search
- Knowledge base lookup
- Semantic search
- Multi-collection management

Usage:
    from kk_utils.rag import RAGEngine, RAGConfig, RAGCollectionManager
    
    # Single collection
    rag = RAGEngine(collection_name="my_kb")
    
    # Multiple collections
    manager = RAGCollectionManager()
    agent_me = manager.get_collection("agent_me_chat")
    hr_docs = manager.get_collection("hr_documents")
"""

from kk_utils.rag.rag_engine import RAGEngine, RAGResult
from kk_utils.rag.config import RAGConfig, get_rag_config
from kk_utils.rag.chunking import ChunkingStrategy, WordChunker, SentenceChunker
from kk_utils.rag.embedding import EmbeddingProvider, get_embedding_function
from kk_utils.rag.collection_manager import RAGCollectionManager, create_rag_collections
from kk_utils.rag.rag_service import RAGService

__all__ = [
    # Core engine
    "RAGEngine",
    "RAGResult",

    # Configuration
    "RAGConfig",
    "get_rag_config",

    # Chunking
    "ChunkingStrategy",
    "WordChunker",
    "SentenceChunker",

    # Embedding
    "EmbeddingProvider",
    "get_embedding_function",

    # Collection Management
    "RAGCollectionManager",
    "create_rag_collections",

    # Service Layer (Unified)
    "RAGService",
]
