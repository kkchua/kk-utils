"""
digital_me_rag skill — rag.py

DigitalMeRAG extends kk_utils.rag.RAGEngine with Digital Me-specific behaviour:
- User-scoped security filtering on every query and document
- Document-type auto-detection (resume, cover_letter, document)
- Convenience search methods (search_resume, search_projects)
- Singleton factory: get_digital_me_rag()

No backend dependencies — only kk_utils.rag.
"""
from typing import Dict, Any, Optional
import logging

from kk_utils.rag import RAGEngine, RAGConfig, RAGResult
from kk_utils.rag.config import get_rag_config

logger = logging.getLogger(__name__)


class DigitalMeRAG(RAGEngine):
    """
    RAG system for the Digital Me knowledge base.

    Extends RAGEngine with:
    - Security: user_id injected into every query / document
    - Auto-detection of document type from filename
    - search_resume() and search_projects() helpers

    Usage:
        rag = DigitalMeRAG()
        rag.add_document("resume.pdf", text, metadata={"type": "resume"})
        result = rag.query("What is my work experience?")
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        rag_config = config or get_rag_config()
        logger.info("Initializing Digital Me RAG")

        super().__init__(
            collection_name="digital_me",
            config=rag_config,
        )

        self.security_enabled = rag_config.security.enable_access_control
        self.default_user_id = rag_config.security.default_user_id

        logger.info(f"Digital Me RAG initialized with {self.get_stats().get('total_chunks', 0)} chunks")

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_confidence: Optional[float] = None,
        return_debug_info: bool = False,
        user_id: Optional[str] = None,
    ) -> RAGResult:
        """
        Query the Digital Me knowledge base with user-scoped security filter.

        Args:
            question: Natural language question
            top_k: Number of chunks to retrieve (default: from config)
            filter_metadata: Additional metadata filter
            min_confidence: Minimum similarity score (default: from config)
            return_debug_info: Include debug info in result
            user_id: User ID for access control (default: from config)
        """
        final_filter = dict(filter_metadata or {})

        uid = user_id or self.default_user_id
        if self.security_enabled and uid:
            final_filter["user_id"] = uid

        return super().query(
            question=question,
            top_k=top_k,
            filter_metadata=final_filter,
            min_confidence=min_confidence,
            return_debug_info=return_debug_info,
        )

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a document to the Digital Me knowledge base.

        Injects user_id and auto-detects document type from filename.

        Args:
            doc_id: Unique document identifier
            text: Document text content
            metadata: Optional metadata (type, source, filename, etc.)
            user_id: User ID for access control
        """
        doc_metadata = dict(metadata or {})

        uid = user_id or self.default_user_id
        if uid:
            doc_metadata["user_id"] = uid

        if "type" not in doc_metadata:
            filename = doc_metadata.get("filename", "").lower()
            if "resume" in filename or "cv" in filename:
                doc_metadata["type"] = "resume"
            elif "cover" in filename and "letter" in filename:
                doc_metadata["type"] = "cover_letter"
            else:
                doc_metadata["type"] = "document"

        return super().add_document(doc_id=doc_id, text=text, metadata=doc_metadata)

    def search_resume(self, query: str, top_k: int = 3, min_confidence: float = 0.15) -> RAGResult:
        """Search resume documents specifically."""
        return self.query(question=query, top_k=top_k, min_confidence=min_confidence,
                          filter_metadata={"type": "resume"})

    def search_projects(self, query: str, top_k: int = 3, min_confidence: float = 0.15) -> RAGResult:
        """Search project documents specifically."""
        return self.query(question=query, top_k=top_k, min_confidence=min_confidence,
                          filter_metadata={"type": "projects"})


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_instance: Optional[DigitalMeRAG] = None


def get_digital_me_rag() -> DigitalMeRAG:
    """Get or create the singleton DigitalMeRAG instance."""
    global _instance
    if _instance is None:
        _instance = DigitalMeRAG()
    return _instance


def reset_digital_me_rag() -> None:
    """Reset the singleton — for testing only."""
    global _instance
    _instance = None
