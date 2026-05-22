"""
RAG Engine - Core implementation of Retrieval-Augmented Generation

Provides generic RAG functionality:
- Document storage with vector embeddings
- Semantic search and retrieval
- Configurable chunking and embedding
- Security filtering and sanitization
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import hashlib
from html.parser import HTMLParser
from pathlib import Path

from kk_utils.rag.config import RAGConfig, get_rag_config
from kk_utils.execution_trace import emit_trace

logger = logging.getLogger(__name__)


class _HTMLTextExtractor(HTMLParser):
    """Lightweight HTML text extractor without extra dependencies."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


@dataclass
class RAGResult:
    """Result from a RAG query."""
    query: str
    chunks: List[Dict[str, Any]]
    confidence: float
    sources: List[str]
    message: str = "Query completed successfully"
    retrieval_time_ms: float = 0.0
    chunks_searched: int = 0
    avg_distance: float = 0.0
    debug: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    @property
    def has_results(self) -> bool:
        """Check if query returned results."""
        return len(self.chunks) > 0 and self.confidence > 0


@dataclass
class DocumentInfo:
    """Information about a document in the RAG system."""
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    chunk_count: int
    added_at: datetime = field(default_factory=datetime.now)


class RAGEngine:
    """
    Generic RAG Engine for semantic search and retrieval.
    
    Features:
    - Multiple chunking strategies (word, sentence, semantic)
    - Multiple embedding providers (default, OpenAI, HuggingFace, Cohere)
    - Configurable retrieval parameters
    - Security filtering and content sanitization
    - Debug mode for detailed logging
    
    Usage:
        rag = RAGEngine(
            collection_name="knowledge_base",
            config=RAGConfig(chunk_size=500)
        )
        
        rag.add_document("doc1", text, metadata={"type": "manual"})
        results = rag.query("How do I...?")
    """
    
    SUPPORTED_FILE_TYPES: Dict[str, str] = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".txt": "txt",
        ".md": "md",
        ".html": "html",
        ".json": "json",
    }

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize RAG engine.
        
        Args:
            collection_name: Name of ChromaDB collection
            persist_directory: Directory for persistence (default: from config)
            config: RAG configuration (default: load from rag_config.yaml)
        """
        self.collection_name = collection_name
        self.config = config or get_rag_config()
        
        logger.info(f"Initializing RAG engine: collection={collection_name}, "
                   f"chunk_size={self.config.chunking.chunk_size}")
        
        # Initialize ChromaDB
        self.client = None
        self.collection = None
        self._init_chromadb(persist_directory)
        
        # Chunking strategy
        self.chunker = self._create_chunker()
        
        logger.info(f"RAG engine initialized with {self.collection.count() if self.collection else 0} chunks")

    @classmethod
    def supported_file_map(cls) -> Dict[str, str]:
        """Return a copy of the supported file type mapping."""
        return dict(cls.SUPPORTED_FILE_TYPES)

    @classmethod
    def supported_file_extensions(cls) -> List[str]:
        """Return supported file extensions including the leading dot."""
        return list(cls.SUPPORTED_FILE_TYPES.keys())

    @classmethod
    def supports_file_type(cls, file_ext: str) -> bool:
        """Check whether a file extension is supported by the ingestion pipeline."""
        return file_ext.lower() in cls.SUPPORTED_FILE_TYPES

    @classmethod
    def file_type_for_extension(cls, file_ext: str) -> Optional[str]:
        """Map a file extension to the canonical file type label."""
        return cls.SUPPORTED_FILE_TYPES.get(file_ext.lower())
    
    def _init_chromadb(self, persist_directory: Optional[str]):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            logger.warning("ChromaDB not installed. RAG disabled. Install with: pip install chromadb")
            return

        # Determine persist directory - use passed value if provided
        if persist_directory is None:
            persist_dir = self.config.storage.persist_directory
            # Derive data dir from RAG_CONFIG_PATH env var (set by the backend .env).
            # This avoids brittle parent-count path guessing across different layouts.
            import os
            config_path_env = os.environ.get("RAG_CONFIG_PATH")
            if config_path_env:
                backend_root = Path(config_path_env).resolve().parent.parent  # config/ -> backend/
                persist_directory = str(backend_root / "data" / persist_dir)
            else:
                # Fallback: resolve relative to CWD (works when CWD is the backend dir)
                persist_directory = str(Path.cwd() / "data" / persist_dir)
            logger.info(f"Using persist directory: {persist_directory}")
        else:
            logger.info(f"Using provided persist directory: {persist_directory}")

        self.persist_path = Path(persist_directory)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        try:
            from chromadb.utils import embedding_functions
            from kk_utils.rag.embedding import get_embedding_function
            
            # Get embedding function based on config
            ef = get_embedding_function(self.config.embedding)
            
            self.client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=chromadb.Settings(
                    anonymized_telemetry=self.config.storage.chromadb.get('anonymized_telemetry', False),
                    allow_reset=self.config.storage.chromadb.get('allow_reset', True),
                )
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": f"RAG knowledge base: {self.collection_name}",
                    "version": "1.0",
                    "embedding_model": self.config.embedding.model,
                },
                embedding_function=ef,
            )
            
            logger.info(f"ChromaDB collection: {self.collection.name}")
            
        except Exception as e:
            logger.error(f"ChromaDB initialization error: {e}", exc_info=True)
            self.client = None
            self.collection = None
    
    def _create_chunker(self):
        """Create chunker based on config strategy."""
        from kk_utils.rag.chunking import ChunkingStrategy

        strategy = self.config.chunking.strategy

        # Use the class method to create chunker
        return ChunkingStrategy.create_chunker(
            strategy=strategy,
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap
        )
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add document to RAG knowledge base.
        
        Args:
            doc_id: Unique document identifier
            text: Document text content
            metadata: Optional metadata (type, user_id, source, etc.)
        
        Returns:
            Dict with chunk count and document info
        """
        if not self.collection:
            return {"error": "RAG not initialized", "chunks": 0}
        
        # Chunk the document
        chunks = self.chunker.chunk(text)
        
        # Add metadata to each chunk
        base_metadata = metadata or {}
        base_metadata["doc_id"] = doc_id
        base_metadata["added_at"] = datetime.now().isoformat()
        
        # Create IDs and metadata for all chunks
        ids = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            ids.append(chunk_id)
            
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            metadatas.append(chunk_metadata)
        
        # Batch add to ChromaDB
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(chunks)} chunks from document {doc_id}")
        
        return {
            "doc_id": doc_id,
            "chunks_added": len(chunks),
            "total_chunks": self.collection.count(),
        }

    async def ingest_file(
        self,
        file_path: str | Path,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        filename: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a supported file into RAG.

        This owns the extraction pipeline for all supported types, then chunks
        and writes the extracted text to ChromaDB.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_ext = path.suffix.lower()
        if not self.supports_file_type(file_ext):
            raise ValueError(
                f"Unsupported file type: {file_ext}. Supported: {self.supported_file_extensions()}"
            )

        file_type = self.file_type_for_extension(file_ext)
        if not file_type:
            raise ValueError(f"Could not resolve file type for extension: {file_ext}")

        extracted_text = self._extract_text(path, file_type)
        file_size_bytes = path.stat().st_size
        resolved_filename = filename or path.name
        resolved_collection = collection_name or self.collection_name

        rag_metadata = dict(metadata or {})
        if user_id:
            rag_metadata["user_id"] = user_id
        rag_metadata["collection"] = resolved_collection
        rag_metadata["doc_id"] = doc_id
        rag_metadata["file_type"] = file_type
        rag_metadata["source_file"] = resolved_filename
        rag_metadata["uploaded_at"] = datetime.now().isoformat()
        rag_metadata["file_size_bytes"] = file_size_bytes

        filename_lower = resolved_filename.lower()
        if "resume" in filename_lower or "cv" in filename_lower:
            rag_metadata["type"] = "resume"
        elif "cover" in filename_lower and "letter" in filename_lower:
            rag_metadata["type"] = "cover_letter"
        else:
            rag_metadata["type"] = "document"

        result = self.add_document(
            doc_id=doc_id,
            text=extracted_text,
            metadata=rag_metadata,
        )

        result.update({
            "file_type": file_type,
            "file_size_bytes": file_size_bytes,
            "text_preview": extracted_text[:500],
            "text": extracted_text,
        })
        return result

    def _extract_text(self, file_path: Path, file_type: str) -> str:
        """Extract text from a supported file type."""
        if file_type == "pdf":
            return self._extract_text_from_pdf(file_path)
        if file_type == "docx":
            return self._extract_text_from_docx(file_path)
        if file_type in ("txt", "md"):
            return self._extract_text_from_text(file_path)
        if file_type == "html":
            return self._extract_text_from_html(file_path)
        if file_type == "json":
            return self._extract_text_from_json(file_path)
        raise ValueError(f"Unknown file type: {file_type}")

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF

            text_parts = []
            with fitz.open(file_path) as doc:
                for page in doc:
                    text_parts.append(page.get_text())

            return "\n".join(text_parts)

        except ImportError:
            logger.warning("PyMuPDF not installed. Install with: pip install pymupdf")
            return "[PDF text extraction not available - install pymupdf]"
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return "[PDF text extraction failed]"

    def _extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX using python-docx."""
        try:
            from docx import Document

            doc = Document(file_path)
            text_parts = [para.text for para in doc.paragraphs if para.text.strip()]

            return "\n".join(text_parts)

        except ImportError:
            logger.warning("python-docx not installed. Install with: pip install python-docx")
            return "[DOCX text extraction not available - install python-docx]"
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return "[DOCX text extraction failed]"

    def _extract_text_from_text(self, file_path: Path) -> str:
        """Extract text from TXT/MD files."""
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Text file reading failed: {e}")
            return "[Text extraction failed]"

    def _extract_text_from_html(self, file_path: Path) -> str:
        """Extract visible text from HTML files."""
        try:
            raw_html = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"HTML file reading failed: {e}")
            return "[HTML extraction failed]"

        try:
            parser = _HTMLTextExtractor()
            parser.feed(raw_html)
            text = parser.get_text().strip()
            return text or "[HTML extraction produced no text]"
        except Exception as e:
            logger.error(f"HTML extraction failed: {e}")
            return "[HTML extraction failed]"

    def _extract_text_from_json(self, file_path: Path) -> str:
        """Extract readable text from JSON files."""
        try:
            raw_json = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"JSON file reading failed: {e}")
            return "[JSON extraction failed]"

        try:
            data = json.loads(raw_json)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            # Fall back to the raw text so the file is still indexable.
            return raw_json
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_confidence: Optional[float] = None,
        return_debug_info: bool = False,
    ) -> RAGResult:
        """
        Query the knowledge base.
        
        Args:
            question: User's question (natural language)
            top_k: Number of chunks to retrieve (default: from config)
            filter_metadata: Filter by metadata (e.g., {"type": "manual"})
            min_confidence: Minimum similarity score (default: from config)
            return_debug_info: If True, include detailed debug information
        
        Returns:
            RAGResult with chunks, confidence, and metadata
        """
        import time
        start_time = time.time()
        
        if not self.collection:
            emit_trace(f"RAG[{self.collection_name}]: unavailable")
            return RAGResult(
                query=question,
                chunks=[],
                confidence=0.0,
                sources=[],
                message="RAG system not available",
                error="RAG not initialized"
            )
        
        # Use config defaults if not provided
        if top_k is None:
            top_k = self.config.retrieval.default_top_k
        if min_confidence is None:
            min_confidence = self.config.retrieval.min_confidence
        
        # Enforce maximum top_k
        top_k = min(top_k, self.config.retrieval.max_top_k)
        
        try:
            emit_trace(
                f"RAG[{self.collection_name}]: query start "
                f"top_k={top_k} min_confidence={min_confidence:.2f} "
                f"filter={filter_metadata or {}} "
                f"question={question[:80]!r}"
            )

            # Log query details
            if self.config.logging.log_queries:
                logger.info(f"RAG query: question='{question[:50]}...', top_k={top_k}, "
                           f"min_confidence={min_confidence:.2f}, filter={filter_metadata}")
            
            # Convert filter_metadata to ChromaDB v4+ operator format
            # ChromaDB v4+ requires {"key": {"$eq": "value"}} instead of {"key": "value"}
            chroma_filter = None
            if filter_metadata:
                if len(filter_metadata) == 1:
                    key, value = list(filter_metadata.items())[0]
                    chroma_filter = {key: {"$eq": value}}
                else:
                    chroma_filter = {
                        "$and": [{key: {"$eq": value}} for key, value in filter_metadata.items()]
                    }

            # Query vector DB
            results = self.collection.query(
                query_texts=[question],
                n_results=top_k,
                where=chroma_filter,
                include=["documents", "metadatas", "distances"],
            )
            
            # Handle empty results
            if not results.get("documents") or not results["documents"][0]:
                logger.warning(f"RAG query returned no results for: {question[:50]}")
                emit_trace(f"RAG[{self.collection_name}]: no results")
                return RAGResult(
                    query=question,
                    chunks=[],
                    confidence=0.0,
                    sources=[],
                    message="No relevant information found"
                )
            
            # Process results
            distances = results["distances"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else []
            
            # Enhanced logging in debug mode
            if self.config.logging.debug_mode:
                self._log_debug_info(question, distances, documents)
            
            # Calculate confidence
            if distances:
                avg_distance = sum(distances) / len(distances)
                confidence = max(0.0, 1.0 - (avg_distance / 2.0))
            else:
                avg_distance = 0.0
                confidence = 0.0
            
            logger.info(f"RAG query confidence: {confidence:.3f} (avg_distance: {avg_distance:.3f})")
            
            # Check minimum confidence
            if confidence < min_confidence:
                logger.info(f"Low confidence: {confidence:.3f} < {min_confidence:.2f}")
                emit_trace(
                    f"RAG[{self.collection_name}]: low confidence "
                    f"{confidence:.3f} < {min_confidence:.2f}"
                )
                return RAGResult(
                    query=question,
                    chunks=[],
                    confidence=confidence,
                    sources=[],
                    message=f"Low confidence ({confidence:.2f} < {min_confidence:.2f})"
                )
            
            # Format results with per-chunk scores
            chunks = []
            sources = set()
            chunk_scores = []
            
            for i, doc in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 0.0
                similarity = 1.0 - (distance / 2.0)
                
                # Track chunk scores for debug info
                if return_debug_info:
                    chunk_scores.append({
                        "chunk_index": i,
                        "distance": round(distance, 6),
                        "similarity": round(similarity, 6),
                        "metadata": {
                            "doc_id": metadata.get("doc_id"),
                            "type": metadata.get("type"),
                            "chunk_index": metadata.get("chunk_index"),
                        }
                    })
                
                chunk_data = {
                    "text": doc,
                    "relevance_score": round(similarity, 6),
                    "distance": round(distance, 6),
                    "metadata": metadata,
                }
                chunks.append(chunk_data)
                
                if metadata and metadata.get("doc_id"):
                    sources.add(metadata["doc_id"])
            
            # Calculate retrieval time
            retrieval_time_ms = (time.time() - start_time) * 1000
            
            # Log performance metrics
            if self.config.logging.log_performance:
                logger.info(f"RAG query performance: {retrieval_time_ms:.2f}ms, "
                           f"chunks_searched={self.collection.count()}, retrieved={len(chunks)}")

            emit_trace(
                f"RAG[{self.collection_name}]: query done "
                f"confidence={confidence:.3f} chunks={len(chunks)} "
                f"time_ms={retrieval_time_ms:.0f}"
            )

            # Build response
            result = RAGResult(
                query=question,
                chunks=chunks,
                confidence=round(confidence, 6),
                sources=list(sources),
                message="Information retrieved successfully",
                retrieval_time_ms=round(retrieval_time_ms, 2),
                chunks_searched=self.collection.count(),
                avg_distance=round(avg_distance, 3),
            )
            
            # Add debug info if requested
            if return_debug_info and self.config.logging.debug_mode:
                result.debug = {
                    "query_length": len(question),
                    "embedding_dimensions": self.config.embedding.dimensions,
                    "chunk_scores": chunk_scores,
                    "retrieval_time_ms": round(retrieval_time_ms, 2),
                    "total_chunks_searched": self.collection.count(),
                    "avg_distance": round(avg_distance, 6),
                    "min_distance": round(min(distances) if distances else 0, 6),
                    "max_distance": round(max(distances) if distances else 0, 6),
                }
            
            return result
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}", exc_info=True)
            emit_trace(f"RAG[{self.collection_name}]: error {e}")
            return RAGResult(
                query=question,
                chunks=[],
                confidence=0.0,
                sources=[],
                message="Query failed",
                error=str(e)
            )
    
    def _log_debug_info(self, question: str, distances: List[float], documents: List[str]):
        """Log detailed debug information for query."""
        logger.info("=" * 60)
        logger.info(f"RAG QUERY: {question[:80]}...")
        logger.info(f"Retrieved {len(distances)} chunks:")
        
        for i, (dist, doc_preview) in enumerate(zip(distances, documents)):
            similarity = 1.0 - (dist / 2.0)
            doc_text_preview = doc_preview[:100].replace('\n', ' ')
            logger.info(f"  Chunk {i+1}: distance={dist:.4f}, similarity={similarity:.4f}, "
                       f"text='{doc_text_preview}...'")
        logger.info("=" * 60)
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete all chunks from a document."""
        if not self.collection:
            return {"error": "RAG not initialized"}
        
        try:
            # Get all chunks for this doc_id
            all_data = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"]
            )
            
            if all_data and all_data.get("ids"):
                self.collection.delete(ids=all_data["ids"])
                logger.info(f"Deleted document {doc_id}")
                return {"deleted": True, "doc_id": doc_id}
            else:
                return {"deleted": False, "message": "Document not found"}
                
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if not self.collection:
            return {"error": "RAG not initialized"}
        
        count = self.collection.count()
        
        try:
            all_data = self.collection.get(include=["metadatas"])
            doc_ids = set()
            
            if all_data and all_data.get("metadatas"):
                for metadata in all_data["metadatas"]:
                    if metadata.get("doc_id"):
                        doc_ids.add(metadata["doc_id"])
            
            return {
                "total_chunks": count,
                "total_documents": len(doc_ids),
                "documents": list(doc_ids),
                "collection_name": self.collection_name,
                "config": {
                    "chunk_size": self.config.chunking.chunk_size,
                    "chunk_overlap": self.config.chunking.chunk_overlap,
                    "embedding_model": self.config.embedding.model,
                    "min_confidence": self.config.retrieval.min_confidence,
                },
            }
        except Exception as e:
            return {
                "total_chunks": count,
                "error": str(e)
            }
    
    def clear(self) -> Dict[str, Any]:
        """Clear all data from RAG."""
        if not self.collection:
            return {"error": "RAG not initialized"}
        
        self.client.delete_collection(self.collection_name)
        self.collection = None
        self._init_chromadb(None)
        
        logger.info("Cleared all RAG data")
        return {"cleared": True}
    
    def get_config(self) -> RAGConfig:
        """Get current RAG configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update RAG configuration at runtime.
        
        Args:
            updates: Nested dict of config updates
                    e.g., {"chunking": {"chunk_size": 600}}
        
        Note:
            Some changes (like chunk_size) only affect NEW documents.
            Embedding changes require re-initialization.
        """
        from kk_utils.rag.config import update_rag_config
        
        # Update global config
        update_rag_config(updates)
        
        # Update local config
        self.config = get_rag_config()
        
        # Update chunker if chunking settings changed
        if 'chunking' in updates:
            self.chunker = self._create_chunker()
            if 'chunk_size' in updates['chunking']:
                self.chunker.chunk_size = updates['chunking']['chunk_size']
            if 'chunk_overlap' in updates['chunking']:
                self.chunker.chunk_overlap = updates['chunking']['chunk_overlap']
        
        logger.info(f"Updated RAG configuration: {updates}")
