"""
KK-Utils - RAG Collection Manager

Manages multiple RAG collections for different domains/categories.

Usage:
    from kk_utils.rag import RAGCollectionManager
    
    # Initialize manager
    manager = RAGCollectionManager()
    
    # Get or create collection
    agent_me = manager.get_collection("agent_me_chat")
    hr_docs = manager.get_collection("hr_documents")
    
    # Add documents
    agent_me.add_document(...)
    hr_docs.add_document(...)
    
    # Query specific collection
    results = agent_me.query("...")
    
    # Search across all collections
    all_results = manager.search_all("...")
"""

from typing import Dict, Optional, List, Any
from kk_utils.rag.rag_engine import RAGEngine
from kk_utils.rag.config import RAGConfig
import logging

logger = logging.getLogger(__name__)


class RAGCollectionManager:
    """
    Manages multiple RAG collections.
    
    Provides unified interface to work with multiple domain-specific collections.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize collection manager.
        
        Args:
            persist_directory: Base directory for ChromaDB persistence
        """
        self.persist_directory = persist_directory
        self.collections: Dict[str, RAGEngine] = {}
        self.config_cache: Dict[str, RAGConfig] = {}
        
        logger.info(f"RAGCollectionManager initialized: persist_directory={persist_directory}")
    
    def get_collection(
        self,
        name: str,
        config: Optional[RAGConfig] = None,
        create_if_missing: bool = True,
    ) -> RAGEngine:
        """
        Get or create a RAG collection.
        
        Args:
            name: Collection name (e.g., "agent_me_chat", "hr_documents")
            config: Optional configuration (uses default if not provided)
            create_if_missing: Create collection if it doesn't exist
        
        Returns:
            RAGEngine instance for the collection
        """
        if name in self.collections:
            logger.debug(f"Collection '{name}' already exists, returning cached instance")
            return self.collections[name]
        
        if not create_if_missing:
            raise ValueError(f"Collection '{name}' does not exist")
        
        # Create new collection
        logger.info(f"Creating new collection: {name}")
        
        if config is None:
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
        
        # Create persist directory for this collection
        collection_persist_dir = None
        if self.persist_directory:
            from pathlib import Path
            collection_persist_dir = str(Path(self.persist_directory) / name)
        
        rag = RAGEngine(
            collection_name=name,
            persist_directory=collection_persist_dir,
            config=config
        )
        
        self.collections[name] = rag
        self.config_cache[name] = config
        
        logger.info(f"Collection '{name}' created successfully")
        return rag
    
    def list_collections(self) -> List[str]:
        """
        List all managed collections.
        
        Returns:
            List of collection names
        """
        return list(self.collections.keys())
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name
        
        Returns:
            True if deleted, False if collection didn't exist
        """
        if name not in self.collections:
            logger.warning(f"Collection '{name}' does not exist")
            return False
        
        logger.info(f"Deleting collection: {name}")
        
        # Clear from ChromaDB
        self.collections[name].clear()
        
        # Remove from cache
        del self.collections[name]
        if name in self.config_cache:
            del self.config_cache[name]
        
        logger.info(f"Collection '{name}' deleted successfully")
        return True
    
    def search_all(
        self,
        query: str,
        top_k: int = 5,
        exclude_collections: Optional[List[str]] = None,
        include_collections: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search across all (or specified) collections.
        
        Args:
            query: Search query
            top_k: Number of results per collection
            exclude_collections: Collections to exclude from search
            include_collections: Only search these collections (if specified)
        
        Returns:
            Dict with collection names as keys and results as values
        """
        results = {}
        
        # Determine which collections to search
        collections_to_search = self.list_collections()
        
        if include_collections:
            collections_to_search = [
                c for c in collections_to_search if c in include_collections
            ]
        
        if exclude_collections:
            collections_to_search = [
                c for c in collections_to_search if c not in exclude_collections
            ]
        
        logger.info(f"Searching {len(collections_to_search)} collections: {collections_to_search}")
        
        # Search each collection
        for collection_name in collections_to_search:
            try:
                rag = self.get_collection(collection_name)
                result = rag.query(query, top_k=top_k)
                
                if result.has_results:
                    results[collection_name] = {
                        "confidence": result.confidence,
                        "chunks": result.chunks,
                        "sources": result.sources,
                    }
                    
                    logger.debug(f"Collection '{collection_name}': {len(result.chunks)} results")
            
            except Exception as e:
                logger.error(f"Error searching collection '{collection_name}': {e}")
                results[collection_name] = {"error": str(e)}
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all collections.
        
        Returns:
            Dict with collection statistics
        """
        stats = {
            "total_collections": len(self.collections),
            "collections": {}
        }
        
        for name, rag in self.collections.items():
            try:
                collection_stats = rag.get_stats()
                stats["collections"][name] = collection_stats
            except Exception as e:
                logger.error(f"Error getting stats for '{name}': {e}")
                stats["collections"][name] = {"error": str(e)}
        
        return stats
    
    def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific collection.
        
        Args:
            name: Collection name
        
        Returns:
            Collection info dict or None if not found
        """
        if name not in self.collections:
            return None
        
        try:
            rag = self.get_collection(name)
            stats = rag.get_stats()
            
            return {
                "name": name,
                "exists": True,
                "stats": stats,
                "config": {
                    "chunking": self.config_cache.get(name, {}).chunking if name in self.config_cache else None,
                    "retrieval": self.config_cache.get(name, {}).retrieval if name in self.config_cache else None,
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection info for '{name}': {e}")
            return {
                "name": name,
                "exists": True,
                "error": str(e)
            }


# Convenience function for quick setup
def create_rag_collections(
    collection_names: List[str],
    persist_directory: Optional[str] = None,
    default_config: Optional[RAGConfig] = None,
) -> RAGCollectionManager:
    """
    Create multiple RAG collections at once.
    
    Args:
        collection_names: List of collection names to create
        persist_directory: Base directory for persistence
        default_config: Default configuration for all collections
    
    Returns:
        RAGCollectionManager with all collections created
    
    Example:
        manager = create_rag_collections([
            "agent_me_chat",
            "agent_me_research",
            "hr_documents",
            "legal_documents"
        ])
    """
    manager = RAGCollectionManager(persist_directory=persist_directory)
    
    for name in collection_names:
        manager.get_collection(name, config=default_config)
    
    return manager
