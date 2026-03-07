"""
Embedding Adapters for RAG

Provides multiple embedding providers:
- Default: ChromaDB default (all-MiniLM-L6-v2, fast, local)
- OpenAI: OpenAI embeddings (text-embedding-3-small, requires API key)
- HuggingFace: HuggingFace embeddings (requires model specification)
- Cohere: Cohere embeddings (requires API key)

Usage:
    from kk_utils.rag.embedding import get_embedding_function, EmbeddingConfig
    
    config = EmbeddingConfig(provider="openai", model="text-embedding-3-small")
    ef = get_embedding_function(config)
"""
from typing import Optional, Any
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    provider: str = "default"
    model: str = "all-MiniLM-L6-v2"
    dimensions: int = 384
    cache_enabled: bool = True
    cache_dir: str = "embedding_cache"
    api_key: Optional[str] = None


class EmbeddingProvider:
    """Enum-like class for embedding providers."""
    
    DEFAULT = "default"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"


def get_embedding_function(config: EmbeddingConfig) -> Any:
    """
    Get embedding function based on configuration.
    
    Args:
        config: Embedding configuration
    
    Returns:
        ChromaDB embedding function instance
    
    Raises:
        ValueError: If provider is not supported
        ImportError: If required package is not installed
    """
    from chromadb.utils import embedding_functions
    
    provider = config.provider.lower()
    
    if provider == EmbeddingProvider.DEFAULT:
        # Default embedding function (all-MiniLM-L6-v2)
        logger.info(f"Using default embedding: {config.model}")
        return embedding_functions.DefaultEmbeddingFunction()
    
    elif provider == EmbeddingProvider.OPENAI:
        # OpenAI embeddings
        try:
            api_key = config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            
            logger.info(f"Using OpenAI embedding: {config.model}")
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=config.model,
            )
        except ImportError as e:
            logger.error(f"OpenAI embedding package not installed: {e}")
            logger.warning("Falling back to default embedding")
            return embedding_functions.DefaultEmbeddingFunction()
    
    elif provider == EmbeddingProvider.HUGGINGFACE:
        # HuggingFace embeddings
        try:
            logger.info(f"Using HuggingFace embedding: {config.model}")
            return embedding_functions.HuggingFaceEmbeddingFunction(
                model_name=config.model,
            )
        except ImportError as e:
            logger.error(f"HuggingFace embedding package not installed: {e}")
            logger.warning("Falling back to default embedding")
            return embedding_functions.DefaultEmbeddingFunction()
    
    elif provider == EmbeddingProvider.COHERE:
        # Cohere embeddings
        try:
            api_key = config.api_key or os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("Cohere API key not provided")
            
            logger.info(f"Using Cohere embedding: {config.model}")
            return embedding_functions.CohereEmbeddingFunction(
                api_key=api_key,
                model_name=config.model,
            )
        except ImportError as e:
            logger.error(f"Cohere embedding package not installed: {e}")
            logger.warning("Falling back to default embedding")
            return embedding_functions.DefaultEmbeddingFunction()
    
    else:
        logger.warning(f"Unknown embedding provider: {provider}, using default")
        return embedding_functions.DefaultEmbeddingFunction()


# Convenience functions for direct usage
def get_default_embedding() -> Any:
    """Get default embedding function."""
    from chromadb.utils import embedding_functions
    return embedding_functions.DefaultEmbeddingFunction()


def get_openai_embedding(model: str = "text-embedding-3-small", api_key: Optional[str] = None) -> Any:
    """
    Get OpenAI embedding function.
    
    Args:
        model: OpenAI embedding model name
        api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
    """
    from chromadb.utils import embedding_functions
    
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required")
    
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=key,
        model_name=model,
    )


def get_huggingface_embedding(model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    """
    Get HuggingFace embedding function.
    
    Args:
        model: HuggingFace model name
    """
    from chromadb.utils import embedding_functions
    
    return embedding_functions.HuggingFaceEmbeddingFunction(
        model_name=model,
    )


def get_cohere_embedding(model: str = "embed-english-v3.0", api_key: Optional[str] = None) -> Any:
    """
    Get Cohere embedding function.
    
    Args:
        model: Cohere embedding model name
        api_key: Cohere API key (default: from COHERE_API_KEY env var)
    """
    from chromadb.utils import embedding_functions
    
    key = api_key or os.getenv("COHERE_API_KEY")
    if not key:
        raise ValueError("Cohere API key required")
    
    return embedding_functions.CohereEmbeddingFunction(
        api_key=key,
        model_name=model,
    )
