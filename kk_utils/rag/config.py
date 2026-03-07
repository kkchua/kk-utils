"""
Personal Assistant Backend - RAG Configuration

Centralized configuration management for Digital Me RAG system.
Loads settings from backend/config/rag_config.yaml

Usage:
    from kk_utils.rag.config import get_rag_config

    config = get_rag_config()
    chunk_size = config.chunking.chunk_size
    min_confidence = config.retrieval.min_confidence
"""
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import yaml
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Chunking configuration."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    strategy: str = "word"


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    default_top_k: int = 3
    max_top_k: int = 10
    min_confidence: float = 0.15
    high_confidence_threshold: float = 0.60
    distance_metric: str = "cosine"
    rerank_enabled: bool = False
    rerank: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    provider: str = "default"
    model: str = "all-MiniLM-L6-v2"
    dimensions: int = 384
    cache_enabled: bool = True
    cache_dir: str = "embedding_cache"


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_access_control: bool = True
    default_user_id: str = "demo_user"
    enable_content_sanitization: bool = True
    sanitization_patterns: List[str] = field(default_factory=list)
    enable_context_tagging: bool = True
    context_tag_format: str = "[Chunk {index}]: {content}"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "DEBUG"
    log_queries: bool = True
    log_results: bool = True
    log_embeddings: bool = False
    log_performance: bool = True
    debug_mode: bool = False
    log_file: str = "rag_service.log"


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    batch_processing: bool = True
    batch_size: int = 32
    parallel_processing: bool = False
    num_workers: int = 4
    result_cache_enabled: bool = False
    cache_ttl_seconds: int = 300
    max_cache_size: int = 100


@dataclass
class StorageConfig:
    """Storage configuration."""
    persist_directory: str = "digital_me_rag"
    persist_enabled: bool = True
    backend: str = "chromadb"
    chromadb: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestingConfig:
    """Testing configuration."""
    test_mode: bool = False
    sample_queries: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGConfig:
    """Main RAG configuration container."""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    
    # Raw config dict for dynamic access
    raw: Dict[str, Any] = field(default_factory=dict)


class RAGConfigLoader:
    """
    RAG Configuration Loader.
    
    Loads configuration from YAML file with environment variable overrides.
    Implements singleton pattern for consistent config access.
    """
    
    _instance: Optional['RAGConfigLoader'] = None
    _config: Optional[RAGConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._config = None
    
    def load_config(self, config_path: Optional[str] = None) -> RAGConfig:
        """
        Load RAG configuration from YAML file.
        
        Args:
            config_path: Optional path to config file.
                        If None, uses default location.
        
        Returns:
            RAGConfig object with all settings
        """
        if self._config is not None:
            return self._config
        
        # Determine config path
        if config_path is None:
            # Default: backend/config/rag_config.yaml
            base_path = Path(__file__).resolve().parent.parent.parent  # backend/
            config_path = str(base_path / "config" / "rag_config.yaml")
        
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"RAG config not found: {config_file}, using defaults")
            self._config = RAGConfig()
            return self._config
        
        try:
            # Load YAML
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            logger.info(f"Loaded RAG config from {config_file}")
            
            # Parse into structured config
            self._config = self._parse_config(raw_config)
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            # Configure logging level
            self._configure_logging()
            
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load RAG config: {e}", exc_info=True)
            self._config = RAGConfig()
            return self._config
    
    def _parse_config(self, raw: Dict[str, Any]) -> RAGConfig:
        """Parse raw YAML config into structured RAGConfig."""
        return RAGConfig(
            raw=raw,
            chunking=ChunkingConfig(
                chunk_size=raw.get('chunking', {}).get('chunk_size', 500),
                chunk_overlap=raw.get('chunking', {}).get('chunk_overlap', 50),
                strategy=raw.get('chunking', {}).get('strategy', 'word'),
            ),
            retrieval=RetrievalConfig(
                default_top_k=raw.get('retrieval', {}).get('default_top_k', 3),
                max_top_k=raw.get('retrieval', {}).get('max_top_k', 10),
                min_confidence=raw.get('retrieval', {}).get('min_confidence', 0.15),
                high_confidence_threshold=raw.get('retrieval', {}).get('high_confidence_threshold', 0.60),
                distance_metric=raw.get('retrieval', {}).get('distance_metric', 'cosine'),
                rerank_enabled=raw.get('retrieval', {}).get('rerank_enabled', False),
                rerank=raw.get('retrieval', {}).get('rerank', {}),
            ),
            embedding=EmbeddingConfig(
                provider=raw.get('embedding', {}).get('provider', 'default'),
                model=raw.get('embedding', {}).get('model', 'all-MiniLM-L6-v2'),
                dimensions=raw.get('embedding', {}).get('dimensions', 384),
                cache_enabled=raw.get('embedding', {}).get('cache_enabled', True),
                cache_dir=raw.get('embedding', {}).get('cache_dir', 'embedding_cache'),
            ),
            security=SecurityConfig(
                enable_access_control=raw.get('security', {}).get('enable_access_control', True),
                default_user_id=raw.get('security', {}).get('default_user_id', 'demo_user'),
                enable_content_sanitization=raw.get('security', {}).get('enable_content_sanitization', True),
                sanitization_patterns=raw.get('security', {}).get('sanitization_patterns', []),
                enable_context_tagging=raw.get('security', {}).get('enable_context_tagging', True),
                context_tag_format=raw.get('security', {}).get('context_tag_format', '[Chunk {index}]: {content}'),
            ),
            logging=LoggingConfig(
                level=raw.get('logging', {}).get('level', 'DEBUG'),
                log_queries=raw.get('logging', {}).get('log_queries', True),
                log_results=raw.get('logging', {}).get('log_results', True),
                log_embeddings=raw.get('logging', {}).get('log_embeddings', False),
                log_performance=raw.get('logging', {}).get('log_performance', True),
                debug_mode=raw.get('logging', {}).get('debug_mode', False),
                log_file=raw.get('logging', {}).get('log_file', 'rag_service.log'),
            ),
            performance=PerformanceConfig(
                batch_processing=raw.get('performance', {}).get('batch_processing', True),
                batch_size=raw.get('performance', {}).get('batch_size', 32),
                parallel_processing=raw.get('performance', {}).get('parallel_processing', False),
                num_workers=raw.get('performance', {}).get('num_workers', 4),
                result_cache_enabled=raw.get('performance', {}).get('result_cache_enabled', False),
                cache_ttl_seconds=raw.get('performance', {}).get('cache_ttl_seconds', 300),
                max_cache_size=raw.get('performance', {}).get('max_cache_size', 100),
            ),
            storage=StorageConfig(
                persist_directory=raw.get('storage', {}).get('persist_directory', 'digital_me_rag'),
                persist_enabled=raw.get('storage', {}).get('persist_enabled', True),
                backend=raw.get('storage', {}).get('backend', 'chromadb'),
                chromadb=raw.get('storage', {}).get('chromadb', {}),
            ),
            testing=TestingConfig(
                test_mode=raw.get('testing', {}).get('test_mode', False),
                sample_queries=raw.get('testing', {}).get('sample_queries', []),
                validation=raw.get('testing', {}).get('validation', {}),
            ),
        )
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to config."""
        import os
        
        if self._config is None:
            return
        
        # Override chunking
        if os.getenv('RAG_CHUNK_SIZE'):
            self._config.chunking.chunk_size = int(os.getenv('RAG_CHUNK_SIZE'))
        if os.getenv('RAG_CHUNK_OVERLAP'):
            self._config.chunking.chunk_overlap = int(os.getenv('RAG_CHUNK_OVERLAP'))
        
        # Override retrieval
        if os.getenv('RAG_MIN_CONFIDENCE'):
            self._config.retrieval.min_confidence = float(os.getenv('RAG_MIN_CONFIDENCE'))
        if os.getenv('RAG_TOP_K'):
            self._config.retrieval.default_top_k = int(os.getenv('RAG_TOP_K'))
        
        # Override embedding
        if os.getenv('RAG_EMBEDDING_PROVIDER'):
            self._config.embedding.provider = os.getenv('RAG_EMBEDDING_PROVIDER')
        if os.getenv('RAG_EMBEDDING_MODEL'):
            self._config.embedding.model = os.getenv('RAG_EMBEDDING_MODEL')
        
        # Override logging
        if os.getenv('RAG_DEBUG_MODE'):
            self._config.logging.debug_mode = os.getenv('RAG_DEBUG_MODE').lower() == 'true'
        if os.getenv('RAG_LOG_LEVEL'):
            self._config.logging.level = os.getenv('RAG_LOG_LEVEL')
        
        logger.debug("Applied environment variable overrides to RAG config")
    
    def _configure_logging(self):
        """Configure logging level based on config."""
        if self._config is None:
            return
        
        log_level = getattr(logging, self._config.logging.level.upper(), logging.INFO)
        logger.setLevel(log_level)
    
    def get_config(self) -> RAGConfig:
        """Get current configuration."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> RAGConfig:
        """Force reload configuration (for runtime updates)."""
        self._config = None
        return self.load_config()
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration dynamically.
        
        Args:
            updates: Nested dict of config updates
                    e.g., {"chunking": {"chunk_size": 600}}
        """
        if self._config is None:
            self.load_config()
        
        # Apply updates recursively
        self._apply_updates(self._config.raw, updates)
        
        # Re-parse config
        self._config = self._parse_config(self._config.raw)
        
        logger.info(f"Updated RAG config: {updates}")
    
    def _apply_updates(self, config_dict: Dict, updates: Dict):
        """Recursively apply updates to config dict."""
        for key, value in updates.items():
            if key in config_dict and isinstance(config_dict[key], dict) and isinstance(value, dict):
                self._apply_updates(config_dict[key], value)
            else:
                config_dict[key] = value
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get human-readable config summary."""
        if self._config is None:
            self.load_config()
        
        return {
            "chunking": {
                "chunk_size": self._config.chunking.chunk_size,
                "chunk_overlap": self._config.chunking.chunk_overlap,
                "strategy": self._config.chunking.strategy,
            },
            "retrieval": {
                "default_top_k": self._config.retrieval.default_top_k,
                "min_confidence": self._config.retrieval.min_confidence,
                "high_confidence_threshold": self._config.retrieval.high_confidence_threshold,
            },
            "embedding": {
                "provider": self._config.embedding.provider,
                "model": self._config.embedding.model,
            },
            "logging": {
                "level": self._config.logging.level,
                "debug_mode": self._config.logging.debug_mode,
            },
        }


# Global config loader instance
_config_loader: Optional[RAGConfigLoader] = None


def get_rag_config() -> RAGConfig:
    """
    Get RAG configuration.
    
    Returns:
        RAGConfig object with all settings
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = RAGConfigLoader()
    return _config_loader.get_config()


def reload_rag_config() -> RAGConfig:
    """
    Reload RAG configuration.
    
    Returns:
        Reloaded RAGConfig object
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = RAGConfigLoader()
    return _config_loader.reload_config()


def update_rag_config(updates: Dict[str, Any]) -> None:
    """
    Update RAG configuration dynamically.
    
    Args:
        updates: Nested dict of config updates
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = RAGConfigLoader()
    _config_loader.update_config(updates)


def get_rag_config_summary() -> Dict[str, Any]:
    """
    Get RAG configuration summary.
    
    Returns:
        Dict with key configuration values
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = RAGConfigLoader()
    return _config_loader.get_config_summary()
