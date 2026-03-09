"""
KK-Utils - Core Utility Library

Common utilities for Python projects:
- Environment loading with fail-fast
- Centralized logging configuration
- YAML config loading
- Path resolution helpers
- RAG client for Personal Assistant API
- RAG core engine (direct ChromaDB access)
- AI service (multi-provider, OpenAI Agents SDK)

Usage:
    from kk_utils import load_environment, setup_logging, ConfigLoader, RAGClient
    from kk_utils.ai import get_ai_service

    # For direct RAG access (no API):
    from kk_utils.rag import RAGEngine, RAGConfig

Version: 1.0.0
Author: KK
"""

from kk_utils.env_loader import load_environment, is_environment_loaded, get_env_path
from kk_utils.logging_config import setup_logging, get_logger, LogContext, log_function_call
from kk_utils.config_loader import ConfigLoader
from kk_utils.path_resolver import (
    get_project_root,
    get_backend_root,
    get_config_path,
    get_logs_path,
)
from kk_utils.rag_client import RAGClient

# RAG Core Engine (direct ChromaDB access)
from kk_utils.rag import (
    RAGEngine,
    RAGResult,
    RAGConfig,
    ChunkingStrategy,
    WordChunker,
    SentenceChunker,
    EmbeddingProvider,
    get_embedding_function,
)

# Agent Tools infrastructure
from kk_utils.agent_tools import (
    agent_tool,
    AgentRegistry,
    get_registry,
    register_tool,
    get_tool,
    execute_tool,
    _auto_register,
)

# Persona & Factory
from kk_utils.persona_config import PersonaConfig, load_persona, list_personas
from kk_utils.factory import AgentMeFactory, AgentConfig

# Skill Manifest
from kk_utils.skill_manifest import SkillManifest, get_skill_manifest, discover_skills

# AI Service (multi-provider, OpenAI Agents SDK)
from kk_utils.ai import (
    AIService,
    CallContext,
    TextResult,
    SummaryResult,
    RewriteResult,
    TaskExtractionResult,
    IntentClassificationResult,
    get_ai_service,
)

__version__ = "1.0.0"
__author__ = "KK"
__all__ = [
    # Environment
    "load_environment",
    "is_environment_loaded",
    "get_env_path",

    # Logging
    "setup_logging",
    "get_logger",
    "LogContext",
    "log_function_call",

    # Config
    "ConfigLoader",

    # Paths
    "get_project_root",
    "get_backend_root",
    "get_config_path",
    "get_logs_path",

    # RAG Client (API)
    "RAGClient",

    # RAG Core (Direct ChromaDB)
    "RAGEngine",
    "RAGResult",
    "RAGConfig",
    "ChunkingStrategy",
    "WordChunker",
    "SentenceChunker",
    "EmbeddingProvider",
    "get_embedding_function",

    # Agent Tools
    "agent_tool",
    "AgentRegistry",
    "get_registry",
    "register_tool",
    "get_tool",
    "execute_tool",
    "_auto_register",

    # AI Service
    "AIService",
    "CallContext",
    "TextResult",
    "SummaryResult",
    "RewriteResult",
    "TaskExtractionResult",
    "IntentClassificationResult",
    "get_ai_service",

    # Persona & Factory
    "PersonaConfig",
    "load_persona",
    "list_personas",
    "AgentConfig",
    "AgentMeFactory",

    # Skill Manifest
    "SkillManifest",
    "get_skill_manifest",
    "discover_skills",
]
