"""
kk_utils.ai — AI service and adapter pipeline module

Provides backend-agnostic AI text processing via the OpenAI Agents SDK,
plus the adapter pattern for structured, file-driven LLM calls.

Usage:

    # Basic AI service
    from kk_utils.ai import get_ai_service, AIService, CallContext

    # Structured output types
    from kk_utils.ai import TextResult, SummaryResult, RewriteResult

    # Adapter pipeline
    from kk_utils.ai import AIRunner, BaseAIAdapter, SchemaAdapterMixin
    from kk_utils.ai.schema_adapter_mixin import build_output_type_schema
"""

from kk_utils.ai.ai_service import (
    AIService,
    CallContext,
    TextResult,
    SummaryResult,
    RewriteResult,
    TaskExtractionResult,
    IntentClassificationResult,
    get_ai_service,
)

from kk_utils.ai.base_ai_adapter import BaseAIAdapter
from kk_utils.ai.schema_adapter_mixin import SchemaAdapterMixin, build_output_type_schema
from kk_utils.ai.ai_runner import AIRunner

__all__ = [
    # AI service
    "AIService",
    "CallContext",
    "TextResult",
    "SummaryResult",
    "RewriteResult",
    "TaskExtractionResult",
    "IntentClassificationResult",
    "get_ai_service",
    # Adapter pipeline
    "BaseAIAdapter",
    "SchemaAdapterMixin",
    "build_output_type_schema",
    "AIRunner",
]
