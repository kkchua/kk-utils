"""
kk_utils.agents.coder — Coder Agent Architecture

Direct coder CLI invocation (subprocess) as an alternative to API-based AI service.
All CLI parameters are data-driven via model_mapping.json — no hardcoding.

Architecture:
    BaseCoderAdapter (abstract interface)
        ↓
    Concrete adapters (DescImageCoderAdapter, CsvGeneratorCoderAdapter, etc.)
        ↓
    invoke_coder() — builds command from cli_params → subprocess → sidecar polling

Model Resolution:
    coder alias → model_mapping.json → full config with cli_params

Sidecar Contract:
    meta.json written by coder → read & validate → enrich with runner_data

Usage:
    from kk_utils.agents.coder import DescImageCoderAdapter

    adapter = DescImageCoderAdapter()
    response = await adapter.execute_coder(
        prompt_text="Describe this image",
        context={"image_path": "images/photo.jpg"},
    )
"""

from .coder_response import CoderResponse
from .base_coder_adapter import BaseCoderAdapter
from .coder_registry import (
    CoderRegistry,
    register_adapter,
    get_adapter,
    list_adapters,
)
from .model_resolver import resolve_coder, load_model_mapping, get_api_key, clear_cache
from .coder_invoker import (
    invoke_coder,
    build_command,
    InvocationResult,
    UsageData,
    InvocationManifest,
    CoderInvocationError,
)
from .sidecar import (
    resolve_meta_json_path,
    read_and_validate_meta_json,
    enrich_sidecar,
    validate_artifact_files,
    compute_prompt_checksum,
    MetaJsonMissingError,
    MetaJsonInvalidError,
    ArtifactMissingError,
)

# Adapter implementations
from .adapters import (
    DescImageCoderAdapter,
    CsvGeneratorCoderAdapter,
)

__all__ = [
    # Response
    "CoderResponse",

    # Base class
    "BaseCoderAdapter",

    # Registry
    "CoderRegistry",
    "register_adapter",
    "get_adapter",
    "list_adapters",

    # Model resolution
    "resolve_coder",
    "load_model_mapping",
    "get_api_key",
    "clear_cache",

    # Invocation
    "invoke_coder",
    "build_command",
    "InvocationResult",
    "UsageData",
    "InvocationManifest",
    "CoderInvocationError",

    # Sidecar contract
    "resolve_meta_json_path",
    "read_and_validate_meta_json",
    "enrich_sidecar",
    "validate_artifact_files",
    "compute_prompt_checksum",
    "MetaJsonMissingError",
    "MetaJsonInvalidError",
    "ArtifactMissingError",

    # Built-in adapters
    "DescImageCoderAdapter",
    "CsvGeneratorCoderAdapter",
]
