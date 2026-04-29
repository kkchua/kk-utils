"""
kk_utils.agents.coder.model_resolver — Model alias resolver

Loads model_mapping.json from kk_utils/config/ and resolves coder
aliases (e.g., "deepseek-chat") into full invocation configs with
cli_params. Plain coder names ("qwen", "claude", "codex") pass
through with default cli_params.

model_mapping.json structure:
    {
      "coder_aliases": {
        "desc_image": {
          "coder": "qwen",
          "model": "qwen-coder-plus-latest",
          "cli_params": { ... },
          ...
        }
      }
    }

Usage:
    from kk_utils.agents.coder import resolve_coder, load_model_mapping

    config = resolve_coder("desc_image")
    # → {"coder": "qwen", "model": "...", "cli_params": {...}, ...}
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_MAPPING: Optional[Dict[str, Dict[str, Any]]] = None
_MAPPING_PATH: Optional[Path] = None


def _config_dir() -> Path:
    """Get kk_utils/config directory."""
    return Path(__file__).parent.parent.parent / "config"


def _resolve_mapping_path() -> Path:
    """
    Resolve model mapping file path based on APP_ENV.

    Priority:
    1. If APP_ENV is set → model_mapping_{APP_ENV}.json
    2. Fallback → model_mapping.json

    Returns:
        Path to the resolved model mapping file
    """
    app_env = os.environ.get("APP_ENV", "").strip()
    if app_env:
        env_path = _config_dir() / f"model_mapping_{app_env}.json"
        if env_path.exists():
            logger.info(f"Using environment model mapping: model_mapping_{app_env}.json")
            return env_path
        logger.warning(
            f"model_mapping_{app_env}.json not found, "
            f"falling back to model_mapping.json"
        )
    return _config_dir() / "model_mapping.json"


def _mapping_path() -> Path:
    """Get resolved model_mapping path (env-aware)."""
    return _resolve_mapping_path()


# ---------------------------------------------------------------------------
# Default CLI params for known coders (fallback if not in mapping)
# ---------------------------------------------------------------------------

DEFAULT_CLI_PARAMS: Dict[str, Dict[str, Any]] = {
    "qwen": {
        "cmd": ["qwen"],
        "flags": ["--output-format", "json", "--approval-mode", "yolo"],
        "prompt_flag": "-p",
        "input_flag": None,
        "model_flag": "-m",
        "schema_flag": None,
        "schema_inline": True,
        "api_key_flag": "--openai-api-key",
        "base_url_flag": "--openai-base-url",
    },
    "claude": {
        "cmd": ["claude"],
        "flags": ["--permission-mode", "bypassPermissions", "--print", "--output-format", "json"],
        "prompt_flag": None,
        "input_flag": "stdin",
        "model_flag": None,
        "schema_flag": "--json-schema",
        "schema_inline": True,
        "api_key_flag": None,
        "base_url_flag": None,
    },
    "codex": {
        "cmd": ["codex", "exec"],
        "flags": ["--sandbox", "workspace-write", "--json"],
        "prompt_flag": None,
        "input_flag": "stdin",
        "model_flag": None,
        "schema_flag": "--output-schema",
        "schema_inline": False,
        "api_key_flag": None,
        "base_url_flag": None,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model_mapping(path: Optional[Path | str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load the model mapping file.

    If *path* is not given, reads ``model_mapping.json`` from kk_utils/config/.
    Results are cached for the lifetime of the process.

    Args:
        path: Optional path to model_mapping.json

    Returns:
        Dict of coder alias → config dict
    """
    global _MAPPING, _MAPPING_PATH

    if _MAPPING is not None and _MAPPING_PATH is not None:
        return _MAPPING

    resolved = Path(path) if path else _mapping_path()
    _MAPPING_PATH = resolved

    if not resolved.exists():
        logger.warning(f"model_mapping.json not found at {resolved}, using empty mapping")
        _MAPPING = {}
        return _MAPPING

    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
        _MAPPING = data.get("coder_aliases", {})
        logger.info(f"Loaded model_mapping.json with {len(_MAPPING)} coder aliases")
        return _MAPPING
    except Exception as e:
        logger.error(f"Failed to load model_mapping.json: {e}")
        _MAPPING = {}
        return _MAPPING


def resolve_coder(name: str, *, mapping_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """
    Resolve a coder name into a full invocation config.

    If *name* matches a key in ``coder_aliases``, returns the config dict
    (which includes ``cli_params``). Otherwise, builds a default config
    using ``DEFAULT_CLI_PARAMS`` for the coder name.

    Args:
        name: Coder alias (e.g., "desc_image", "deepseek-chat") or
              plain coder name ("qwen", "claude", "codex")
        mapping_path: Optional path to model_mapping.json

    Returns:
        Full coder config dict with at minimum:
        - "coder": coder binary name
        - "cli_params": CLI parameter spec
    """
    aliases = load_model_mapping(path=mapping_path)
    config = aliases.get(name)

    if config is not None:
        # Ensure cli_params exists (may be missing from mapping)
        if "cli_params" not in config:
            coder_name = config.get("coder", name)
            config["cli_params"] = DEFAULT_CLI_PARAMS.get(coder_name, {}).copy()
        return config.copy()

    # Fallback: treat as plain coder name with defaults
    if name in DEFAULT_CLI_PARAMS:
        logger.debug(f"Coder '{name}' not in mapping, using default cli_params")
        return {
            "coder": name,
            "cli_params": DEFAULT_CLI_PARAMS[name].copy(),
        }

    # Unknown coder — return minimal config
    logger.warning(f"Unknown coder '{name}', returning minimal config")
    return {
        "coder": name,
        "cli_params": DEFAULT_CLI_PARAMS.get("qwen", {}).copy(),
    }


def get_api_key(coder_config: Dict[str, Any]) -> Optional[str]:
    """
    Retrieve the API key for a coder config from environment variables.

    Args:
        coder_config: Coder config dict with "openai_api_key_env" field

    Returns:
        API key value or None
    """
    env_key = coder_config.get("openai_api_key_env")
    if env_key:
        return os.environ.get(env_key)
    return None


def clear_cache() -> None:
    """Clear the cached model mapping — for testing."""
    global _MAPPING, _MAPPING_PATH
    _MAPPING = None
    _MAPPING_PATH = None
