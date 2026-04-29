"""
kk_utils.agents.coder.sidecar — meta.json sidecar contract

Coder agents communicate results via a meta.json sidecar file written
to disk. This module handles:
- Resolving expected meta.json path from context/artifact key
- Reading and validating the sidecar (v2 schema)
- Enriching with runner_data (invocation metadata)
- Validating artifact files exist on disk

Schema (v2):
    {
      "schema_version": "v2",
      "coder_result": {
        "status": "APPROVED" | "REJECTED",
        "remark": "...",
        "artifacts": {"KEY": "relative/path"},
        "reject_code": null,
        "reject_type": null,
        "recorded_at": "2026-04-26T..."
      },
      "runner_data": { ... }  // appended by enrich_sidecar
    }

Usage:
    from kk_utils.agents.coder.sidecar import (
        resolve_meta_json_path,
        read_and_validate_meta_json,
        enrich_sidecar,
        validate_artifact_files,
    )

    meta_path = resolve_meta_json_path(artifact_path="images/desc.json")
    meta = read_and_validate_meta_json(meta_path)
    enrich_sidecar(meta_path, step="desc_image", ...)
    validate_artifact_files(meta["coder_result"]["artifacts"], cwd)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import tempfile
from pathlib import Path, PurePath
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class MetaJsonMissingError(Exception):
    """Coder did not write meta.json to expected path."""
    pass


class MetaJsonInvalidError(Exception):
    """meta.json present but schema invalid."""
    pass


class ArtifactMissingError(Exception):
    """meta.json referenced a file that doesn't exist on disk."""
    def __init__(self, message: str, missing: Optional[list] = None):
        super().__init__(message)
        self.missing = missing or []


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_meta_json_path(artifact_path: str) -> Path:
    """
    Derive meta.json path from an artifact path.

    E.g., "images/desc.json" → "images/desc.meta.json"

    Args:
        artifact_path: Relative artifact path

    Returns:
        Path object for the expected meta.json location
    """
    p = PurePath(artifact_path)
    meta_relative = str(p.parent / f"{p.stem}.meta.json")
    return Path(meta_relative)


# ---------------------------------------------------------------------------
# Read & validate
# ---------------------------------------------------------------------------

def read_and_validate_meta_json(path: Path) -> Dict[str, Any]:
    """
    Read and validate coder-written meta.json.

    Accepts:
    - v2 format: schema_version = "v2"

    Args:
        path: Path to meta.json

    Returns:
        Parsed and validated meta dict (with normalized status case)

    Raises:
        MetaJsonMissingError: If file absent
        MetaJsonInvalidError: If schema invalid
    """
    if not path.exists() or not path.is_file():
        raise MetaJsonMissingError(
            f"Coder did not write meta.json to expected path: {path}"
        )

    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise MetaJsonInvalidError(
            f"meta.json at {path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(meta, dict):
        raise MetaJsonInvalidError(f"meta.json at {path} is not a JSON object")

    # Validate schema version
    schema_v = str(meta.get("schema_version") or "").strip()
    sidecar_v = str(meta.get("sidecar_version") or "").strip()
    if schema_v not in ("v2",) and sidecar_v not in ("artifact_meta_v1",):
        raise MetaJsonInvalidError(
            f"meta.json at {path} has unrecognised version: "
            f"schema_version={schema_v!r}, sidecar_version={sidecar_v!r}. "
            "Expected schema_version='v2'."
        )

    coder_result = meta.get("coder_result")
    if not isinstance(coder_result, dict):
        raise MetaJsonInvalidError(
            f"meta.json at {path} is missing coder_result object"
        )

    # Validate and normalize status
    status = str(coder_result.get("status") or "").strip().upper()
    if status not in ("APPROVED", "REJECTED"):
        raise MetaJsonInvalidError(
            f"meta.json at {path} has invalid coder_result.status: {status!r}. "
            "Must be 'APPROVED' or 'REJECTED'."
        )
    coder_result["status"] = status  # normalize case in-place

    if not isinstance(coder_result.get("artifacts"), dict):
        raise MetaJsonInvalidError(
            f"meta.json at {path} is missing coder_result.artifacts object"
        )

    recorded_at = str(coder_result.get("recorded_at") or "").strip()
    if not recorded_at:
        raise MetaJsonInvalidError(
            f"meta.json at {path} is missing coder_result.recorded_at"
        )

    return meta


# ---------------------------------------------------------------------------
# Sidecar enrichment
# ---------------------------------------------------------------------------

def enrich_sidecar(
    *,
    meta_path: Path,
    step: str,
    coder_used: str,
    invoked_at: str,
    finished_at: str,
    prompt_checksum: str,
    project_root: Optional[Path] = None,
) -> None:
    """
    Atomically append runner_data section to existing meta.json.

    Never modifies coder_result. Idempotent — overwrites runner_data if present.

    Args:
        meta_path: Path to meta.json
        step: Step/adapter name
        coder_used: Coder binary name (e.g., "qwen")
        invoked_at: ISO timestamp of invocation start
        finished_at: ISO timestamp of invocation end
        prompt_checksum: SHA256 of prompt text
        project_root: Optional project root for path resolution
    """
    if not meta_path.exists():
        logger.warning(f"Cannot enrich sidecar — meta.json not found at {meta_path}")
        return

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Cannot enrich sidecar — meta.json at {meta_path} is not valid JSON")
        return

    meta["runner_data"] = {
        "step": step,
        "coder_used": coder_used,
        "invoked_at": invoked_at,
        "finished_at": finished_at,
        "prompt_checksum": f"sha256:{prompt_checksum}",
        "enriched_at": _now_iso(),
        "runner_version": "kk-utils-coder-v1",
    }

    # Atomic write: write to temp, then replace
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=meta_path.parent, prefix=".tmp_", suffix=".json"
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        Path(tmp_path).replace(meta_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------

def validate_artifact_files(
    artifacts: Dict[str, str],
    *,
    project_root: Path,
) -> None:
    """
    Raise ArtifactMissingError if any artifact path doesn't exist on disk.

    Args:
        artifacts: Dict of key → relative path from meta.json
        project_root: Project root directory
    """
    missing = [
        path_str
        for path_str in artifacts.values()
        if path_str and not (project_root / path_str).exists()
    ]
    if missing:
        raise ArtifactMissingError(
            f"Artifact files claimed in meta.json do not exist on disk: {missing}",
            missing=missing,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current time as ISO 8601 string."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def compute_prompt_checksum(prompt_text: str) -> str:
    """Compute SHA256 checksum of prompt text."""
    import hashlib
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
