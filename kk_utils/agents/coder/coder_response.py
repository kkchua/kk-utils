"""
kk_utils.agents.coder.coder_response — Coder response dataclass

Standardized response format for all coder adapters.
Extends AgentResponse with coder-specific fields (return code, usage, artifacts).

Usage:
    from kk_utils.agents.coder import CoderResponse

    response = CoderResponse(
        response_text="Image description generated",
        agent_type="coder_desc_image",
        persona_name="desc_image",
        collection="",
        return_code=0,
        artifacts={"IMAGE_DESC_FOLDER": "images/descriptions/"},
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from ..agent_response import AgentResponse


@dataclass
class CoderResponse(AgentResponse):
    """
    Standardized response from any coder adapter.

    Extends AgentResponse with coder-specific fields:
    - meta_json_path: Path to the meta.json sidecar that was read
    - return_code: Coder CLI exit code (0 = success)
    - usage: Token usage, cost, duration from invocation
    - artifacts: Dict of artifact key → relative path from meta.json
    - reject_code: Coder rejection code (from meta.json)
    - raw_events: List of raw stdout lines/events from coder
    """
    meta_json_path: str = ""
    return_code: int = 0
    usage: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    reject_code: Optional[str] = None
    raw_events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        base = super().to_dict()
        base.update({
            "meta_json_path": self.meta_json_path,
            "return_code": self.return_code,
            "usage": self.usage,
            "artifacts": self.artifacts,
            "reject_code": self.reject_code,
        })
        return base
