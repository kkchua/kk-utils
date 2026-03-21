"""
kk_utils.agents.adapters.ai_assistant — AI Assistant Adapter

General conversational AI assistant with optional tools.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from kk_utils.persona_config import PersonaConfig
from kk_utils.agents.base_agent_adapter import BaseAgentAdapter

logger = logging.getLogger(__name__)

_ADAPTER_DIR = Path(__file__).parent


class AIAssistantAdapter(BaseAgentAdapter):
    """AI Assistant: General conversational AI."""

    adapter_name = "ai_assistant"
    _ADAPTER_DIR = _ADAPTER_DIR
    DEFAULT_SKILLS = []
    DEFAULT_SKILL_TAGS = []

    def get_skills(self) -> List[str]:
        return self.DEFAULT_SKILLS

    def get_skill_tags(self) -> List[str]:
        return self.DEFAULT_SKILL_TAGS

    def get_tools_config(self, persona: Optional[PersonaConfig]) -> Optional[Dict[str, Any]]:
        """Load tools config from schema."""
        return super().get_tools_config(persona)
