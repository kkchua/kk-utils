"""
kk_utils.agents.adapters.agent_me — AgentMe Adapter

Digital twin agent with access to skills and tools.
Use for personas that represent a specific person (e.g., Keng Koon).

Skills:
- digital_me: Personal knowledge base
- notes: Note management
- web_search: Web research

Usage:
    from kk_utils.agents.adapters import AgentMeAdapter
    
    adapter = AgentMeAdapter()
    skills = adapter.get_skills()  # ["digital_me", "notes", "web_search"]
    tools = adapter._load_tools_from_registry(adapter.get_skill_tags())
    response = await adapter.execute_chat(messages, tools, model)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from kk_utils.persona_config import PersonaConfig
from kk_utils.agents.base_agent_adapter import BaseAgentAdapter

logger = logging.getLogger(__name__)

_ADAPTER_DIR = Path(__file__).parent


class AgentMeAdapter(BaseAgentAdapter):
    """
    AgentMe: Digital twin agent with skills and tools.
    
    This adapter provides:
    - Access to personal knowledge base (digital_me)
    - Note management capabilities
    - Web search functionality
    - Professional, first-person voice
    """
    
    adapter_name = "agent_me"
    _ADAPTER_DIR = _ADAPTER_DIR
    
    # Default skills for AgentMe
    # Can be overridden by persona config
    DEFAULT_SKILLS = ["digital_me", "notes", "web_search"]
    DEFAULT_SKILL_TAGS = ["digital_me", "notes", "web_search"]
    
    def get_skills(self) -> List[str]:
        """
        Return list of skill modules to load.
        
        Priority:
        1. Schema config (from prompts/{template}_schema.json)
        2. DEFAULT_SKILLS fallback
        
        Returns:
            List of skill module names
        """
        # Will be populated by MasterAgent from schema
        return self.DEFAULT_SKILLS
    
    def get_skill_tags(self) -> List[str]:
        """
        Return skill tags for tool filtering.
        
        Priority:
        1. Schema config (from prompts/{template}_schema.json)
        2. DEFAULT_SKILL_TAGS fallback
        
        Returns:
            List of skill tags
        """
        # Will be populated by MasterAgent from schema
        return self.DEFAULT_SKILL_TAGS
    
    def get_tools_config(self, persona: Optional[PersonaConfig]) -> Optional[Dict[str, Any]]:
        """Load tools config from schema."""
        return super().get_tools_config(persona)

