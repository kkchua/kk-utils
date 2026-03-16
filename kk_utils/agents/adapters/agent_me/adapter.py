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
from kk_utils.agents.agent_response import AgentResponse

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

    def build_system_prompt(self, persona: "PersonaConfig") -> str:
        """
        Build system prompt for AgentMe.

        NOTE: This method is kept for backward compatibility but is NOT used.
        MasterAgent now controls all system prompt loading via _load_system_prompt().
        """
        # This should not be called - MasterAgent handles prompt loading
        logger.warning("build_system_prompt() called - this should be handled by MasterAgent")
        return ""
    
    async def execute_chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        model: str,
        persona: Optional["PersonaConfig"] = None,
    ) -> AgentResponse:
        """
        Execute chat with tools.
        
        Args:
            messages: List of message dicts
            tools: List of tool schemas
            model: AI model to use
            persona: Full persona config object (for trace name, metadata, etc.)
            
        Returns:
            AgentResponse with the AI reply
        """
        try:
            # Execute chat using base class helper
            # Pass persona for trace name and metadata extraction
            response_data = await self._execute_chat_with_tools(
                messages=messages,
                tools=tools,
                model=model,
                persona=persona,
            )
            
            response_text = response_data.get("response", "")
            
            # Extract metadata from persona if available
            metadata = {
                "model": model,
                "tool_count": len(tools),
            }
            if persona:
                metadata["persona_name"] = persona.name
                metadata["persona_display_name"] = persona.display_name
                metadata["collection"] = persona.collection
            
            return AgentResponse(
                response_text=response_text,
                agent_type="agent_me",
                persona_name=persona.name if persona else "agent_me",
                collection=persona.collection if persona else "agent_me",
                tools_available=len(tools),
                metadata=metadata,
            )
            
        except Exception as e:
            logger.error(f"AgentMeAdapter chat failed: {e}", exc_info=True)
            return AgentResponse(
                response_text="I encountered an error. Please try again.",
                agent_type="agent_me",
                persona_name=persona.name if persona else "agent_me",
                collection=persona.collection if persona else "agent_me",
                tools_available=len(tools),
                error=f"chat_failed: {str(e)}",
                success=False,
            )
