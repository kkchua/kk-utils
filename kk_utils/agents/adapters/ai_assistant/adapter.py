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
from kk_utils.agents.agent_response import AgentResponse

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
    
    def build_system_prompt(self, persona: PersonaConfig) -> str:
        if persona.system_prompt and persona.system_prompt.strip():
            return persona.system_prompt
        template_name = persona.adapter_prompt_template or "default"
        try:
            return self.load_prompt_template(template_name)
        except FileNotFoundError:
            return self._get_fallback_prompt()
    
    def _get_fallback_prompt(self) -> str:
        return """You are a helpful, friendly AI assistant.

Guidelines:
- Be concise but thorough
- Be friendly and approachable
- Provide accurate information
- Ask clarifying questions when needed
"""
    
    async def execute_chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        model: str,
        persona: Optional["PersonaConfig"] = None,
    ) -> AgentResponse:
        """
        Execute chat with optional tools.
        
        Args:
            messages: List of message dicts
            tools: List of tool schemas (may be empty)
            model: AI model to use
            persona: Full persona config object (for trace name, metadata, etc.)
            
        Returns:
            AgentResponse with the AI reply
        """
        try:
            # Extract trace name from persona if available
            agent_name = persona.display_name if persona else "AI Assistant"
            
            if not tools:
                response_data = await self._execute_simple_chat(messages, model)
            else:
                response_data = await self._execute_chat_with_tools(
                    messages=messages,
                    tools=tools,
                    model=model,
                    persona=persona,
                )

            response_text = response_data.get("response", "")
            
            # Extract metadata from persona if available
            metadata = {"model": model, "tool_count": len(tools)}
            if persona:
                metadata["persona_name"] = persona.name
                metadata["persona_display_name"] = persona.display_name
                metadata["collection"] = persona.collection

            return AgentResponse(
                response_text=response_text,
                agent_type="ai_assistant",
                persona_name=persona.name if persona else "ai_assistant",
                collection=persona.collection if persona else "ai_assistant",
                tools_available=len(tools),
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"AIAssistantAdapter chat failed: {e}", exc_info=True)
            return AgentResponse(
                response_text="I encountered an error. Please try again.",
                agent_type="ai_assistant",
                persona_name=persona.name if persona else "ai_assistant",
                collection=persona.collection if persona else "ai_assistant",
                tools_available=len(tools),
                error=f"chat_failed: {str(e)}",
                success=False,
            )
    
    async def _execute_simple_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
    ) -> Dict[str, Any]:
        """Execute chat without tools (pure conversation)."""
        from ...ai import AIService
        
        ai_service = AIService(model=model)
        
        # Extract system prompt
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_messages.append(msg)
        
        # Call AI without tools
        response = await ai_service.generate_text(
            prompt=user_messages[-1]["content"] if user_messages else "",
            system_prompt=system_prompt,
        )
        
        return {"response": response.text} if hasattr(response, 'text') else {"response": str(response)}
    
    async def _execute_chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        model: str,
    ) -> Dict[str, Any]:
        """Execute chat with tools."""
        return await super()._execute_chat_with_tools(messages, tools, model)