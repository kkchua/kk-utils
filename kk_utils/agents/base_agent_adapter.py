"""
kk_utils.agents.base_agent_adapter — Abstract base for all agent adapters

Defines the interface that all agent adapters must implement.
Adapters are responsible for:
- Defining which skills/tools to use
- Building system prompts
- Executing chat via AIService

Usage:
    class MyCustomAdapter(BaseAgentAdapter):
        adapter_name = "my_custom_agent"
        
        def get_skills(self) -> List[str]:
            return ["web_search", "notes"]
        
        def get_skill_tags(self) -> List[str]:
            return ["web_search", "notes"]
        
        def build_system_prompt(self, persona: PersonaConfig) -> str:
            return "You are a custom assistant..."
        
        async def execute_chat(...) -> AgentResponse:
            # Call AIService with tools
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)

from .agent_response import AgentResponse
from ..persona_config import PersonaConfig


class BaseAgentAdapter(ABC):
    """
    Abstract base class for all agent adapters.
    
    All adapters must implement:
    - get_skills(): Return list of skill modules to load
    - get_skill_tags(): Return tags for tool filtering
    - build_system_prompt(): Build system prompt for the agent
    - execute_chat(): Execute AI chat with tools
    
    Optionally override:
    - post_process(): Transform response before returning
    - load_prompt_template(): Load prompt from file (provided)
    - load_schema(): Load schema from file (provided)
    """
    
    adapter_name: str
    _ADAPTER_DIR: Path
    
    def __init__(self):
        """Initialize the adapter."""
        pass
    
    @abstractmethod
    def get_skills(self) -> List[str]:
        """
        Return list of skill module names to load.
        
        These are module names under kk_agent_skills/ that will be
        imported to register tools.
        
        Example:
            return ["digital_me", "notes", "web_search"]
        
        Returns:
            List of skill module names
        """
        pass
    
    @abstractmethod
    def get_skill_tags(self) -> List[str]:
        """
        Return skill tags for tool filtering.
        
        Tools are filtered by these tags from the AgentRegistry.
        
        Example:
            return ["digital_me", "notes", "web_search"]
        
        Returns:
            List of skill tags
        """
        pass
    
    @abstractmethod
    def build_system_prompt(self, persona: PersonaConfig) -> str:
        """
        Build system prompt for the agent.
        
        Can use:
        - persona.system_prompt (from YAML config)
        - persona.adapter_prompt_template (template name)
        - Custom logic
        
        Args:
            persona: Persona configuration
            
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    async def execute_chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        model: str,
    ) -> AgentResponse:
        """
        Execute AI chat with tools.
        
        This is the core method that calls AIService.
        
        Args:
            messages: List of message dicts [{role, content}, ...]
            tools: List of OpenAI-compatible tool schemas
            model: AI model to use (e.g., "openai/gpt-5-nano")
            
        Returns:
            AgentResponse with the AI reply
        """
        pass
    
    def post_process(self, response: AgentResponse) -> AgentResponse:
        """
        Optional: Transform response before returning.
        
        Override to augment or modify the response.
        
        Args:
            response: Agent response to process
            
        Returns:
            Processed response
        """
        return response
    
    def load_prompt_template(self, template_name: str) -> str:
        """
        Load prompt template from prompts/{template_name}.txt
        
        Args:
            template_name: Name of template file (without .txt)
            
        Returns:
            Prompt template content
            
        Raises:
            FileNotFoundError: If template doesn't exist
        """
        prompt_file = self._ADAPTER_DIR / "prompts" / f"{template_name}.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8")
    
    def load_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """
        Load schema from prompts/{schema_name}.json
        
        Args:
            schema_name: Name of schema file (without .json)
            
        Returns:
            Schema dict or None if not found
        """
        schema_file = self._ADAPTER_DIR / "prompts" / f"{schema_name}.json"
        if not schema_file.exists():
            return None
        import json
        return json.loads(schema_file.read_text(encoding="utf-8"))
    
    def get_tools_config(self, persona: PersonaConfig) -> Optional[Dict[str, Any]]:
        """
        Load tools config from schema JSON.
        
        Schema filename is derived from adapter_prompt_template:
        - adapter_prompt_template="default" → loads default_schema.json
        - adapter_prompt_template="friendly" → loads friendly_schema.json
        
        Args:
            persona: Persona configuration
            
        Returns:
            Schema dict or None if not found
        """
        template_name = persona.adapter_prompt_template or "default"
        schema_name = f"{template_name}_schema"
        return self.load_schema(schema_name)
    
    def _load_tools_from_registry(self, skill_tags: List[str]) -> List[Dict[str, Any]]:
        """
        Load tools from AgentRegistry by skill tags.
        
        Args:
            skill_tags: Tags to filter tools by
            
        Returns:
            List of OpenAI-compatible tool schemas
        """
        from ..agent_tools import get_registry
        registry = get_registry()
        return registry.get_tools_for_tags(skill_tags)
    
    async def _execute_chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        model: str,
        persona: Optional["PersonaConfig"] = None,
    ) -> Dict[str, Any]:
        """
        Execute chat using kk_utils.ai.AIService.

        Helper method for execute_chat implementations.

        Args:
            messages: List of message dicts
            tools: List of tool schemas
            model: AI model to use
            persona: Full persona config object (for trace name, metadata, etc.)

        Returns:
            Raw response dict from AIService
        """
        from ..ai import AIService

        ai_service = AIService(api_model=model)

        # Extract system prompt and last user message
        system_prompt = ""
        last_user_message = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                last_user_message = msg["content"]

        # Extract trace name from persona if available
        agent_name = None
        if persona:
            agent_name = persona.display_name or persona.name

        # GOVERNOR: Validate tool call limits BEFORE calling AI
        # This prevents excessive tool calls that waste money
        try:
            from app.core.governor import PersonalAssistantGovernor
            governor = PersonalAssistantGovernor.instance()
            
            # Get tool schemas that would be available
            tool_names = [t.get("function", {}).get("name", "") for t in tools]
            logger.info(f"Available tools for {agent_name}: {len(tool_names)} tools")
            
            # Note: We can't validate actual tool calls yet (LLM hasn't made them)
            # But we can log the available tools for audit
            logger.info(f"Governor: Agent {agent_name} has {len(tools)} tools available")
            
        except ImportError:
            # Governor not available (running outside backend)
            logger.debug("Governor not available - skipping tool call validation")
        except Exception as e:
            logger.warning(f"Governor tool validation error: {e}")

        # Call AI with tools
        response = await ai_service.chat_with_tools(
            message=last_user_message,
            tools=tools,
            system_prompt=system_prompt,
            conversation_history=[m for m in messages if m["role"] in ["user", "assistant"] and m.get("role") != "system"],
            agent_name=agent_name,
        )

        return {"response": response} if isinstance(response, str) else response
