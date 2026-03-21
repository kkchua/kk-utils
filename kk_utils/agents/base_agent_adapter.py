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
import os

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
    
    async def execute_chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        model: str,
        persona: Optional["PersonaConfig"] = None,
    ) -> AgentResponse:
        """
        Execute AI chat — centralized LLM call for all adapters.

        All adapters share this single implementation so that any cross-cutting
        concern (retries, token counting, new providers) only needs to be added
        in one place.

        Override post_process() to customize the response after the LLM call.

        Args:
            messages: List of message dicts [{role, content}, ...]
            tools: List of OpenAI-compatible tool schemas
            model: AI model to use (e.g., "openai/gpt-5-nano")
            persona: Full persona config (for trace name, metadata, etc.)

        Returns:
            AgentResponse with the AI reply and trace_events in metadata
        """
        try:
            response_data = await self._execute_chat_with_tools(
                messages=messages,
                tools=tools,
                model=model,
                persona=persona,
            )
            response_text = response_data.get("response", "")
            trace_events = response_data.get("trace_events", [])

            metadata: Dict[str, Any] = {
                "model": model,
                "tool_count": len(tools),
                "trace_events": trace_events,
            }
            if persona:
                metadata["persona_name"] = persona.name
                metadata["persona_display_name"] = persona.display_name
                metadata["collection"] = persona.collection

            return AgentResponse(
                response_text=response_text,
                agent_type=getattr(self, "adapter_name", "agent"),
                persona_name=persona.name if persona else "agent",
                collection=persona.collection if persona else "agent",
                tools_available=len(tools),
                metadata=metadata,
            )
        except Exception as e:
            adapter_name = getattr(self, "adapter_name", type(self).__name__)
            logger.error(f"{adapter_name} chat failed: {e}", exc_info=True)
            return AgentResponse(
                response_text="I encountered an error. Please try again.",
                agent_type=adapter_name,
                persona_name=persona.name if persona else "agent",
                collection=persona.collection if persona else "agent",
                tools_available=len(tools),
                error=f"chat_failed: {str(e)}",
                success=False,
            )
    
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
            Raw response dict from AIService with trace_events in metadata
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

        # Capture trace events for tool execution visibility
        trace_events: List[str] = []

        def trace_callback(event: str):
            """Capture trace events for display in UI."""
            trace_events.append(event)
            logger.debug(f"Trace: {event}")

        # Call AI with tools - pass trace_callback to capture tool execution events
        response = await ai_service.chat_with_tools(
            message=last_user_message,
            tools=tools,
            system_prompt=system_prompt,
            conversation_history=[m for m in messages if m["role"] in ["user", "assistant"] and m.get("role") != "system"],
            agent_name=agent_name,
            trace_callback=trace_callback,
        )

        # Return response with trace_events in metadata
        result = {"response": response} if isinstance(response, str) else response.copy()
        result["trace_events"] = trace_events
        return result

    async def generate_vision_raw(
        self,
        system_prompt: str,
        user_text: str,
        image_b64: str,
        image_mime: str,
        model: str,
    ) -> Dict[str, Any]:
        """
        Delegate vision LLM call to AIService — no LLM logic in the adapter layer.

        All LLM logic (Agents SDK, provider routing, tracing) lives in AIService.
        """
        from ..ai import AIService

        full_model = model or os.environ.get("API_MODEL", "openai/gpt-4o-mini")
        ai_service = AIService(api_model=full_model)

        logger.info(
            f"  [vision] adapter={getattr(self, 'adapter_name', type(self).__name__)} "
            f"provider={ai_service.provider} model={ai_service.model} | image_mime={image_mime}"
        )

        return await ai_service.generate_vision_raw(
            system_prompt=system_prompt,
            user_text=user_text,
            image_b64=image_b64,
            image_mime=image_mime,
        )
