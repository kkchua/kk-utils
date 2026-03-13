"""
kk_utils.agents.master_agent — Master Agent orchestrator

Backend-agnostic agent orchestrator that:
- Resolves persona from config
- Selects appropriate adapter based on persona.adapter_type
- Loads skills and tools via adapter
- Delegates chat execution to adapter
- Returns standardized AgentResponse

Usage:
    from kk_utils.agents import MasterAgent
    
    agent = MasterAgent(personas_config_path="config/personas.yaml")
    response = await agent.chat(
        message="Hello",
        persona_name="ai_assistant",
        user_id="user123",
        user_role="demo",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .agent_response import AgentResponse
from .base_agent_adapter import BaseAgentAdapter
from .agent_registry import AgentRegistry
from ..persona_config import PersonaConfig, load_persona

logger = logging.getLogger(__name__)


class MasterAgent:
    """
    Master Agent - backend-agnostic orchestrator.
    
    Reusable across:
    - personal-assistant backend
    - gradio-apps
    - future projects
    
    Responsibilities:
    - Load persona from config
    - Select adapter based on persona.adapter_type
    - Load skills and tools via adapter
    - Execute chat via adapter
    - Return standardized AgentResponse
    
    NOT responsible for:
    - Governor access control (backend-specific)
    - Guardrails (backend-specific)
    - Audit logging (backend-specific)
    - Usage tracking (backend-specific)
    """
    
    def __init__(
        self,
        personas_config_path: Optional[str] = None,
        auto_register_adapters: bool = True,
    ):
        """
        Initialize Master Agent.
        
        Args:
            personas_config_path: Path to personas.yaml (required for chat)
            auto_register_adapters: If True, auto-register built-in adapters
        """
        self.personas_config_path = Path(personas_config_path) if personas_config_path else None
        self.adapter_registry = AgentRegistry.instance()
        
        if auto_register_adapters:
            self._register_builtin_adapters()
    
    def _register_builtin_adapters(self) -> None:
        """Register built-in adapters."""
        try:
            from .adapters import AgentMeAdapter, AIAssistantAdapter

            self.adapter_registry.register("agent_me", AgentMeAdapter, override=True)
            self.adapter_registry.register("ai_assistant", AIAssistantAdapter, override=True)
            logger.info("Registered built-in agent adapters: agent_me, ai_assistant")
        except ImportError as e:
            logger.warning(f"Could not register built-in adapters: {e}")
    
    def set_personas_config_path(self, path: str) -> None:
        """
        Set personas config path.
        
        Args:
            path: Path to personas.yaml
        """
        self.personas_config_path = Path(path)
    
    async def chat(
        self,
        message: str,
        persona_name: str,
        user_id: str,
        user_role: str = "demo",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model: str = "openai/gpt-5-nano",
    ) -> AgentResponse:
        """
        Process a chat message.
        
        This is backend-agnostic - no Governor, no audit, no usage tracking.
        Those are added by the backend's OrchestratorService.
        
        Args:
            message: User message
            persona_name: Name of persona to use
            user_id: User identifier
            user_role: User role (for adapter context)
            conversation_history: Previous messages [{role, content}, ...]
            model: AI model to use
            
        Returns:
            AgentResponse with the AI reply
            
        Raises:
            ValueError: If persona not found or config path not set
            KeyError: If adapter not found
        """
        if not self.personas_config_path:
            raise ValueError("personas_config_path not set")
        
        # 1. Load persona
        persona = load_persona(persona_name, config_path=self.personas_config_path)
        if persona is None:
            raise ValueError(f"Persona '{persona_name}' not found")
        
        # 2. Get adapter
        adapter_type = persona.adapter_type or "agent_me"
        adapter_class = self.adapter_registry.get_adapter(adapter_type)
        adapter: BaseAgentAdapter = adapter_class()
        
        logger.info(f"MasterAgent: persona={persona_name!r} adapter={adapter_type!r}")

        # 3. Load schema config (optional)
        schema_config = adapter.get_tools_config(persona) or {}

        # 4. Load skills - Priority: schema config > persona config > adapter default
        skills = schema_config.get("skills") or persona.skills or adapter.get_skills()
        skill_tags = schema_config.get("skill_tags") or persona.skill_tags or adapter.get_skill_tags()

        # 5. Load tools from registry
        tools = self._load_tools(skill_tags)
        
        # 6. Build system prompt
        system_prompt = adapter.build_system_prompt(persona)
        
        # 7. Build messages
        messages = self._build_messages(system_prompt, conversation_history, message)
        
        # 8. Execute chat via adapter
        # Pass full persona object for trace name and metadata extraction
        response = await adapter.execute_chat(
            messages=messages,
            tools=tools,
            model=model,
            persona=persona,
        )
        
        # 9. Post-process (adapter-specific)
        response = adapter.post_process(response)
        
        return response
    
    def _load_tools(self, skill_tags: List[str]) -> List[Dict[str, Any]]:
        """
        Load tools from AgentRegistry by skill tags.
        
        Args:
            skill_tags: Tags to filter tools by
            
        Returns:
            List of OpenAI-compatible tool schemas
        """
        from ..agent_tools import get_registry
        
        if not skill_tags:
            return []
        
        registry = get_registry()
        return registry.get_tools_for_tags(skill_tags)
    
    def _build_messages(
        self,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]],
        message: str,
    ) -> List[Dict[str, str]]:
        """
        Build messages list for AI service.
        
        Args:
            system_prompt: System prompt
            conversation_history: Previous messages
            message: Current user message
            
        Returns:
            List of message dicts
        """
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": message})
        
        return messages
    
    def list_available_adapters(self) -> List[str]:
        """
        List all registered adapter names.
        
        Returns:
            List of adapter names
        """
        return self.adapter_registry.list_adapters()
    
    def get_adapter_info(self, adapter_name: str) -> Dict[str, Any]:
        """
        Get info about a registered adapter.
        
        Args:
            adapter_name: Adapter name
            
        Returns:
            Dict with adapter info
        """
        return self.adapter_registry.get_adapter_info(adapter_name)
