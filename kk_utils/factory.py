"""
kk_utils.factory — Agent Factory

Provides:
1. AgentMeFactory: Original agent with skills/tools (backward compatible)
2. MasterAgentFactory: Pluggable agent architecture (new)
3. BaseAgent: Abstract base class for all agents (new)

Usage:
    # Original usage (still works):
    from kk_utils.factory import AgentMeFactory
    agent_cfg = AgentMeFactory.for_persona(persona, user_role="admin")
    
    # New pluggable usage:
    from kk_utils.factory import MasterAgentFactory
    agent = MasterAgentFactory.create(persona, user_role="admin")
    response = await agent.chat("Hello", user_id="user123")
"""
from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Original AgentMeFactory (backward compatible)
# =============================================================================

@dataclass
class AgentConfig:
    """Ready-to-use agent configuration for a persona."""
    tools: List[Dict]
    system_prompt: str
    persona: "PersonaConfig"  # type: ignore[name-defined]

    @property
    def tool_count(self) -> int:
        return len(self.tools)

    def __repr__(self) -> str:
        tool_names = [t.get("function", {}).get("name", "?") for t in self.tools]
        return (
            f"AgentConfig(persona={self.persona.name!r}, "
            f"tools={tool_names}, "
            f"collection={self.persona.collection!r})"
        )


class AgentMeFactory:
    """
    Factory that loads skill modules for a persona and returns an AgentConfig.

    Access is enforced by PersonalAssistantGovernor: the caller's SL must be
    >= the persona's collection SL, otherwise PermissionError is raised.
    """

    @staticmethod
    def check_access(persona: "PersonaConfig", user_role: str) -> None:  # type: ignore[name-defined]
        """
        Raise PermissionError if user_role cannot access the persona's collection.
        """
        try:
            from app.core.governor import PersonalAssistantGovernor
            governor = PersonalAssistantGovernor.instance()
            if not governor.check_collection_access(persona.collection, user_role):
                raise PermissionError(
                    f"Role '{user_role}' cannot access persona '{persona.name}' "
                    f"(collection '{persona.collection}')"
                )
        except ImportError:
            # Running outside the backend — skip check
            logger.debug("Governor not available — skipping access check")

    @staticmethod
    def load_skills(skill_names: List[str]) -> List[str]:
        """
        Import skill tool modules from kk_agent_skills.
        """
        loaded = []
        for skill in skill_names:
            module_path = f"kk_agent_skills.{skill}.tools"
            try:
                importlib.import_module(module_path)
                logger.debug(f"Loaded skill: {skill}")
                loaded.append(skill)
            except ImportError as e:
                logger.warning(f"Skill '{skill}' not available ({module_path}): {e}")
            except Exception as e:
                logger.error(f"Error loading skill '{skill}': {e}", exc_info=True)
        return loaded

    @staticmethod
    def get_tools(tags: List[str]) -> List[Dict]:
        """
        Get OpenAI-compatible tool schemas for the given tags.
        """
        from kk_utils.agent_tools import get_registry
        return get_registry().get_tools_for_tags(tags)

    @classmethod
    def for_persona(
        cls,
        persona: "PersonaConfig",  # type: ignore[name-defined]
        user_role: str = "admin",
    ) -> AgentConfig:
        """
        Check access, load skills, and return an AgentConfig.
        """
        cls.check_access(persona, user_role)

        loaded = cls.load_skills(persona.skills)
        tools = cls.get_tools(persona.skill_tags)

        logger.info(
            f"AgentMeFactory: persona={persona.name!r} role={user_role!r} "
            f"skills_loaded={loaded} tools={len(tools)}"
        )

        # Append global system prompt suffix from Governor
        system_prompt = persona.system_prompt
        try:
            from app.core.governor import PersonalAssistantGovernor
            suffix = PersonalAssistantGovernor.instance().get_global_system_prompt_suffix()
            if suffix:
                system_prompt = system_prompt.rstrip() + "\n" + suffix
        except ImportError:
            pass

        return AgentConfig(
            tools=tools,
            system_prompt=system_prompt,
            persona=persona,
        )

    @classmethod
    def for_persona_name(
        cls,
        persona_name: str,
        user_role: str,
        config_path: Optional[Path] = None,
    ) -> AgentConfig:
        """
        Convenience: resolve persona by name and return AgentConfig.
        """
        from kk_utils.persona_config import load_persona
        persona = load_persona(persona_name, config_path=config_path)
        if persona is None:
            raise ValueError(f"Persona '{persona_name}' not found in personas.yaml")
        return cls.for_persona(persona, user_role=user_role)


# =============================================================================
# New Pluggable Agent Architecture
# =============================================================================

@dataclass
class AgentResponse:
    """Standardized response from any agent."""
    response_text: str
    agent_type: str
    persona_name: str
    collection: str
    tools_available: int = 0
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """
    Abstract base class for all agent types.
    
    All agents must implement this interface for consistency.
    """

    def __init__(self, persona: "PersonaConfig", user_role: str):  # type: ignore[name-defined]
        self.persona = persona
        self.user_role = user_role
        self.agent_type = "base"

    @abstractmethod
    async def chat(
        self,
        message: str,
        user_id: str,
        conversation_history: Optional[List[Dict]] = None,
        model: str = "openai/gpt-5-nano",
        session_id: Optional[str] = None,
    ) -> AgentResponse:
        """Process a chat message."""
        pass

    @abstractmethod
    def get_tools(self) -> List[Dict]:
        """Get list of available tools."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass


class AgentMeAgent(BaseAgent):
    """
    AgentMe-based agent with skills/tools.
    
    This is the original agent behavior - loads skills and uses tool calling.
    """

    def __init__(self, persona: "PersonaConfig", user_role: str):  # type: ignore[name-defined]
        super().__init__(persona, user_role)
        self.agent_type = "agent_me"
        
        # Load agent config using existing AgentMeFactory
        self.agent_config = AgentMeFactory.for_persona(persona, user_role)
        
        logger.info(
            f"AgentMeAgent created: persona={persona.name!r} "
            f"tools={self.agent_config.tool_count}"
        )

    async def chat(
        self,
        message: str,
        user_id: str,
        conversation_history: Optional[List[Dict]] = None,
        model: str = "openai/gpt-5-nano",
        session_id: Optional[str] = None,
    ) -> AgentResponse:
        """Chat using AgentMe with tool calling."""
        from kk_utils.ai.ai_service import AIService
        
        start_time = datetime.now()
        
        try:
            ai_service = AIService(model=model)
            
            # Build messages with system prompt
            messages = [
                {"role": "system", "content": self.get_system_prompt()}
            ]
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Call AI with tools
            tools = self.get_tools()
            response = await ai_service.chat_with_tools(
                messages=messages,
                tools=tools,
            )
            
            response_text = response.get("content", "") if isinstance(response, dict) else str(response)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                response_text=response_text,
                agent_type="agent_me",
                persona_name=self.persona.name,
                collection=self.persona.collection,
                tools_available=len(tools),
                metadata={
                    "user_id": user_id,
                    "user_role": self.user_role,
                    "elapsed_seconds": elapsed,
                },
            )
            
        except Exception as e:
            logger.error(f"AgentMeAgent chat failed: {e}", exc_info=True)
            return AgentResponse(
                response_text="I encountered an error. Please try again.",
                agent_type="agent_me",
                persona_name=self.persona.name,
                collection=self.persona.collection,
                tools_available=len(self.get_tools()),
                error=f"chat_failed: {str(e)}",
            )

    def get_tools(self) -> List[Dict]:
        """Get OpenAI-compatible tool schemas."""
        return self.agent_config.tools

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return self.agent_config.system_prompt


class AIAssistantAgent(BaseAgent):
    """
    Simple AI assistant without skills/tools.
    
    Used for generic conversational AI interactions.
    """

    def __init__(self, persona: "PersonaConfig", user_role: str):  # type: ignore[name-defined]
        super().__init__(persona, user_role)
        self.agent_type = "ai_assistant"
        
        logger.info(f"AIAssistantAgent created: persona={persona.name!r}")

    async def chat(
        self,
        message: str,
        user_id: str,
        conversation_history: Optional[List[Dict]] = None,
        model: str = "openai/gpt-5-nano",
        session_id: Optional[str] = None,
    ) -> AgentResponse:
        """Chat using simple AI without tools."""
        from kk_utils.ai.ai_service import AIService
        
        start_time = datetime.now()
        
        try:
            ai_service = AIService(model=model)
            
            # Build messages with system prompt
            messages = [
                {"role": "system", "content": self.get_system_prompt()}
            ]
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Call AI without tools (pure conversation)
            response = await ai_service.chat(messages=messages)
            
            response_text = response.get("content", "") if isinstance(response, dict) else str(response)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                response_text=response_text,
                agent_type="ai_assistant",
                persona_name=self.persona.name,
                collection=self.persona.collection,
                tools_available=0,
                metadata={
                    "user_id": user_id,
                    "user_role": self.user_role,
                    "elapsed_seconds": elapsed,
                },
            )
            
        except Exception as e:
            logger.error(f"AIAssistantAgent chat failed: {e}", exc_info=True)
            return AgentResponse(
                response_text="I encountered an error. Please try again.",
                agent_type="ai_assistant",
                persona_name=self.persona.name,
                collection=self.persona.collection,
                tools_available=0,
                error=f"chat_failed: {str(e)}",
            )

    def get_tools(self) -> List[Dict]:
        """AI Assistant has no tools."""
        return []

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return self.persona.system_prompt


class MasterAgentFactory:
    """
    Factory that creates appropriate agent instances based on persona config.
    
    Supports pluggable agent types:
    - agent_me: Agent with skills/tools (default for personas with skills)
    - ai_assistant: Simple conversational AI (default for personas without skills)
    - custom: Future custom implementations
    
    Usage:
        agent = MasterAgentFactory.create(persona, user_role="admin")
        response = await agent.chat("Hello", user_id="user123")
    """
    
    # Registry of agent types
    _agent_types: Dict[str, type] = {
        "agent_me": AgentMeAgent,
        "ai_assistant": AIAssistantAgent,
    }
    
    @classmethod
    def register_agent_type(cls, name: str, agent_class: type) -> None:
        """
        Register a new agent type.
        
        Args:
            name: Agent type name (e.g., "custom_agent")
            agent_class: Agent class that extends BaseAgent
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"Agent class must extend BaseAgent")
        cls._agent_types[name] = agent_class
        logger.info(f"Registered agent type: {name}")
    
    @classmethod
    def create(
        cls,
        persona: "PersonaConfig",  # type: ignore[name-defined]
        user_role: str = "admin",
    ) -> BaseAgent:
        """
        Create an agent instance based on persona configuration.
        
        Args:
            persona: PersonaConfig from load_persona()
            user_role: User role for access control
        
        Returns:
            BaseAgent instance (AgentMeAgent, AIAssistantAgent, etc.)
        
        Raises:
            ValueError: If agent_type in persona config is not recognized
            PermissionError: If access check fails
        """
        # Determine agent type from persona config
        agent_type = getattr(persona, 'agent_type', None)
        
        # Default logic: no skills -> ai_assistant, otherwise agent_me
        if agent_type is None:
            skills = getattr(persona, 'skills', [])
            if not skills:
                agent_type = "ai_assistant"
            else:
                agent_type = "agent_me"
        
        logger.info(f"Creating agent: type={agent_type!r} persona={persona.name!r}")
        
        # Get agent class from registry
        agent_class = cls._agent_types.get(agent_type)
        if agent_class is None:
            raise ValueError(
                f"Unknown agent type '{agent_type}'. "
                f"Available types: {list(cls._agent_types.keys())}"
            )
        
        # Create and return agent instance
        return agent_class(persona, user_role)
    
    @classmethod
    def create_from_name(
        cls,
        persona_name: str,
        user_role: str,
        config_path: Optional[Path] = None,
    ) -> BaseAgent:
        """
        Convenience: resolve persona by name and create agent.
        
        Args:
            persona_name: Key from personas.yaml
            user_role: User role string
            config_path: Path to personas.yaml
        
        Returns:
            BaseAgent instance
        """
        from kk_utils.persona_config import load_persona
        
        persona = load_persona(persona_name, config_path=config_path)
        if persona is None:
            raise ValueError(f"Persona '{persona_name}' not found")
        
        return cls.create(persona, user_role)
