"""
kk_utils.factory — AgentMeFactory

Loads skill modules for a persona and returns tools + system prompt
ready for native tool calling.

Access control: the factory checks PersonalAssistantGovernor before loading —
a caller can interact with a persona only if their SL >= the persona's collection SL.

Usage:
    from kk_utils.factory import AgentMeFactory
    from kk_utils.persona_config import load_persona
    from pathlib import Path

    config_path = Path("config/personas.yaml")
    persona = load_persona("kengkoon", config_path=config_path)

    # With access check (raises PermissionError if denied):
    agent_cfg = AgentMeFactory.for_persona(persona, user_role="admin")

    # agent_cfg.tools        → list of OpenAI-compatible tool schemas
    # agent_cfg.system_prompt → str to inject as system message
    # agent_cfg.persona       → PersonaConfig

Phase 4 will call:
    response = await ai_service.chat_with_tools(
        message=message,
        tools=agent_cfg.tools,
        system_prompt=agent_cfg.system_prompt,
        conversation_history=history,
    )
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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

        Args:
            persona: PersonaConfig with collection name
            user_role: Caller's role string (e.g. "admin", "user", "demo")

        Raises:
            PermissionError: If access is denied.
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
            # Running outside the backend (e.g. tests, notebooks) — skip check
            logger.debug("Governor not available — skipping access check")

    @staticmethod
    def load_skills(skill_names: List[str]) -> List[str]:
        """
        Import skill tool modules from kk_agent_skills.

        Each import triggers _auto_register(), registering the skill's tools
        into the global AgentRegistry.

        Args:
            skill_names: Module names under kk_agent_skills (e.g. ["digital_me", "notes"])

        Returns:
            List of successfully loaded skill names.
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
        Get OpenAI-compatible tool schemas for the given tags from the global registry.

        Args:
            tags: Skill tags to include (e.g. ["digital_me", "notes"])

        Returns:
            Deduplicated list of OpenAI tool schema dicts.
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
        Check access, load skills, and return an AgentConfig for the given persona.

        Args:
            persona: PersonaConfig from load_persona()
            user_role: Caller's role (used for Governor access check)

        Returns:
            AgentConfig with tools list and system prompt.

        Raises:
            PermissionError: If user_role cannot access the persona's collection.
        """
        cls.check_access(persona, user_role)

        loaded = cls.load_skills(persona.skills)
        tools = cls.get_tools(persona.skill_tags)

        logger.info(
            f"AgentMeFactory: persona={persona.name!r} role={user_role!r} "
            f"skills_loaded={loaded} tools={len(tools)}"
        )

        # Append global system prompt suffix from Governor (if available)
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
        Convenience: resolve persona by name, check access, return AgentConfig.

        Args:
            persona_name: Key from personas.yaml (e.g. "kengkoon", "test")
            user_role: Caller's role string
            config_path: Path to personas.yaml (optional)

        Returns:
            AgentConfig ready for use.

        Raises:
            ValueError: If persona_name is not found.
            PermissionError: If user_role cannot access the persona.
        """
        from kk_utils.persona_config import load_persona
        persona = load_persona(persona_name, config_path=config_path)
        if persona is None:
            raise ValueError(f"Persona '{persona_name}' not found in personas.yaml")
        return cls.for_persona(persona, user_role=user_role)
