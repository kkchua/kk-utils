"""
kk_utils.persona_config — Persona configuration loader

Loads persona definitions from PostgreSQL.
Each persona is a digital twin of a real person, backed by its own isolated
ChromaDB collection.

Access to a persona is governed by the Governor's collection security levels:
  user_SL >= collection_SL  →  access granted

Usage:
    from kk_utils.persona_config import load_persona

    persona = load_persona("kengkoon", db_session=session)
    print(persona.display_name)   # "Keng Koon"
    print(persona.collection)     # "persona_kengkoon"
    print(persona.skills)         # ["digital_me", "notes", "web_search"]
    print(persona.system_prompt)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional  # Optional kept for return types

logger = logging.getLogger(__name__)


@dataclass
class PersonaConfig:
    """Configuration for a single digital-twin persona."""
    name: str
    display_name: str
    collection: str       # ChromaDB collection name (also used for SL check)
    skills: List[str]     # Persona-specific skills; adapter defaults are merged at runtime
    skill_tags: List[str] # Persona-specific tool tags; adapter defaults are merged at runtime
    system_prompt: str
    # NEW: Adapter configuration (for Master Agent architecture)
    adapter_type: Optional[str] = None  # e.g., "agent_me", "ai_assistant"
    adapter_prompt_template: Optional[str] = "default"  # prompts/{template}.txt
    adapter_schema: Optional[str] = None  # prompts/{schema}.json
    
    def __post_init__(self):
        """Auto-detect adapter_type if not specified."""
        if self.adapter_type is None:
            # Auto-detect: has skills -> agent_me, no skills -> ai_assistant
            if self.skills:
                self.adapter_type = "agent_me"
            else:
                self.adapter_type = "ai_assistant"


def _load_persona_from_db(persona_name: str, db_session) -> Optional[PersonaConfig]:
    """Load a persona from PostgreSQL if a session is available."""
    if db_session is None:
        return None
    try:
        from app.models.persona import Persona
        from app.services.prompt_service import get_prompt_service
        from kk_utils.skill_manifest import get_skill_manifest
        from app.services.skill_discovery_service import get_skill_discovery_service

        persona = (
            db_session.query(Persona)
            .filter(Persona.persona_name == persona_name)
            .first()
        )
        if persona is None:
            return None

        prompt = get_prompt_service().get(db_session, namespace="agent", adapter="", name=persona_name)
        if not prompt or not (prompt.prompt_text or "").strip():
            raise ValueError(
                f"Persona {persona_name!r} is missing an enabled llm_prompts row "
                f"(namespace='agent', adapter='', name={persona_name!r})"
            )
        system_prompt = prompt.prompt_text

        discovery = get_skill_discovery_service()
        derived_tags: list[str] = []
        seen: set[str] = set()
        for skill_name in persona.skills or []:
            manifest = get_skill_manifest(skill_name)
            if manifest and manifest.tags:
                source_tags = list(manifest.tags)
            else:
                skill_detail = discovery.get_skill_details(skill_name)
                source_tags = list(skill_detail.tags or []) if skill_detail else []
            if not source_tags:
                source_tags = [skill_name]
            for tag in source_tags:
                if tag and tag not in seen:
                    seen.add(tag)
                    derived_tags.append(tag)

        return PersonaConfig(
            name=persona.persona_name,
            display_name=persona.display_name,
            collection=persona.collection,
            skills=list(persona.skills or []),
            skill_tags=derived_tags,
            system_prompt=system_prompt.strip(),
            adapter_type=persona.adapter_type or "agent_me",
            adapter_prompt_template=persona.persona_name,
            adapter_schema=None,
        )
    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"Failed to load persona '{persona_name}' from DB: {e}")
        return None


def load_persona(
    persona_name: str,
    config_path: Optional[Path] = None,
    db_session=None,
    allow_yaml_fallback: bool = False,
) -> Optional[PersonaConfig]:
    """
    Load a persona by name.

    Personas are DB-backed. `config_path` and `allow_yaml_fallback` are kept
    only for API compatibility with older callers.
    
    Note:
    Adapter baseline skills/tags are merged later by MasterAgent. The stored
    persona skills/tags are treated as additions.

    Args:
        persona_name: Persona key (e.g. "kengkoon", "test")
        config_path: Deprecated legacy parameter. Ignored.
        db_session: Optional SQLAlchemy session for DB-backed personas.
        allow_yaml_fallback: Deprecated legacy parameter. Ignored.

    Returns:
        PersonaConfig or None if persona not found.
    """
    db_persona = _load_persona_from_db(persona_name, db_session)
    if db_persona is not None:
        return db_persona

    if config_path is not None or allow_yaml_fallback:
        logger.warning(
            "Persona '%s' was requested without a DB-backed persona session; YAML fallback has been removed",
            persona_name,
        )
    return None


def list_personas(config_path: Optional[Path] = None) -> List[PersonaConfig]:
    """
    Legacy helper retained for compatibility.

    Personas are stored in PostgreSQL, so this function no longer reads YAML.
    """
    if config_path is not None:
        logger.warning("list_personas(config_path=...) is deprecated; YAML persona loading has been removed")
    return []
