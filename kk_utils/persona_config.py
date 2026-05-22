"""
kk_utils.persona_config — Persona configuration loader

Loads persona definitions from PostgreSQL when a DB session is available.
Each persona is a digital twin of a real person, backed by its own isolated
ChromaDB collection.

Access to a persona is governed by the Governor's collection security levels:
  user_SL >= collection_SL  →  access granted

Usage:
    from kk_utils.persona_config import load_persona, list_personas

    persona = load_persona("kengkoon", db_session=session)
    print(persona.display_name)   # "Keng Koon"
    print(persona.collection)     # "persona_kengkoon"
    print(persona.skills)         # ["digital_me", "notes", "web_search"]
    print(persona.system_prompt)

Note: DB-backed personas are the primary runtime source of truth. A config_path
may still be supplied for legacy standalone apps that have not moved to DB.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional  # Optional kept for return types

import yaml

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


def _load_yaml(config_path: Path) -> Dict:
    """Load and parse a legacy persona config file."""
    if not config_path.exists():
        logger.warning(f"Persona config not found: {config_path}")
        return {}
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.error(f"Failed to load persona config from {config_path}: {e}")
        return {}


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
        system_prompt = prompt.prompt_text if prompt and prompt.prompt_text else "You are a helpful AI assistant."

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
    except Exception as e:
        logger.warning(f"Failed to load persona '{persona_name}' from DB: {e}")
        return None


def load_persona(
    persona_name: str,
    config_path: Optional[Path] = None,
    db_session=None,
    allow_yaml_fallback: bool = True,
) -> Optional[PersonaConfig]:
    """
    Load a persona by name.

    Priority:
    1. PostgreSQL personas table when db_session is provided
    2. legacy config file fallback (only when allow_yaml_fallback=True)
    
    Note:
    Adapter baseline skills/tags are merged later by MasterAgent. The stored
    persona skills/tags are treated as additions.

    Args:
        persona_name: Persona key (e.g. "kengkoon", "test")
        config_path: Path to a legacy persona config file.
        db_session: Optional SQLAlchemy session for DB-backed personas.
        allow_yaml_fallback: If False, do not fall back to the legacy config.

    Returns:
        PersonaConfig or None if persona not found.
    """
    db_persona = _load_persona_from_db(persona_name, db_session)
    if db_persona is not None:
        return db_persona

    if not allow_yaml_fallback or config_path is None:
        if db_session is not None:
            logger.warning(
                "Persona '%s' not found in PostgreSQL personas; legacy config fallback disabled",
                persona_name,
            )
        return None

    path = config_path
    data = _load_yaml(path)
    raw = data.get("personas", {}).get(persona_name)
    if raw is None:
        logger.warning(f"Persona '{persona_name}' not found in {path}")
        return None

    return PersonaConfig(
        name=persona_name,
        display_name=raw.get("display_name", persona_name),
        collection=raw.get("collection", persona_name),
        skills=raw.get("skills", []),
        skill_tags=raw.get("skill_tags", []),
        system_prompt=raw.get("system_prompt", "You are a helpful AI assistant.").strip(),
        adapter_type=raw.get("adapter_type"),
        adapter_prompt_template=raw.get("adapter_prompt_template", "default"),
        adapter_schema=raw.get("adapter_schema"),
    )


def list_personas(config_path: Path) -> List[PersonaConfig]:
    """
    Return all personas defined in the legacy persona config file.

    Args:
        config_path: Path to the legacy persona config file.

    Returns:
        List of PersonaConfig (all defined personas, regardless of access level).
    """
    path = config_path
    data = _load_yaml(path)
    result = []
    for name, raw in data.get("personas", {}).items():
        result.append(PersonaConfig(
            name=name,
            display_name=raw.get("display_name", name),
            collection=raw.get("collection", name),
            skills=raw.get("skills", []),
            skill_tags=raw.get("skill_tags", []),
            system_prompt=raw.get("system_prompt", "You are a helpful AI assistant.").strip(),
            adapter_type=raw.get("adapter_type"),
            adapter_prompt_template=raw.get("adapter_prompt_template", "default"),
            adapter_schema=raw.get("adapter_schema"),
        ))
    return result
