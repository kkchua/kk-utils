"""
kk_utils.persona_config — Persona configuration loader

Loads persona definitions from personas.yaml. Each persona is a digital twin
of a real person, backed by its own isolated ChromaDB collection.

Access to a persona is governed by the Governor's collection security levels:
  user_SL >= collection_SL  →  access granted

Usage:
    from kk_utils.persona_config import load_persona, list_personas
    from pathlib import Path

    config = Path("config/personas.yaml")   # each app provides its own path
    persona = load_persona("kengkoon", config_path=config)
    print(persona.display_name)   # "Keng Koon"
    print(persona.collection)     # "persona_kengkoon"
    print(persona.skills)         # ["digital_me", "notes", "web_search"]
    print(persona.system_prompt)

Note: config_path is required — kk-utils has no default persona config. Each
application (backend, gradio app, etc.) provides its own personas.yaml path.
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
    skills: List[str]     # kk_agent_skills module names to import
    skill_tags: List[str] # tool registry tags to include
    system_prompt: str


def _load_yaml(config_path: Path) -> Dict:
    """Load and parse personas.yaml."""
    if not config_path.exists():
        logger.warning(f"Persona config not found: {config_path}")
        return {}
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.error(f"Failed to load persona config from {config_path}: {e}")
        return {}


def load_persona(
    persona_name: str,
    config_path: Path,
) -> Optional[PersonaConfig]:
    """
    Load a persona by name from personas.yaml.

    Args:
        persona_name: Persona key (e.g. "kengkoon", "test")
        config_path: Path to personas.yaml (required — each app provides its own).

    Returns:
        PersonaConfig or None if persona not found.
    """
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
    )


def list_personas(config_path: Path) -> List[PersonaConfig]:
    """
    Return all personas defined in personas.yaml.

    Args:
        config_path: Path to personas.yaml (required — each app provides its own).

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
        ))
    return result
