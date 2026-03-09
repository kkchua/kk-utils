"""
kk_utils.skill_manifest — SkillManifest dataclass

A machine-readable descriptor for a kk_agent_skills skill module.
Each skill's skill.py declares a SKILL instance that the factory
and registry can introspect.

Usage (in a skill's skill.py):
    from kk_utils.skill_manifest import SkillManifest

    SKILL = SkillManifest(
        name="digital_me",
        display_name="Digital Me",
        description="RAG-powered personal profile queries",
        version="1.0.0",
        tags=["digital_me", "rag", "resume"],
        collection="digital_me",
        capabilities=["tool_provider"],
        min_access_level="demo",
    )

Usage (factory / registry):
    from kk_agent_skills.digital_me.skill import SKILL
    print(SKILL.name, SKILL.tags, SKILL.collection)

    # Or discover all skills:
    from kk_utils.skill_manifest import discover_skills
    skills = discover_skills()
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Known skill module names under kk_agent_skills
_KNOWN_SKILLS = [
    "digital_me",
    "digital_me_rag",
    "notes",
    "web_search",
    "ai_tools",
    "article_generation",
]


@dataclass
class SkillManifest:
    """Machine-readable descriptor for a skill module."""
    name: str
    display_name: str
    description: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    collection: Optional[str] = None    # default RAG collection, if any
    capabilities: List[str] = field(default_factory=list)  # e.g. ["tool_provider", "rag_engine"]
    min_access_level: str = "user"      # minimum role required: "anonymous", "demo", "user", "admin"

    def __repr__(self) -> str:
        return (
            f"SkillManifest(name={self.name!r}, tags={self.tags}, "
            f"collection={self.collection!r}, access={self.min_access_level!r})"
        )


def get_skill_manifest(skill_name: str) -> Optional[SkillManifest]:
    """
    Load the SkillManifest for a skill by name.

    Imports kk_agent_skills.<skill_name>.skill and returns its SKILL attribute.

    Args:
        skill_name: Skill module name (e.g. "digital_me", "notes")

    Returns:
        SkillManifest or None if not found / no skill.py.
    """
    try:
        mod = importlib.import_module(f"kk_agent_skills.{skill_name}.skill")
        manifest = getattr(mod, "SKILL", None)
        if not isinstance(manifest, SkillManifest):
            logger.warning(f"kk_agent_skills.{skill_name}.skill.SKILL is not a SkillManifest")
            return None
        return manifest
    except ImportError:
        logger.debug(f"No skill.py found for '{skill_name}'")
        return None
    except Exception as e:
        logger.error(f"Error loading skill manifest for '{skill_name}': {e}")
        return None


def discover_skills(skill_names: Optional[List[str]] = None) -> List[SkillManifest]:
    """
    Discover and return SkillManifests for all known skills (or a subset).

    Args:
        skill_names: List of skill names to check. Defaults to all known skills.

    Returns:
        List of SkillManifest for skills that have a skill.py.
    """
    names = skill_names or _KNOWN_SKILLS
    result = []
    for name in names:
        manifest = get_skill_manifest(name)
        if manifest:
            result.append(manifest)
    return result
