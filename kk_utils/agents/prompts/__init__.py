"""
kk_utils.agents.prompts — Centralized prompt templates

Master prompt templates for all agent adapters.

Structure:
    prompts/
    └── master/
        ├── kengkoon.txt         # Digital twin of Keng Koon
        ├── koon.txt             # Digital twin of Koon
        ├── agent_master.txt     # Skill execution assistant
        ├── researcher.txt       # Research assistant
        ├── content_creator.txt  # Content creation assistant
        ├── demo_assistant.txt   # Demo assistant
        └── default.txt          # Fallback template (from adapter)

Usage:
    from kk_utils.agents.prompts import load_master_prompt

    prompt = load_master_prompt("kengkoon")
"""

from pathlib import Path
from typing import Optional

_PROMPTS_DIR = Path(__file__).parent
_MASTER_DIR = _PROMPTS_DIR / "master"


def load_master_prompt(template_name: str) -> str:
    """
    Load master prompt template by name.

    Args:
        template_name: Name of template (e.g., "kengkoon", "researcher")

    Returns:
        Prompt template string

    Raises:
        FileNotFoundError: If template not found
    """
    template_path = _MASTER_DIR / f"{template_name}.txt"

    if template_path.exists():
        return template_path.read_text(encoding='utf-8')

    raise FileNotFoundError(f"Master prompt template not found: {template_name}")


def list_master_prompts() -> list:
    """List all available master prompt templates."""
    if not _MASTER_DIR.exists():
        return []

    return [
        f.stem for f in _MASTER_DIR.glob("*.txt")
        if not f.name.startswith("_")
    ]
