"""
kk_utils.agents.prompts — legacy compatibility shim

Runtime prompt loading is DB-only via llm_prompts. Static master prompt files
have been removed.
"""


def load_master_prompt(template_name: str) -> str:
    """
    Legacy helper retained for compatibility.
    """
    raise FileNotFoundError(
        "Static master prompt files have been removed. "
        f"Prompt {template_name!r} must exist in llm_prompts."
    )


def list_master_prompts() -> list:
    """Static master prompt files have been removed."""
    return []
