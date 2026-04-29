"""
kk_utils.agents — Master Agent architecture

Pluggable agent adapter pattern for reusable AI agents.

Architecture:
    MasterAgent (orchestrator)
        ↓
    BaseAgentAdapter (API-based interface)
        ↓
    Concrete adapters (AgentMeAdapter, AIAssistantAdapter, etc.)

Coder Architecture (direct CLI invocation):
    MasterAgent → CoderRegistry → BaseCoderAdapter
        ↓
    Concrete adapters (DescImageCoderAdapter, CsvGeneratorCoderAdapter, etc.)
        ↓
    coder CLI subprocess + meta.json sidecar

Usage:
    from kk_utils.agents import MasterAgent, AgentResponse

    agent = MasterAgent(personas_config_path="config/personas.yaml")
    response = await agent.chat(
        message="Hello",
        persona_name="ai_assistant",
        user_id="user123",
    )

    # Coder adapters (direct CLI):
    from kk_utils.agents.coder import DescImageCoderAdapter

    adapter = DescImageCoderAdapter()
    response = await adapter.execute_coder(
        prompt_text="Describe this image",
        context={"image_path": "images/photo.jpg"},
    )
"""

from .agent_response import AgentResponse
from .base_agent_adapter import BaseAgentAdapter
from .agent_registry import AgentRegistry
from .master_agent import MasterAgent

# Coder module (direct CLI invocation)
# from . import coder

__all__ = [
    # Agent API
    "AgentResponse",
    "BaseAgentAdapter",
    "AgentRegistry",
    "MasterAgent",

    # Coder module (access via kk_utils.agents.coder)
    # "coder",
]
