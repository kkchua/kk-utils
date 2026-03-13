"""
kk_utils.agents — Master Agent architecture

Pluggable agent adapter pattern for reusable AI agents.

Architecture:
    MasterAgent (orchestrator)
        ↓
    BaseAgentAdapter (interface)
        ↓
    Concrete adapters (AgentMeAdapter, AIAssistantAdapter, etc.)

Usage:
    from kk_utils.agents import MasterAgent, AgentResponse
    
    agent = MasterAgent(personas_config_path="config/personas.yaml")
    response = await agent.chat(
        message="Hello",
        persona_name="ai_assistant",
        user_id="user123",
    )
"""

from .agent_response import AgentResponse
from .base_agent_adapter import BaseAgentAdapter
from .agent_registry import AgentRegistry
from .master_agent import MasterAgent

__all__ = [
    "AgentResponse",
    "BaseAgentAdapter",
    "AgentRegistry",
    "MasterAgent",
]
