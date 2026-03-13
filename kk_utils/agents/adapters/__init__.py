"""
kk_utils.agents.adapters — Agent adapter implementations

Available adapters:
- AgentMeAdapter: Digital twin with skills and tools
- AIAssistantAdapter: General conversational AI
"""

from .agent_me.adapter import AgentMeAdapter
from .ai_assistant.adapter import AIAssistantAdapter

__all__ = [
    "AgentMeAdapter",
    "AIAssistantAdapter",
]
