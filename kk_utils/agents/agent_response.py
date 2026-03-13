"""
kk_utils.agents.agent_response — Agent response dataclasses

Standardized response format for all agent adapters.
Used by MasterAgent and all adapter implementations.

Usage:
    from kk_utils.agents import AgentResponse
    
    response = AgentResponse(
        response_text="Hello! How can I help you?",
        agent_type="ai_assistant",
        persona_name="ai_assistant",
        collection="ai_assistant",
        tools_available=0,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class AgentResponse:
    """
    Standardized response from any agent adapter.
    
    Attributes:
        response_text: The AI's text response to the user
        agent_type: Type of agent that generated the response (e.g., "agent_me", "ai_assistant")
        persona_name: Name of the persona used
        collection: ChromaDB collection name for the persona
        tools_available: Number of tools available to the agent
        metadata: Additional metadata (execution time, model used, etc.)
        error: Error message if any (None if successful)
        success: Whether the agent call was successful
    """
    response_text: str
    agent_type: str
    persona_name: str
    collection: str
    tools_available: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    success: bool = True
    
    @property
    def has_error(self) -> bool:
        """Check if response has an error."""
        return self.error is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "response_text": self.response_text,
            "agent_type": self.agent_type,
            "persona_name": self.persona_name,
            "collection": self.collection,
            "tools_available": self.tools_available,
            "metadata": self.metadata,
            "error": self.error,
            "success": self.success,
        }
