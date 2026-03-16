"""
kk_utils.agents.skill_handlers.base_handler — Base class for all skill handlers

Every skill handler must inherit from this base class and implement:
- handle(): Execute the skill
- can_handle(): Check if this handler can process the skill
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent_response import AgentResponse


class SkillContext:
    """
    Context information passed to skill handlers.
    
    Contains all information needed to execute a skill:
    - User information (id, role)
    - Persona information
    - Conversation context
    - Attachments
    - RAG collection
    """
    
    def __init__(
        self,
        user_id: str,
        user_role: str = "demo",
        persona_name: Optional[str] = None,
        persona_collection: Optional[str] = None,
        conversation_history: Optional[list] = None,
        attachments: Optional[list] = None,
        model: str = "openai/gpt-5-nano",
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.user_id = user_id
        self.user_role = user_role
        self.persona_name = persona_name
        self.persona_collection = persona_collection
        self.conversation_history = conversation_history or []
        self.attachments = attachments or []
        self.model = model
        self.extra = extra or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get extra context value."""
        return self.extra.get(key, default)


class SkillResult:
    """
    Result from skill execution.
    
    Attributes:
        output: The skill output (dict, str, or any serializable type)
        success: Whether the skill executed successfully
        error: Error message if failed
        downloads: List of downloadable files (file paths or URLs)
        requires_polling: Whether client should poll for results
        metadata: Additional metadata about the execution
    """
    
    def __init__(
        self,
        output: Any,
        success: bool = True,
        error: Optional[str] = None,
        downloads: Optional[list] = None,
        requires_polling: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.output = output
        self.success = success
        self.error = error
        self.downloads = downloads or []
        self.requires_polling = requires_polling
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for response serialization."""
        return {
            "output": self.output,
            "success": self.success,
            "error": self.error,
            "downloads": self.downloads,
            "requires_polling": self.requires_polling,
            "metadata": self.metadata,
        }


class BaseSkillHandler(ABC):
    """
    Abstract base class for all skill handlers.
    
    Each handler is responsible for a specific skill execution pattern:
    - Standard: Direct tool call → return result
    - Prompt Selection: Choose prompt template → execute with template
    - Batch File: Process multiple files → aggregate results
    - ComfyUI: Submit to ComfyUI → return task_id (submit & forget)
    """
    
    handler_type: str = "base"  # Override in subclass
    
    @abstractmethod
    async def handle(
        self,
        skill_name: str,
        tool_call: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """
        Execute a skill and return result.
        
        Args:
            skill_name: Name of skill module
            tool_call: Tool call from AI {name, arguments}
            context: Execution context
        
        Returns:
            SkillResult with output
        """
        pass
    
    @abstractmethod
    def can_handle(self, skill_name: str, tool_call: Dict[str, Any]) -> bool:
        """
        Check if this handler can process the skill.
        
        Args:
            skill_name: Name of skill module
            tool_call: Tool call from AI
        
        Returns:
            True if this handler should process the skill
        """
        pass
    
    async def handle_batch(
        self,
        skill_name: str,
        tool_calls: list,
        context: SkillContext,
    ) -> SkillResult:
        """
        Handle batch execution (multiple tool calls).
        
        Default implementation: Process each tool call sequentially.
        Override in subclass for custom batch behavior.
        
        Args:
            skill_name: Name of skill module
            tool_calls: List of tool calls
            context: Execution context
        
        Returns:
            SkillResult with aggregated outputs
        """
        results = []
        for tool_call in tool_calls:
            result = await self.handle(skill_name, tool_call, context)
            results.append(result)
        
        # Aggregate results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        return SkillResult(
            output={
                "batch_results": [r.output for r in results],
                "total": len(tool_calls),
                "successful": len(successful),
                "failed": len(failed),
            },
            success=len(failed) == 0,
            downloads=[d for r in results for d in r.downloads],
            metadata={
                "batch": True,
                "total": len(tool_calls),
                "successful": len(successful),
                "failed": len(failed),
            },
        )
    
    def _get_skill_metadata(self, skill_name: str) -> Dict[str, Any]:
        """
        Get skill metadata from config.
        
        Args:
            skill_name: Name of skill module
        
        Returns:
            Skill metadata dict
        """
        # Try to load from skill's config.json
        import json
        from pathlib import Path
        
        # Look for config in kk-agent-skills
        skill_config_paths = [
            Path(__file__).parent.parent.parent.parent / "kk-agent-skills" / "kk_agent_skills" / skill_name / "config.json",
            Path(__file__).parent.parent.parent.parent / "kk_agent_skills" / skill_name / "config.json",
        ]
        
        for config_path in skill_config_paths:
            if config_path.exists():
                try:
                    return json.loads(config_path.read_text(encoding='utf-8'))
                except Exception:
                    pass
        
        # Return default metadata
        return {
            "name": skill_name,
            "handler_type": self.handler_type,
        }
