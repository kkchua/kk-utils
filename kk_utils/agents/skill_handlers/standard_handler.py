"""
kk_utils.agents.skill_handlers.standard_handler — Standard skill execution

Handles simple skills: call tool → return result

Used for:
- digital_me tools (get_work_experience, get_skills, etc.)
- notes tools
- web_search tools
- Any skill that doesn't need special handling
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

from .base_handler import BaseSkillHandler, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class StandardSkillHandler(BaseSkillHandler):
    """
    Standard skill handler - direct tool execution.
    
    Flow:
    1. Load tool from registry
    2. Execute tool with arguments
    3. Return result
    """
    
    handler_type = "standard"
    
    async def handle(
        self,
        skill_name: str,
        tool_call: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """
        Execute standard skill (direct tool call).
        
        Args:
            skill_name: Name of skill module
            tool_call: Tool call from AI {name, arguments}
            context: Execution context
        
        Returns:
            SkillResult with tool output
        """
        try:
            # 1. Load tool from registry
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})
            
            if not tool_name:
                return SkillResult(
                    output={"error": "Tool name not provided"},
                    success=False,
                    error="Tool name not provided",
                )
            
            # Inject context into tool arguments
            tool_args["user_id"] = context.user_id
            
            # 2. Execute tool
            from ..agent_tools import get_registry
            
            registry = get_registry()
            tool_fn = registry.get_tool_function(tool_name)
            
            if tool_fn is None:
                return SkillResult(
                    output={"error": f"Tool '{tool_name}' not found"},
                    success=False,
                    error=f"Tool '{tool_name}' not found",
                )
            
            # Execute tool (sync or async)
            import inspect
            if inspect.iscoroutinefunction(tool_fn):
                result = await tool_fn(**tool_args)
            else:
                result = tool_fn(**tool_args)
            
            # 3. Return result
            return SkillResult(
                output=result,
                success=True,
                metadata={
                    "skill_name": skill_name,
                    "tool_name": tool_name,
                    "handler_type": self.handler_type,
                },
            )
            
        except Exception as e:
            logger.error(f"StandardSkillHandler failed: {e}", exc_info=True)
            return SkillResult(
                output={"error": str(e)},
                success=False,
                error=str(e),
                metadata={
                    "skill_name": skill_name,
                    "handler_type": self.handler_type,
                },
            )
    
    def can_handle(self, skill_name: str, tool_call: Dict[str, Any]) -> bool:
        """
        Check if this handler can process the skill.
        
        Standard handler is the default fallback for all skills.
        """
        skill_meta = self._get_skill_metadata(skill_name)
        handler_type = skill_meta.get("handler_type", "standard")
        
        # Handle if handler_type is "standard" or not specified
        return handler_type in ("standard", None)
