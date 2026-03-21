"""
kk_utils.agents.skill_handlers — Skill handler infrastructure

Skill handlers manage different execution patterns:
- Standard: Direct tool call → return result
- Prompt Selection: Choose prompt template → execute
- Batch File: Process multiple files → aggregate
- ComfyUI: Submit to ComfyUI → return task_id (submit & forget)

Usage:
    from kk_utils.agents.skill_handlers import (
        SkillHandlerRegistry,
        StandardSkillHandler,
        ComfyUISkillHandler,
    )
    
    registry = SkillHandlerRegistry.instance()
    registry.register("standard", StandardSkillHandler())
    registry.register("comfyui", ComfyUISkillHandler())
    
    handler = registry.get_handler("comfyui")
    result = await handler.handle(skill_name, tool_call, context)
"""

from .base_handler import BaseSkillHandler, SkillContext, SkillResult
from .registry import (
    SkillHandlerRegistry,
    get_registry,
    register_handler,
    get_handler,
)
from .standard_handler import StandardSkillHandler
from .comfyui_handler import ComfyUISkillHandler

__all__ = [
    # Base classes
    "BaseSkillHandler",
    "SkillContext",
    "SkillResult",
    
    # Registry
    "SkillHandlerRegistry",
    "get_registry",
    "register_handler",
    "get_handler",
    
    # Built-in handlers
    "StandardSkillHandler",
    "ComfyUISkillHandler",
]
