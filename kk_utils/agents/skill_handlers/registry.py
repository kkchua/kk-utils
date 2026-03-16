"""
kk_utils.agents.skill_handlers.registry — Skill handler registry

Registry for skill handlers. Handlers are selected based on:
- handler_type from skill metadata
- can_handle() check at runtime

Usage:
    from kk_utils.agents.skill_handlers import SkillHandlerRegistry
    
    registry = SkillHandlerRegistry.instance()
    handler = registry.get_handler("comfyui")
    result = await handler.handle(skill_name, tool_call, context)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Type, List

from .base_handler import BaseSkillHandler, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class SkillHandlerRegistry:
    """
    Registry for skill handlers.
    
    Singleton pattern - use instance() to get the registry.
    
    Handlers are registered by handler_type:
    - "standard": Direct tool execution
    - "prompt_selection": Prompt template selection
    - "batch_file": Batch file processing
    - "comfyui": ComfyUI submission
    """
    
    _instance: Optional["SkillHandlerRegistry"] = None
    
    def __init__(self):
        self._handlers: Dict[str, BaseSkillHandler] = {}
        self._handler_classes: Dict[str, Type[BaseSkillHandler]] = {}
    
    @classmethod
    def instance(cls) -> "SkillHandlerRegistry":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton - for testing only."""
        cls._instance = None
    
    def register(
        self,
        handler_type: str,
        handler: Optional[BaseSkillHandler] = None,
        handler_class: Optional[Type[BaseSkillHandler]] = None,
        override: bool = False,
    ) -> None:
        """
        Register a skill handler.
        
        Args:
            handler_type: Type identifier (e.g., "standard", "comfyui")
            handler: Handler instance (optional)
            handler_class: Handler class (optional, will instantiate)
            override: If True, override existing registration
        """
        if handler_type in self._handlers and not override:
            logger.warning(f"Handler '{handler_type}' already registered, skipping")
            return
        
        if handler is None and handler_class is None:
            raise ValueError("Must provide either handler or handler_class")
        
        if handler is None:
            # Instantiate from class
            handler = handler_class()
        
        self._handlers[handler_type] = handler
        self._handler_classes[handler_type] = handler.__class__
        logger.info(f"Registered skill handler: {handler_type} ({handler.__class__.__name__})")
    
    def get_handler(self, handler_type: str) -> Optional[BaseSkillHandler]:
        """
        Get handler by type.
        
        Args:
            handler_type: Type identifier
        
        Returns:
            Handler instance or None
        """
        handler = self._handlers.get(handler_type)
        
        if handler is None:
            logger.warning(f"Handler '{handler_type}' not found")
            return None
        
        return handler
    
    def list_handlers(self) -> List[str]:
        """List all registered handler types."""
        return list(self._handlers.keys())
    
    def get_handler_info(self, handler_type: str) -> Dict[str, Any]:
        """
        Get info about a handler.
        
        Args:
            handler_type: Type identifier
        
        Returns:
            Dict with handler info
        """
        handler = self._handlers.get(handler_type)
        
        if handler is None:
            return {"error": f"Handler '{handler_type}' not found"}
        
        return {
            "type": handler_type,
            "class": handler.__class__.__name__,
            "module": handler.__class__.__module__,
        }
    
    def clear(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._handler_classes.clear()


# Convenience functions
def get_registry() -> SkillHandlerRegistry:
    """Get the global skill handler registry."""
    return SkillHandlerRegistry.instance()


def register_handler(
    handler_type: str,
    handler: Optional[BaseSkillHandler] = None,
    handler_class: Optional[Type[BaseSkillHandler]] = None,
    override: bool = False,
) -> None:
    """Register a skill handler."""
    registry = get_registry()
    registry.register(handler_type, handler, handler_class, override)


def get_handler(handler_type: str) -> Optional[BaseSkillHandler]:
    """Get handler by type."""
    registry = get_registry()
    return registry.get_handler(handler_type)
