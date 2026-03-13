"""
kk_utils.agents.agent_registry — Adapter registry singleton

Thread-safe registry for agent adapters with name-based lookup.
Supports dynamic registration of new adapter types.

Usage:
    from kk_utils.agents import AgentRegistry, AgentMeAdapter
    
    registry = AgentRegistry.instance()
    registry.register("agent_me", AgentMeAdapter)
    
    adapter_class = registry.get_adapter("agent_me")
    adapter = adapter_class()
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Type

from .base_agent_adapter import BaseAgentAdapter

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Singleton registry for agent adapters.
    
    Features:
    - Register adapter classes by name
    - Lookup adapters by name
    - List all registered adapters
    - Thread-safe registration
    """
    
    _instance: Optional["AgentRegistry"] = None
    _lock: threading.RLock = threading.RLock()
    
    def __init__(self) -> None:
        self._adapters: Dict[str, Type[BaseAgentAdapter]] = {}
    
    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------
    
    @classmethod
    def instance(cls) -> "AgentRegistry":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton — for testing only."""
        cls._instance = None
    
    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    
    def register(
        self,
        name: str,
        adapter_class: Type[BaseAgentAdapter],
        override: bool = False,
    ) -> None:
        """
        Register an adapter class.
        
        Args:
            name: Adapter name (e.g., "agent_me", "ai_assistant")
            adapter_class: Adapter class that extends BaseAgentAdapter
            override: If True, allow overriding existing registration
            
        Raises:
            ValueError: If name already registered and override=False
            TypeError: If adapter_class doesn't extend BaseAgentAdapter
        """
        if not issubclass(adapter_class, BaseAgentAdapter):
            raise TypeError(f"Adapter class must extend BaseAgentAdapter")

        with self._lock:
            if name in self._adapters and not override:
                raise ValueError(
                    f"Adapter '{name}' already registered. "
                    f"Use override=True to replace it."
                )
            self._adapters[name] = adapter_class
            logger.info(f"Registered agent adapter: {name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister an adapter.

        Args:
            name: Adapter name to remove

        Returns:
            True if unregistered, False if not found
        """
        with self._lock:
            if name in self._adapters:
                del self._adapters[name]
                logger.info(f"Unregistered agent adapter: {name}")
                return True
            return False
    
    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    
    def get_adapter(self, name: str) -> Type[BaseAgentAdapter]:
        """
        Get adapter class by name.
        
        Args:
            name: Adapter name

        Returns:
            Adapter class

        Raises:
            KeyError: If adapter not found
        """
        with self._lock:
            if name not in self._adapters:
                available = list(self._adapters.keys())
                raise KeyError(
                    f"Adapter '{name}' not found. Available: {available}"
                )
            return self._adapters[name]
    
    def has_adapter(self, name: str) -> bool:
        """
        Check if adapter is registered.
        
        Args:
            name: Adapter name
            
        Returns:
            True if registered
        """
        return name in self._adapters
    
    def list_adapters(self) -> List[str]:
        """
        List all registered adapter names.
        
        Returns:
            List of adapter names
        """
        return list(self._adapters.keys())
    
    def get_adapter_info(self, name: str) -> Dict:
        """
        Get info about a registered adapter.
        
        Args:
            name: Adapter name
            
        Returns:
            Dict with adapter info
        """
        adapter_class = self.get_adapter(name)
        return {
            "name": name,
            "class_name": adapter_class.__name__,
            "module": adapter_class.__module__,
            "adapter_name": getattr(adapter_class, "adapter_name", name),
        }


# ------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------

def register_adapter(
    name: str,
    adapter_class: Type[BaseAgentAdapter],
    override: bool = False,
) -> None:
    """Convenience: Register an adapter."""
    AgentRegistry.instance().register(name, adapter_class, override)


def get_adapter(name: str) -> Type[BaseAgentAdapter]:
    """Convenience: Get an adapter class."""
    return AgentRegistry.instance().get_adapter(name)


def list_adapters() -> List[str]:
    """Convenience: List all registered adapters."""
    return AgentRegistry.instance().list_adapters()
