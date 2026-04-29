"""
kk_utils.agents.coder.coder_registry — Adapter registry singleton

Thread-safe registry for coder adapters with name-based lookup.
Supports dynamic registration of new coder adapter types.

Usage:
    from kk_utils.agents.coder import CoderRegistry, DescImageCoderAdapter

    registry = CoderRegistry.instance()
    registry.register("desc_image", DescImageCoderAdapter)

    adapter_class = registry.get_adapter("desc_image")
    adapter = adapter_class()
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class CoderRegistry:
    """
    Singleton registry for coder adapters.

    Features:
    - Register coder adapter classes by name
    - Lookup adapters by name
    - List all registered adapters
    - Thread-safe registration
    """

    _instance: Optional["CoderRegistry"] = None
    _lock: threading.RLock = threading.RLock()

    def __init__(self) -> None:
        self._adapters: Dict[str, Type] = {}

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> "CoderRegistry":
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
        adapter_class: Type,
        override: bool = False,
    ) -> None:
        """
        Register a coder adapter class.

        Args:
            name: Adapter name (e.g., "desc_image", "csv_generator")
            adapter_class: Adapter class (must have adapter_name attribute)
            override: If True, allow overriding existing registration

        Raises:
            ValueError: If name already registered and override=False
        """
        with self._lock:
            if name in self._adapters and not override:
                raise ValueError(
                    f"Coder adapter '{name}' already registered. "
                    f"Use override=True to replace it."
                )
            self._adapters[name] = adapter_class
            logger.info(f"Registered coder adapter: {name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister a coder adapter.

        Args:
            name: Adapter name to remove

        Returns:
            True if unregistered, False if not found
        """
        with self._lock:
            if name in self._adapters:
                del self._adapters[name]
                logger.info(f"Unregistered coder adapter: {name}")
                return True
            return False

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_adapter(self, name: str) -> Type:
        """
        Get coder adapter class by name.

        Args:
            name: Adapter name (e.g., "desc_image")

        Returns:
            Adapter class

        Raises:
            KeyError: If adapter not found
        """
        with self._lock:
            if name not in self._adapters:
                available = list(self._adapters.keys())
                raise KeyError(
                    f"Coder adapter '{name}' not found. Available: {available}"
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
        with self._lock:
            return name in self._adapters

    def list_adapters(self) -> List[str]:
        """
        List all registered coder adapter names.

        Returns:
            List of adapter names
        """
        with self._lock:
            return list(self._adapters.keys())

    def get_adapter_info(self, name: str) -> Dict:
        """
        Get info about a registered coder adapter.

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
    adapter_class: Type,
    override: bool = False,
) -> None:
    """Convenience: Register a coder adapter."""
    CoderRegistry.instance().register(name, adapter_class, override)


def get_adapter(name: str) -> Type:
    """Convenience: Get a coder adapter class."""
    return CoderRegistry.instance().get_adapter(name)


def list_adapters() -> List[str]:
    """Convenience: List all registered coder adapters."""
    return CoderRegistry.instance().list_adapters()
