"""
kk_utils.agent_tools.registry — AgentRegistry singleton

Thread-safe registry for all agent tools with tag-based discovery,
OpenAI-compatible schema export, and access-level enforcement.

Usage:
    from kk_utils.agent_tools import get_registry, agent_tool

    registry = get_registry()
    registry.register(my_tool_fn)        # fn decorated with @agent_tool
    tools = registry.get_tools_by_tag("notes")
    result = registry.execute("create_note", title="...", content="...")
"""
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Singleton registry for agent tools.

    Features:
    - Register tools decorated with @agent_tool
    - Discover tools by tag or access level
    - Export OpenAI-compatible schemas for LLM function calling
    - Execute tools with error handling
    - Access-level enforcement
    """

    _instance: Optional["AgentRegistry"] = None
    _lock: threading.RLock = threading.RLock()

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._agents: Dict[str, Any] = {}   # for A2A / agent references

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

    def register(self, fn: Callable, prefix: str = "") -> None:
        """
        Register a function decorated with @agent_tool.

        Args:
            fn: Function with __agent_tool__ attribute
            prefix: Optional skill namespace prefix (e.g. "digital_me").
                    When provided, the tool is registered under both
                    "{prefix}__{id}" (used by the LLM) and the original
                    "{id}" (for backward-compatible execute() calls).
        """
        if not hasattr(fn, "__agent_tool__"):
            logger.warning(f"Function {fn.__name__} is not decorated with @agent_tool — skipped")
            return

        tool_info = fn.__agent_tool__
        original_id = tool_info["id"]

        if prefix:
            prefixed_id = f"{prefix}__{original_id}"
            prefixed_schema = {
                **fn.__openai_schema__,
                "function": {
                    **fn.__openai_schema__["function"],
                    "name": prefixed_id,
                },
            }
            prefixed_info = {**tool_info, "id": prefixed_id}

            # Primary entry: prefixed name shown to the LLM
            self._tools[prefixed_id] = {
                "function": fn,
                "info": prefixed_info,
                "openai_schema": prefixed_schema,
            }
            # Backward-compat alias: original name still works for execute()
            # Marked as _alias=True so tag-based schema lookups skip it (avoid duplicates)
            self._tools[original_id] = {
                "function": fn,
                "info": tool_info,
                "openai_schema": fn.__openai_schema__,
                "_alias": True,
            }
            logger.info(
                f"Registered agent tool: {prefixed_id} (alias: {original_id}) "
                f"tags={tool_info.get('tags', [])}"
            )
        else:
            self._tools[original_id] = {
                "function": fn,
                "info": tool_info,
                "openai_schema": fn.__openai_schema__,
            }
            logger.info(f"Registered agent tool: {original_id} tags={tool_info.get('tags', [])}")

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_tool(self, tool_id: str) -> Optional[Callable]:
        """Get tool function by ID."""
        entry = self._tools.get(tool_id)
        return entry["function"] if entry else None

    def get_tool_info(self, tool_id: str) -> Optional[Dict]:
        """Get tool metadata dict."""
        entry = self._tools.get(tool_id)
        return entry["info"] if entry else None

    def get_all_tools(self) -> List[Dict]:
        """Return all tools as OpenAI-compatible schemas."""
        return [e["openai_schema"] for e in self._tools.values()]

    def get_tools_by_tag(self, tag: str) -> List[Dict]:
        """Return tools whose tags include *tag*.
        Alias entries (backward-compat originals for prefixed tools) are excluded
        so the LLM sees only the prefixed name, not both names.
        """
        return [
            e["openai_schema"]
            for e in self._tools.values()
            if tag in e["info"].get("tags", []) and not e.get("_alias", False)
        ]

    def get_tools_for_tags(self, tags: List[str]) -> List[Dict]:
        """Return tools matching ANY of the provided tags (deduped)."""
        seen: set = set()
        result: List[Dict] = []
        for tag in tags:
            for schema in self.get_tools_by_tag(tag):
                tool_id = schema["function"]["name"]
                if tool_id not in seen:
                    result.append(schema)
                    seen.add(tool_id)
        return result

    def get_tools_by_access_level(self, access_level: str) -> List[Dict]:
        """Return tools accessible at *access_level* or below."""
        return [
            e["openai_schema"]
            for e in self._tools.values()
            if self._has_access(access_level, e["info"].get("access_level", "user"))
        ]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, tool_id: str, **kwargs) -> Any:
        """
        Execute a tool by ID.

        Returns:
            Tool result dict, or {"error": "..."} on failure.
        """
        entry = self._tools.get(tool_id)
        if not entry:
            return {"error": f"Tool '{tool_id}' not found"}

        try:
            result = entry["function"](**kwargs)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            logger.error(f"Tool execution failed [{tool_id}]: {e}", exc_info=True)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_tools(self) -> List[Tuple[str, Dict]]:
        """Return list of (tool_id, tool_info) tuples."""
        return [(tid, e["info"]) for tid, e in self._tools.items()]

    def is_registered(self, tool_id: str) -> bool:
        return tool_id in self._tools

    def registered_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"AgentRegistry({len(self._tools)} tools: {self.registered_tool_names()})"

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _has_access(self, user_level: str, required_level: str) -> bool:
        hierarchy = {"anonymous": 0, "user": 1, "owner": 2, "admin": 3}
        return hierarchy.get(user_level, 0) >= hierarchy.get(required_level, 0)

    def _get_user_level(self, user_id: str) -> str:
        if user_id and user_id != "anonymous":
            return "user"
        return "anonymous"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_registry() -> AgentRegistry:
    """Return the global AgentRegistry singleton."""
    return AgentRegistry.instance()


def register_tool(fn: Callable) -> None:
    """Register a tool function in the global registry."""
    get_registry().register(fn)


def get_tool(tool_id: str) -> Optional[Callable]:
    """Get a tool function by ID from the global registry."""
    return get_registry().get_tool(tool_id)


def execute_tool(tool_id: str, **kwargs) -> Any:
    """Execute a tool by ID using the global registry."""
    return get_registry().execute(tool_id, **kwargs)


def _auto_register(module=None) -> None:
    """
    Scan *module* (or the caller's module) and register every function
    decorated with @agent_tool.

    Call at the bottom of any tools.py:
        _auto_register()

    Skill prefix is auto-detected from the calling module path.
    For a module named ``kk_agent_skills.digital_me.tools``, the prefix
    ``digital_me`` is applied so the LLM sees ``digital_me__get_education``
    instead of ``get_education``.  Modules outside the ``kk_agent_skills``
    namespace are registered without a prefix.
    """
    import sys
    import inspect

    if module is None:
        # Walk up one frame to get the calling module
        frame = sys._getframe(1)
        module_name = frame.f_globals.get("__name__", "")
        module = sys.modules.get(module_name)
    else:
        module_name = getattr(module, "__name__", "")

    if module is None:
        logger.warning("_auto_register: could not determine calling module")
        return

    # Derive skill prefix from module path.
    # Pattern: kk_agent_skills.<skill_name>.tools → prefix = <skill_name>
    parts = module_name.split(".")
    skill_prefix = (
        parts[-2]
        if len(parts) >= 3 and parts[0] == "kk_agent_skills" and parts[-1] == "tools"
        else ""
    )

    registry = get_registry()
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if callable(obj) and hasattr(obj, "__agent_tool__"):
            registry.register(obj, prefix=skill_prefix)
