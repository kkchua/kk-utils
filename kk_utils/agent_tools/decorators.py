"""
kk_utils.agent_tools.decorators — @agent_tool decorator

Marks functions as agent tools with metadata for discovery, schema
generation, and LLM function-calling compatibility.

Usage:
    from kk_utils.agent_tools import agent_tool

    @agent_tool(name="My Tool", description="...", tags=["tag"])
    def my_tool(param: str) -> dict:
        return {"result": ..., "success": True}
"""
from functools import wraps
from typing import Callable, List, Optional, Dict, Any
import logging
import inspect

logger = logging.getLogger(__name__)


def agent_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    input_modes: Optional[List[str]] = None,
    output_modes: Optional[List[str]] = None,
    access_level: str = "user",        # anonymous | user | owner | admin
    sensitivity: str = "low",          # low | medium | high | critical
    input_schema: Optional[Dict[str, Any]] = None,
    requires_confirmation: bool = False,
    is_destructive: bool = False,
):
    """
    Decorator that marks a function as an agent tool.

    Args:
        name: Display name (defaults to title-cased function name)
        description: Tool description (defaults to docstring first line)
        tags: Discovery tags for tool categorization
        input_modes: MIME types accepted (default: ["application/json"])
        output_modes: MIME types returned (default: ["application/json"])
        access_level: Required access level (anonymous, user, owner, admin)
        sensitivity: Data sensitivity (low, medium, high, critical)
        input_schema: JSON schema for input validation (auto-built if None)
        requires_confirmation: Whether tool requires user confirmation
        is_destructive: Whether tool performs a destructive action

    Example:
        @agent_tool(
            name="Create Note",
            description="Create a new note in a specified group",
            tags=["notes", "create"],
            access_level="user",
            sensitivity="low",
        )
        def create_note(title: str, content: str, group_id: int) -> dict:
            \"\"\"Create a new note.\"\"\"
            ...
    """
    _tags = tags or []
    _input_modes = input_modes or ["application/json"]
    _output_modes = output_modes or ["application/json"]

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        schema = input_schema or _build_schema_from_hints(fn)

        tool_id = fn.__name__
        tool_name = name or fn.__name__.replace("_", " ").title()
        tool_desc = description or _get_description(fn)

        wrapper.__agent_tool__ = {
            "id": tool_id,
            "name": tool_name,
            "description": tool_desc,
            "tags": _tags,
            "input_modes": _input_modes,
            "output_modes": _output_modes,
            "access_level": access_level,
            "sensitivity": sensitivity,
            "parameters": schema,
            "requires_confirmation": requires_confirmation,
            "is_destructive": is_destructive,
            "function": fn,
        }

        wrapper.__openai_schema__ = {
            "type": "function",
            "function": {
                "name": tool_id,
                "description": tool_desc,
                "parameters": schema,
            }
        }

        logger.debug(f"Decorated agent tool: {tool_id} tags={_tags}")
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_description(fn: Callable) -> str:
    doc = (fn.__doc__ or "").strip()
    if not doc:
        return fn.__name__.replace("_", " ").title()
    return doc.split("\n")[0].strip()


def _build_schema_from_hints(fn: Callable) -> Dict[str, Any]:
    """Build a JSON schema dict from a function's type hints."""
    import typing

    sig = inspect.signature(fn)
    hints: Dict[str, Any] = {}

    try:
        hints = typing.get_type_hints(fn)
        hints.pop("return", None)
    except Exception:
        hints = {k: v for k, v in getattr(fn, "__annotations__", {}).items() if k != "return"}

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        py_type = hints.get(param_name, str)

        # Unwrap Optional[X] / Union[X, None]
        origin = getattr(py_type, "__origin__", None)
        if origin is not None:
            args = [a for a in getattr(py_type, "__args__", []) if a is not type(None)]
            py_type = args[0] if args else str
            origin = getattr(py_type, "__origin__", None)

        # Unwrap generic aliases: list[str] → list
        if origin is not None:
            py_type = origin

        param_schema: Dict[str, Any] = {
            "type": _py_type_to_json(py_type),
            "description": _get_param_description(fn, param_name),
        }

        if param.default is not inspect.Parameter.empty:
            param_schema["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = param_schema

    return {"type": "object", "properties": properties, "required": required}


def _py_type_to_json(py_type) -> str:
    return {str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object"}.get(py_type, "string")


def _get_param_description(fn: Callable, param_name: str) -> str:
    doc = fn.__doc__ or ""
    if "Args:" not in doc:
        return ""
    args_section = doc.split("Args:")[1]
    for line in args_section.split("\n"):
        line = line.strip()
        if line.startswith(f"{param_name}:"):
            return line.split(":", 1)[1].strip()
        if line.startswith(f"{param_name} "):
            return line.split(" ", 1)[1].strip()
    return ""
