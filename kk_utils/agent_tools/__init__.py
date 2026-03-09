"""
kk_utils.agent_tools — Agent tool infrastructure

Provides the @agent_tool decorator and AgentRegistry for building
agentskills.io-compatible skill packages.

Usage:
    from kk_utils.agent_tools import agent_tool, get_registry, _auto_register

    @agent_tool(name="My Tool", tags=["mytag"])
    def my_tool(param: str) -> dict:
        return {"result": param, "success": True}

    _auto_register()   # call at end of tools.py
"""

from kk_utils.agent_tools.decorators import agent_tool
from kk_utils.agent_tools.registry import (
    AgentRegistry,
    get_registry,
    register_tool,
    get_tool,
    execute_tool,
    _auto_register,
)

__all__ = [
    "agent_tool",
    "AgentRegistry",
    "get_registry",
    "register_tool",
    "get_tool",
    "execute_tool",
    "_auto_register",
]
