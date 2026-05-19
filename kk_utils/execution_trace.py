"""
kk_utils.execution_trace — request-scoped trace bridge

Provides a lightweight context for nested skill / RAG / helper calls to emit
human-readable trace events into the same callback used by AIService.

This keeps the UI trace stream in sync across:
- LLM tool-loop events
- nested skill helper calls
- RAG queries
- internal HTTP tool calls
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Callable, Optional, Tuple

TraceCallback = Optional[Callable[[str], None]]

_TRACE_CALLBACK: ContextVar[TraceCallback] = ContextVar("kk_utils_trace_callback", default=None)
_TRACE_PREFIX: ContextVar[str] = ContextVar("kk_utils_trace_prefix", default="Agent:")


def set_trace_context(
    trace_callback: TraceCallback = None,
    trace_prefix: Optional[str] = None,
) -> Tuple:
    """
    Bind the current request trace callback/prefix into contextvars.

    Returns tokens that must be passed back to reset_trace_context().
    """
    callback_token = _TRACE_CALLBACK.set(trace_callback)
    prefix_token = None
    if trace_prefix is not None:
        prefix_token = _TRACE_PREFIX.set(trace_prefix)
    return callback_token, prefix_token


def reset_trace_context(tokens: Tuple) -> None:
    """Reset the trace context created by set_trace_context()."""
    callback_token, prefix_token = tokens
    if prefix_token is not None:
        _TRACE_PREFIX.reset(prefix_token)
    if callback_token is not None:
        _TRACE_CALLBACK.reset(callback_token)


def get_trace_callback() -> TraceCallback:
    """Return the active trace callback, if any."""
    return _TRACE_CALLBACK.get()


def get_trace_prefix() -> str:
    """Return the active trace prefix, if any."""
    return _TRACE_PREFIX.get()


def emit_trace(message: str) -> None:
    """
    Emit a trace event into the active callback, if one exists.

    The current prefix is automatically added unless the message already starts
    with it.
    """
    callback = _TRACE_CALLBACK.get()
    if not callback:
        return

    prefix = _TRACE_PREFIX.get()
    formatted = message if message.startswith(prefix) else f"{prefix} {message}"
    try:
        callback(formatted)
    except Exception:
        # Trace must never break the request path.
        pass
