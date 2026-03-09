"""
kk_utils/ai/ai_runner.py

Pipeline runner for AI adapters.

Orchestrates the three-step flow:
  1. build_prompt(**kwargs)      → system prompt
  2. user_message(**kwargs)      → user-turn content
  3. generate_structured(...)    → validated Pydantic instance via Agents SDK
  4. post_process(result, **kwargs) → final output returned to caller

Usage:

    from kk_utils.ai import AIRunner, get_ai_service
    from my_adapters.rag.adapter import RAGAdapter

    runner = AIRunner(get_ai_service())

    # async context:
    result = await runner.run(RAGAdapter(), context=chunks, question="What is X?")

    # sync context:
    result = runner.run_sync(RAGAdapter(), context=chunks, question="What is X?")

    # With call attribution:
    from kk_utils.ai import CallContext
    ctx = CallContext(agent_name="rag", feature_name="ask")
    result = await runner.run(RAGAdapter(), context=ctx, question="What is X?")

dry_run support:
    result = runner.dry_run(RAGAdapter(), context="...", question="...")
    # Prints full system prompt + Pydantic JSON schema without making an API call.
    # Returns {"dry_run": True, "system_prompt": ..., "user_message": ..., "schema": ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from pydantic import BaseModel

from kk_utils.ai.base_ai_adapter import BaseAIAdapter

logger = logging.getLogger(__name__)


class AIRunner:
    """
    Stateless pipeline runner. Holds a reference to the AIService and
    delegates prompt building / output type resolution to the adapter.

    One runner instance can be shared across adapters.
    """

    def __init__(self, ai_service: Any) -> None:
        """
        Args:
            ai_service: An AIService instance (from kk_utils.ai or a subclass).
        """
        self._ai = ai_service

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    async def run(
        self,
        adapter: BaseAIAdapter,
        call_context: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """
        Execute the full adapter pipeline and return post_process() output.

        Args:
            adapter      : Concrete BaseAIAdapter implementation.
            call_context : Optional CallContext for usage attribution. Use this
                           for attribution — do NOT name adapter kwargs "call_context".
            **kwargs     : Forwarded to build_prompt(), user_message(), post_process().
                           These become the template substitution variables
                           (e.g. context=..., question=...).

        Returns:
            Whatever adapter.post_process() returns (default: Pydantic instance).
        """
        system_prompt = adapter.build_prompt(**kwargs)
        user_msg      = adapter.user_message(**kwargs)
        output_type   = adapter.output_type_schema()

        logger.debug(
            f"[AIRunner] adapter={adapter.__class__.__name__} "
            f"output_type={output_type.__name__}"
        )

        result: BaseModel = await self._ai.generate_structured(
            prompt=user_msg,
            system_prompt=system_prompt,
            output_type=output_type,
            context=call_context,
        )

        return adapter.post_process(result, **kwargs)

    def run_sync(
        self,
        adapter: BaseAIAdapter,
        call_context: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """
        Synchronous wrapper around run(). Convenient for non-async callers.

        Do NOT call this from inside a running event loop (e.g. FastAPI handlers)
        — use `await runner.run(...)` instead.
        """
        return asyncio.run(self.run(adapter, call_context=call_context, **kwargs))

    # ------------------------------------------------------------------
    # Dry-run / debug
    # ------------------------------------------------------------------

    def dry_run(
        self,
        adapter: BaseAIAdapter,
        **kwargs,
    ) -> dict:
        """
        Build prompt + schema without making any API call.

        Prints the full system prompt and Pydantic JSON schema to stdout,
        matching the conductor ai_runner dry-run format for consistency.

        Returns a dict with keys: dry_run, system_prompt, user_message, schema.
        """
        system_prompt = adapter.build_prompt(**kwargs)
        user_msg      = adapter.user_message(**kwargs)
        output_type   = adapter.output_type_schema()
        schema_json   = json.dumps(output_type.model_json_schema(), indent=2)

        sep = "=" * 70
        print(f"\n{sep}")
        print(f"DRY RUN — adapter: {adapter.__class__.__name__}")
        print(sep)

        print(f"\n{sep}")
        print("SYSTEM PROMPT")
        print(sep)
        print(system_prompt)

        print(f"\n{sep}")
        print("USER MESSAGE")
        print(sep)
        print(user_msg)

        print(f"\n{sep}")
        print(f"OUTPUT SCHEMA (Pydantic JSON Schema for {output_type.__name__})")
        print(sep)
        print(schema_json)
        print()

        logger.info(f"[AIRunner] DRY RUN complete — no API call made")

        return {
            "dry_run":       True,
            "system_prompt": system_prompt,
            "user_message":  user_msg,
            "schema":        output_type.model_json_schema(),
        }
