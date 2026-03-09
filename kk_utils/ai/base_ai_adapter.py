"""
kk_utils/ai/base_ai_adapter.py

Abstract base class for all AI adapters.

Two patterns are supported:

  1. HARDCODED SCHEMA (recommended for well-typed outputs)
     ─────────────────────────────────────────────────────
     Subclass BaseAIAdapter directly. Define your Pydantic output model in
     adapter.py and return it from output_type_schema(). Load the system
     prompt from a .txt file in build_prompt().

     Used when: output shape is fixed and well-known (RAG answer, summary, etc.)

  2. SCHEMA-DRIVEN (prompt + Pydantic model built from external files)
     ──────────────────────────────────────────────────────────────────
     Inherit SchemaAdapterMixin alongside BaseAIAdapter. Place in the adapter
     folder:
       master_prompt.txt      — system prompt with {{var}} placeholders
       {name}_schema.json     — JSON Schema that drives Pydantic model generation
                                and optional field validators

     The mixin provides: build_prompt(), output_type_schema() — the concrete
     adapter only needs to implement user_message().

     Used when: output shape is defined externally / shared across adapters.

Pipeline (AIRunner.run):
  build_prompt(**kwargs) → system_prompt
  user_message(**kwargs) → user_message
  output_type_schema()   → Pydantic class
  generate_structured()  → validated Pydantic instance
  post_process()         → final output (default: the instance as-is)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel


class BaseAIAdapter(ABC):
    """
    Abstract base for all AI adapters.

    Concrete adapters must implement:
      build_prompt(**kwargs)     -> str              # system prompt
      user_message(**kwargs)     -> str              # user-turn content
      output_type_schema()       -> Type[BaseModel]  # Pydantic output model

    Optionally override:
      post_process(data, **kwargs) -> Any            # transform validated output
    """

    @abstractmethod
    def build_prompt(self, **kwargs) -> str:
        """
        Return the complete system prompt string.

        kwargs are the same keyword arguments passed to AIRunner.run().
        Typical use: load a .txt file and substitute {{placeholders}}.
        """

    @abstractmethod
    def user_message(self, **kwargs) -> str:
        """
        Return the user-turn message string.

        This is what the model sees as the user's input. Keep it focused —
        system context (e.g. RAG chunks) belongs in build_prompt(), not here.
        """

    @abstractmethod
    def output_type_schema(self) -> Type[BaseModel]:
        """
        Return the Pydantic model class that defines the expected JSON output.

        The Agents SDK uses this as the structured output contract — the model
        is forced to produce JSON satisfying this schema. No manual JSON
        parsing needed.

        Hardcoded adapters: return a class defined in adapter.py.
        Schema-driven adapters: SchemaAdapterMixin implements this automatically
            from the _schema.json file.
        """

    def post_process(self, data: BaseModel, **kwargs) -> Any:
        """
        Optional hook called after the AI response is validated.

        Override to transform or augment the Pydantic output before it is
        returned to the caller.

        Default: return the validated instance unchanged.
        """
        return data
