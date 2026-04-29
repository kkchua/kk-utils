"""
kk_utils.agents.coder.adapters.csv_generator — CSV Generation Coder Adapter

Generates CSV files from structured data using coder CLI.
Use for tasks that require transforming data into CSV format.

Skills:
- Data transformation to CSV
- Schema validation
- Output to file with meta.json sidecar

Usage:
    from kk_utils.agents.coder.adapters import CsvGeneratorCoderAdapter

    adapter = CsvGeneratorCoderAdapter()
    response = await adapter.execute_coder(
        prompt_text="Convert the following data to CSV format...",
        context={
            "data": "...",
            "persona_collection": "csv_generator",
        },
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ...base_coder_adapter import BaseCoderAdapter
from ...coder_response import CoderResponse

logger = logging.getLogger(__name__)


class CsvGeneratorCoderAdapter(BaseCoderAdapter):
    """
    CsvGenerator: CSV file generation via coder CLI.

    Uses the 'csv_generator' coder alias from model_mapping.json.
    """

    adapter_name = "csv_generator"
    coder_alias = "csv_generator"

    FALLBACK_SYSTEM_PROMPT = (
        "You are a data processing assistant. "
        "Transform the provided data into a properly formatted CSV file. "
        "Ensure proper quoting, escaping, and delimiter handling. "
        "Output the CSV content to a file and report the file path in your artifacts."
    )

    def build_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build system prompt — DB first, file fallback, then hardcoded.

        Args:
            context: Optional context with db_session

        Returns:
            System prompt string
        """
        ctx = context or {}
        db_session = ctx.get("db_session")

        # Try DB
        prompt = self.load_prompt_from_db("default", db_session)
        if prompt:
            return prompt

        # Try file
        prompt = self.load_prompt_from_file("default")
        if prompt:
            return prompt

        # Fallback
        return self.FALLBACK_SYSTEM_PROMPT

    async def execute_coder(
        self,
        prompt_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> CoderResponse:
        """
        Execute CSV generation coder.

        Args:
            prompt_text: CSV generation request / task
            context: Optional context with:
                - data: Source data to convert
                - persona_collection: RAG collection name
                - db_session: Optional DB session for prompt loading
                - cwd: Working directory

        Returns:
            CoderResponse with CSV output path
        """
        ctx = context or {}
        cwd = ctx.get("cwd")
        if cwd:
            cwd = Path(cwd)

        # Determine output artifact path for sidecar derivation
        artifact_path = ctx.get("artifact_output_path", "output/data.csv")

        return await self._invoke(
            user_text=prompt_text,
            context=ctx,
            cwd=cwd,
            artifact_output_path=artifact_path,
        )
