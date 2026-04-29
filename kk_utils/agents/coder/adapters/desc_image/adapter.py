"""
kk_utils.agents.coder.adapters.desc_image — Image Description Coder Adapter

Generates image descriptions using coder CLI.
Use for tasks that require visual understanding of images.

Skills:
- Analyzes image content
- Produces structured descriptions
- Outputs to file with meta.json sidecar

Usage:
    from kk_utils.agents.coder.adapters import DescImageCoderAdapter

    adapter = DescImageCoderAdapter()
    response = await adapter.execute_coder(
        prompt_text="Describe the main subject and mood of this image",
        context={
            "image_path": "images/photo.jpg",
            "persona_collection": "desc_image",
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


class DescImageCoderAdapter(BaseCoderAdapter):
    """
    DescImage: Image description generation via coder CLI.

    Uses the 'desc_image' coder alias from model_mapping.json.
    """

    adapter_name = "desc_image"
    coder_alias = "desc_image"

    FALLBACK_SYSTEM_PROMPT = (
        "You are an image analysis assistant. "
        "Examine the provided image and produce a clear, detailed description. "
        "Focus on subject, composition, lighting, mood, and notable details."
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
        Execute image description coder.

        Args:
            prompt_text: Description request / task
            context: Optional context with:
                - image_path: Path to the image file
                - persona_collection: RAG collection name
                - db_session: Optional DB session for prompt loading
                - cwd: Working directory

        Returns:
            CoderResponse with description output
        """
        ctx = context or {}
        image_path = ctx.get("image_path", "")
        cwd = ctx.get("cwd")
        if cwd:
            cwd = Path(cwd)

        # Build user text with image context
        user_text = prompt_text
        if image_path:
            user_text = f"Image: {image_path}\n\n{prompt_text}"

        # Determine output artifact path for sidecar derivation
        artifact_path = ctx.get("artifact_output_path", "images/desc.json")

        return await self._invoke(
            user_text=user_text,
            context=ctx,
            cwd=cwd,
            artifact_output_path=artifact_path,
        )
