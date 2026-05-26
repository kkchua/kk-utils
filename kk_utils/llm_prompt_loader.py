"""
kk_utils.llm_prompt_loader

Reusable loader for prompt text stored in PostgreSQL llm_prompts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import text

from kk_utils.database import get_db_context

logger = logging.getLogger(__name__)


def _load_prompt_from_db(
    namespace: str,
    adapter: str,
    name: str,
    db_session=None,
) -> Optional[str]:
    """Return prompt_text from llm_prompts or None if not found."""
    query = text(
        """
        SELECT prompt_text
        FROM llm_prompts
        WHERE namespace = :namespace
          AND adapter = :adapter
          AND name = :name
          AND is_enabled = true
        ORDER BY updated_at DESC, id DESC
        LIMIT 1
        """
    )

    def _run(session) -> Optional[str]:
        row = session.execute(
            query,
            {"namespace": namespace, "adapter": adapter, "name": name},
        ).mappings().first()
        if not row:
            return None
        prompt_text = row.get("prompt_text") or ""
        return prompt_text.strip() or None

    try:
        if db_session is not None:
            return _run(db_session)
        with get_db_context() as session:
            return _run(session)
    except Exception as exc:
        logger.warning(
            "Failed to load llm prompt %s/%s/%s from DB: %s",
            namespace,
            adapter,
            name,
            exc,
        )
        return None


def load_llm_prompt(
    namespace: str,
    adapter: str,
    name: str,
    *,
    db_session=None,
    fallback_path: str | Path | None = None,
    fallback_text: str | None = None,
) -> str:
    """
    Load prompt text from PostgreSQL llm_prompts only.
    """
    if fallback_path is not None or fallback_text is not None:
        raise ValueError(
            "Static prompt fallbacks have been removed; load_llm_prompt() "
            "accepts DB-backed prompts only"
        )

    prompt_text = _load_prompt_from_db(namespace, adapter, name, db_session=db_session)
    if prompt_text:
        logger.info("Loaded prompt %s/%s/%s from DB", namespace, adapter, name)
        return prompt_text

    raise FileNotFoundError(
        f"Unable to load enabled DB prompt {namespace}/{adapter}/{name}"
    )
