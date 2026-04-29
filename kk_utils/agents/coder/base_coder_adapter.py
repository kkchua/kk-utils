"""
kk_utils.agents.coder.base_coder_adapter — Abstract base for all coder adapters

Defines the interface that all coder adapters must implement.
Coder adapters invoke coder CLIs (qwen, claude, codex) via subprocess
and parse results from the meta.json sidecar contract.

Responsibilities:
- Resolve coder config from model_mapping.json
- Build system prompt (DB-first, file fallback)
- Invoke coder CLI with prompt + schema
- Parse meta.json sidecar and return CoderResponse

Usage:
    class DescImageCoderAdapter(BaseCoderAdapter):
        adapter_name = "desc_image"
        coder_alias = "desc_image"

        def build_system_prompt(self, context: dict) -> str:
            return load_prompt_from_db("desc_image", "default")

        async def execute_coder(self, prompt_text, context) -> CoderResponse:
            return await self._invoke(prompt_text, context)
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .coder_response import CoderResponse
from .coder_invoker import (
    CoderInvocationError,
    InvocationResult,
    invoke_coder,
)
from .model_resolver import resolve_coder, get_api_key
from .sidecar import (
    MetaJsonMissingError,
    MetaJsonInvalidError,
    ArtifactMissingError,
    read_and_validate_meta_json,
    enrich_sidecar,
    validate_artifact_files,
    compute_prompt_checksum,
)

logger = logging.getLogger(__name__)


class BaseCoderAdapter(ABC):
    """
    Abstract base class for all coder adapters.

    All adapters must implement:
    - build_system_prompt(): Build system prompt for the coder
    - execute_coder(): Invoke coder CLI and return CoderResponse

    Class attributes:
    - adapter_name: Unique identifier (e.g., "desc_image")
    - coder_alias: Model mapping key (e.g., "desc_image" → full config)
    """

    adapter_name: str
    coder_alias: str

    def __init__(self):
        """Initialize the coder adapter."""
        pass

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def build_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build system prompt for the coder.

        Implementations should:
        1. Try DB: llm_prompts(namespace="coder", adapter=adapter_name, name="default")
        2. Fallback: read from prompts/ directory
        3. Fallback: minimal hardcoded string

        Args:
            context: Optional context dict for prompt variable substitution

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    async def execute_coder(
        self,
        prompt_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> CoderResponse:
        """
        Execute coder CLI with prompt and return structured response.

        Args:
            prompt_text: User message / task description
            context: Optional context (persona, attachments, etc.)

        Returns:
            CoderResponse with parsed output, artifacts, and usage
        """
        pass

    # ------------------------------------------------------------------
    # Shared invocation helper
    # ------------------------------------------------------------------

    async def _invoke(
        self,
        user_text: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        cwd: Optional[Path] = None,
        sidecar_path: Optional[Path] = None,
        artifact_output_path: Optional[str] = None,
    ) -> CoderResponse:
        """
        Shared invocation logic for all coder adapters.

        Handles:
        1. Resolve coder config from model_mapping.json
        2. Build system prompt
        3. Load schema
        4. Compute meta.json sidecar path
        5. Invoke coder CLI
        6. Read & validate meta.json
        7. Validate artifact files
        8. Enrich sidecar
        9. Return CoderResponse

        Args:
            user_text: User message / task description
            context: Optional context dict
            cwd: Working directory (default: current dir)
            sidecar_path: Override sidecar path (default: derived from artifact_output_path)
            artifact_output_path: Expected artifact output path for sidecar derivation

        Returns:
            CoderResponse with parsed output and artifacts
        """
        ctx = context or {}
        work_dir = cwd or Path.cwd()

        # 1. Resolve coder config
        coder_config = resolve_coder(self.coder_alias)
        coder_name = coder_config.get("coder", "unknown")
        logger.info(
            f"[{self.adapter_name}] coder={coder_name} "
            f"model={coder_config.get('model', 'N/A')}"
        )

        # 2. Build system prompt
        system_prompt = self.build_system_prompt(ctx)

        # Apply context variable substitution to system prompt
        input_values = ctx.get("input_values", {})
        if input_values:
            for key, value in input_values.items():
                system_prompt = system_prompt.replace(f"{{{key}}}", str(value))

        # 3. Combine system + user text
        full_prompt = f"{system_prompt}\n\n---\n\n{user_text}"

        # 4. Load schema
        schema_text = self._load_schema()

        # 5. Compute sidecar path
        if sidecar_path is None:
            if artifact_output_path:
                from .sidecar import resolve_meta_json_path
                sidecar_path = work_dir / resolve_meta_json_path(artifact_output_path)
            else:
                # Default: {adapter_name}.meta.json in cwd
                sidecar_path = work_dir / f"{self.adapter_name}.meta.json"

        # Ensure parent directory exists
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)

        # 6. Compute prompt checksum
        prompt_checksum = compute_prompt_checksum(full_prompt)

        # 7. Invoke coder
        now_iso = lambda: datetime.now(timezone.utc).isoformat()

        try:
            invocation_result = invoke_coder(
                coder_config=coder_config,
                step=self.adapter_name,
                prompt_text=full_prompt,
                schema_text=schema_text,
                cwd=work_dir,
                prompt_checksum=prompt_checksum,
                now_iso_fn=now_iso,
                sidecar_path=sidecar_path,
            )
        except CoderInvocationError as e:
            logger.error(f"[{self.adapter_name}] coder invocation failed: {e}")
            return CoderResponse(
                response_text=f"Coder invocation failed: {e}",
                agent_type=f"coder_{self.adapter_name}",
                persona_name=self.adapter_name,
                collection="",
                return_code=e.return_code,
                error=f"coder_invocation_failed: {e}",
                success=False,
            )

        # 8. Read & validate meta.json sidecar
        try:
            meta = read_and_validate_meta_json(sidecar_path)
        except (MetaJsonMissingError, MetaJsonInvalidError) as e:
            logger.error(f"[{self.adapter_name}] sidecar validation failed: {e}")
            return CoderResponse(
                response_text=f"Sidecar validation failed: {e}",
                agent_type=f"coder_{self.adapter_name}",
                persona_name=self.adapter_name,
                collection="",
                error=f"sidecar_validation_failed: {e}",
                success=False,
            )

        coder_result = meta["coder_result"]
        status = coder_result["status"]
        artifacts = coder_result.get("artifacts", {})

        # 9. Validate artifact files exist
        try:
            validate_artifact_files(artifacts, project_root=work_dir)
        except ArtifactMissingError as e:
            logger.warning(f"[{self.adapter_name}] artifact validation warning: {e}")

        # 10. Enrich sidecar with runner_data
        try:
            enrich_sidecar(
                meta_path=sidecar_path,
                step=self.adapter_name,
                coder_used=coder_name,
                invoked_at=invocation_result.usage.started_at,
                finished_at=invocation_result.usage.finished_at,
                prompt_checksum=prompt_checksum,
            )
        except Exception as e:
            logger.warning(f"[{self.adapter_name}] sidecar enrichment failed: {e}")

        # 11. Build response text from parsed result or artifacts
        response_text = self._build_response_text(invocation_result, coder_result)

        # 12. Return CoderResponse
        return CoderResponse(
            response_text=response_text,
            agent_type=f"coder_{self.adapter_name}",
            persona_name=self.adapter_name,
            collection=ctx.get("persona_collection", ""),
            meta_json_path=str(sidecar_path),
            return_code=invocation_result.return_code,
            usage=self._usage_to_dict(invocation_result.usage),
            artifacts=artifacts,
            reject_code=coder_result.get("reject_code"),
            raw_events=invocation_result.raw_events,
            success=status == "APPROVED",
            error=coder_result.get("remark") if status == "REJECTED" else None,
            metadata={
                "coder": coder_name,
                "model": coder_config.get("model"),
                "schema_version": meta.get("schema_version", "v2"),
                "prompt_checksum": prompt_checksum,
                "sidecar_path": str(sidecar_path),
            },
        )

    # ------------------------------------------------------------------
    # Schema loading
    # ------------------------------------------------------------------

    def _load_schema(self) -> str:
        """
        Load LLM response schema from schema/ directory.

        Returns:
            JSON schema as string
        """
        schema_file = Path(__file__).parent / "schema" / "llm_response_schema.json"
        if schema_file.exists():
            return schema_file.read_text(encoding="utf-8")
        # Minimal fallback schema
        return json.dumps({
            "type": "object",
            "required": ["status", "remark", "artifacts"],
            "properties": {
                "status": {"type": "string", "enum": ["APPROVED", "REJECTED"]},
                "remark": {"type": "string"},
                "artifacts": {"type": "object"},
            },
        })

    # ------------------------------------------------------------------
    # Response building helpers
    # ------------------------------------------------------------------

    def _build_response_text(
        self,
        invocation: InvocationResult,
        coder_result: Dict[str, Any],
    ) -> str:
        """
        Build human-readable response text from invocation + sidecar.

        Override in subclass for adapter-specific formatting.

        Args:
            invocation: Invocation result from coder CLI
            coder_result: coder_result dict from meta.json

        Returns:
            Response text string
        """
        # Try parsed result first
        parsed = invocation.parsed_result
        if parsed:
            remark = parsed.get("remark", "")
            if remark:
                return remark

        # Fallback to sidecar remark
        remark = coder_result.get("remark", "")
        if remark:
            return remark

        # Fallback to artifacts summary
        artifacts = coder_result.get("artifacts", {})
        if artifacts:
            artifact_list = ", ".join(f"{k}={v}" for k, v in artifacts.items())
            return f"Artifacts produced: {artifact_list}"

        return "Coder execution completed."

    def _usage_to_dict(self, usage) -> Dict[str, Any]:
        """Convert UsageData to dict."""
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cost": usage.cost,
            "duration_ms": usage.duration_ms,
            "usage_source": usage.usage_source,
        }

    # ------------------------------------------------------------------
    # Prompt loading helper
    # ------------------------------------------------------------------

    def load_prompt_from_db(
        self,
        template_name: str = "default",
        db_session=None,
    ) -> Optional[str]:
        """
        Load prompt template from DB (llm_prompts table).

        Args:
            template_name: Prompt template name
            db_session: Optional SQLAlchemy session

        Returns:
            Prompt text or None if not found
        """
        if db_session is None:
            return None
        try:
            from app.services.prompt_service import get_prompt_service
            llm_prompt = get_prompt_service().get(
                db_session, namespace="coder", adapter=self.adapter_name, name=template_name
            )
            if llm_prompt and llm_prompt.prompt_text:
                logger.info(
                    f"[{self.adapter_name}] Loaded prompt from DB "
                    f"'{template_name}' (id={llm_prompt.id}, len={len(llm_prompt.prompt_text)} chars)"
                )
                return llm_prompt.prompt_text
        except ImportError:
            pass  # Prompt service not available
        except Exception as e:
            logger.warning(f"[{self.adapter_name}] DB prompt loading failed: {e}")
        return None

    def load_prompt_from_file(self, template_name: str = "default") -> Optional[str]:
        """
        Load prompt template from adapter's prompts/ directory.

        Args:
            template_name: Prompt template name (without .txt)

        Returns:
            Prompt text or None if not found
        """
        # Try adapter-specific prompts directory
        adapter_dir = Path(__file__).parent / "adapters" / self.adapter_name / "prompts"
        prompt_file = adapter_dir / f"{template_name}.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")

        # Try shared coder prompts directory
        shared_prompts = Path(__file__).parent / "prompts"
        prompt_file = shared_prompts / f"{self.adapter_name}_{template_name}.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")

        return None
