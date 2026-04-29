"""
kk_utils.agents.coder.coder_invoker — Coder subprocess invocation

Builds CLI command from cli_params config, launches coder via subprocess,
polls for sidecar (meta.json) as early-exit signal, and returns structured
InvocationResult.

All CLI parameters are data-driven via model_mapping.json — no hardcoding
of --output-format, --approval-mode, etc.

Usage:
    from kk_utils.agents.coder import invoke_coder, build_command

    config = resolve_coder("desc_image")
    result = await invoke_coder(
        coder_config=config,
        prompt_text="Describe this image...",
        schema_text='{"type": "object", ...}',
        sidecar_path=Path("images/desc.meta.json"),
        cwd=Path("/project"),
    )
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CODER_TIMEOUT_SECONDS = 600
SIDECAR_POLL_INTERVAL_SECONDS = 3.0
SIDECAR_SETTLE_DELAY_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UsageData:
    """Token usage and timing from a coder invocation."""
    step: str = ""
    coder_used: str = ""
    usage_source: str = "not_available"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    duration_ms: Optional[int] = None
    started_at: str = ""
    finished_at: str = ""


@dataclass
class InvocationManifest:
    """Record of what was invoked."""
    step_name: str = ""
    coder_used: str = ""
    command: List[str] = field(default_factory=list)
    cwd: str = ""
    prompt_checksum: str = ""
    started_at: str = ""
    finished_at: str = ""
    return_code: int = 0


@dataclass
class InvocationResult:
    """Complete result from a coder invocation."""
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    parsed_result: Dict[str, Any] = field(default_factory=dict)
    usage: UsageData = field(default_factory=UsageData)
    manifest: InvocationManifest = field(default_factory=InvocationManifest)
    raw_events: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class CoderInvocationError(Exception):
    """Coder subprocess failed or timed out."""
    def __init__(
        self,
        message: str,
        command: Optional[List[str]] = None,
        return_code: int = 1,
        stdout: str = "",
        stderr: str = "",
        raw_events: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.command = command or []
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.raw_events = raw_events or []


# ---------------------------------------------------------------------------
# Timeout config
# ---------------------------------------------------------------------------

def _coder_timeout_seconds() -> int:
    """Get coder timeout from env, fallback to default."""
    raw = os.environ.get("CODER_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_CODER_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CODER_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_CODER_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Command builder — data-driven from cli_params
# ---------------------------------------------------------------------------

def build_command(
    *,
    coder_config: Dict[str, Any],
    prompt_text: str,
    schema_text: str,
    session_id: str,
    schema_path: Optional[Path] = None,
) -> tuple[List[str], str | None]:
    """
    Build coder CLI command from cli_params config.

    All flags come from model_mapping.json — nothing hardcoded.

    Args:
        coder_config: Full coder config (from resolve_coder)
        prompt_text: Full prompt (system + user message)
        schema_text: JSON schema as string
        session_id: Unique session identifier
        schema_path: Optional path to write schema (for codex-style)

    Returns:
        Tuple of (command list, input_text_for_stdin_or_None)
    """
    cli = coder_config.get("cli_params", {})
    cmd = list(cli.get("cmd", ["qwen"]))

    # Replace {{session_id}} template in flags
    flags = [f.replace("{{session_id}}", session_id) for f in cli.get("flags", [])]
    cmd.extend(flags)

    # Model flag
    if cli.get("model_flag") and coder_config.get("model"):
        cmd.extend([cli["model_flag"], coder_config["model"]])

    # API key flag
    api_key_flag = cli.get("api_key_flag")
    api_key_env = coder_config.get("openai_api_key_env")
    if api_key_flag and api_key_env:
        api_key = os.environ.get(api_key_env, "")
        if api_key:
            cmd.extend([api_key_flag, api_key])

    # Base URL flag
    base_url_flag = cli.get("base_url_flag")
    base_url = coder_config.get("openai_base_url")
    if base_url_flag and base_url:
        cmd.extend([base_url_flag, base_url])

    # Schema flag
    schema_flag = cli.get("schema_flag")
    schema_inline = cli.get("schema_inline", True)
    input_text: Optional[str] = None

    if schema_flag:
        if schema_inline:
            cmd.extend([schema_flag, schema_text])
        elif schema_path:
            cmd.extend([schema_flag, str(schema_path)])

    # Auth type flag (some coders need --auth-type)
    if coder_config.get("auth_type"):
        cmd.extend(["--auth-type", coder_config["auth_type"]])

    # Prompt: either as flag argument or via stdin
    input_flag = cli.get("prompt_flag")
    input_mode = cli.get("input_flag")

    if input_flag:
        # Pass prompt as command-line argument
        cmd.extend([input_flag, prompt_text])
        input_text = None
    elif input_mode == "stdin":
        # Pass prompt via stdin
        input_text = prompt_text
    else:
        # Default: try stdin
        input_text = prompt_text

    return cmd, input_text


# ---------------------------------------------------------------------------
# Sidecar polling helpers
# ---------------------------------------------------------------------------

def _is_valid_sidecar_json(path: Path) -> bool:
    """
    Return True iff path contains a valid meta.json matching the v2 schema.

    Required fields: schema_version, coder_result.status, coder_result.artifacts,
    coder_result.recorded_at.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        sv = data.get("schema_version") or data.get("sidecar_version") or ""
        if sv not in ("v2", "artifact_meta_v1"):
            return False
        cr = data.get("coder_result")
        if not isinstance(cr, dict):
            return False
        if str(cr.get("status", "")).upper() not in ("APPROVED", "REJECTED"):
            return False
        if not isinstance(cr.get("artifacts"), dict):
            return False
        if not str(cr.get("recorded_at") or "").strip():
            return False
        return True
    except Exception:
        return False


def _run_with_sidecar_poll(
    cmd: List[str],
    *,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    input_text: Optional[str] = None,
    timeout_seconds: int,
    sidecar_path: Optional[Path] = None,
    step: str = "",
) -> tuple[int, str, str]:
    """
    Launch cmd via Popen, polling for sidecar completion as early-exit signal.

    If sidecar_path becomes valid before the process exits, the process is
    terminated and rc=0 is returned — allowing the runner to proceed without
    waiting the full timeout.

    Returns:
        (return_code, stdout, stderr)

    Raises:
        subprocess.TimeoutExpired — if neither process nor sidecar is ready
                                    within timeout_seconds.
    """
    import threading

    # Record pre-existing sidecar mtime to avoid stale signals
    sidecar_pre_mtime: Optional[float] = None
    if sidecar_path is not None:
        try:
            sidecar_pre_mtime = sidecar_path.stat().st_mtime if sidecar_path.exists() else None
        except OSError:
            sidecar_pre_mtime = None

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if input_text is not None:
        try:
            proc.stdin.write(input_text)
            proc.stdin.close()
        except BrokenPipeError:
            pass

    chunks_out: List[str] = []
    chunks_err: List[str] = []

    def _drain(pipe, buf: List[str]) -> None:
        for line in iter(pipe.readline, ""):
            buf.append(line)

    t_out = threading.Thread(target=_drain, args=(proc.stdout, chunks_out), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, chunks_err), daemon=True)
    t_out.start()
    t_err.start()

    deadline = time.monotonic() + timeout_seconds
    sidecar_triggered = False

    while True:
        if proc.poll() is not None:
            break
        if time.monotonic() >= deadline:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            t_out.join(timeout=2)
            t_err.join(timeout=2)
            raise subprocess.TimeoutExpired(cmd, timeout_seconds)

        # Check for sidecar completion
        if sidecar_path is not None and sidecar_path.exists():
            try:
                current_mtime = sidecar_path.stat().st_mtime
            except OSError:
                current_mtime = None
            is_new_sidecar = (
                current_mtime is not None
                and (sidecar_pre_mtime is None or current_mtime > sidecar_pre_mtime)
            )
            if is_new_sidecar and _is_valid_sidecar_json(sidecar_path):
                time.sleep(SIDECAR_SETTLE_DELAY_SECONDS)
                sidecar_triggered = True
                label = f" step={step}" if step else ""
                logger.info(f"[coder_invoker] sidecar detected — terminating process early{label}")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break

        time.sleep(SIDECAR_POLL_INTERVAL_SECONDS)

    t_out.join(timeout=5)
    t_err.join(timeout=5)
    rc = 0 if sidecar_triggered else (proc.returncode if proc.returncode is not None else 0)
    return rc, "".join(chunks_out), "".join(chunks_err)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract first JSON object from text (handles markdown code blocks)."""
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    import re
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON-like content
    brace_start = text.find("{")
    if brace_start >= 0:
        for i in range(brace_start, len(text)):
            if text[i] == "}":
                candidate = text[brace_start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

    raise ValueError(f"No valid JSON object found in text: {text[:200]}...")


def _parse_json_payload(text: str) -> Any:
    """Parse JSON from text, return None on failure."""
    candidate = text.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _extract_result_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract result dict from coder response payload."""
    direct = payload.get("result")
    if isinstance(direct, dict):
        return direct
    structured = payload.get("structured_output")
    if isinstance(structured, dict):
        return structured
    for key in ("output", "message", "response", "final", "content"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    try:
        return _extract_json_object(json.dumps(payload))
    except ValueError as exc:
        raise ValueError("Failed to extract model result from structured payload") from exc


def _usage_from_payload(payload: Optional[Dict], step: str = "", coder: str = "") -> UsageData:
    """Extract token usage from coder response payload."""
    usage = UsageData(step=step, coder_used=coder)
    if not payload:
        return usage

    # Look for usage info in various locations
    for usage_key in ("usage", "token_usage", "token_count"):
        u = payload.get(usage_key)
        if isinstance(u, dict):
            usage.input_tokens = u.get("input_tokens") or u.get("prompt_tokens")
            usage.output_tokens = u.get("output_tokens") or u.get("completion_tokens")
            usage.total_tokens = u.get("total_tokens")
            usage.usage_source = "api"
            break

    return usage


# ---------------------------------------------------------------------------
# Main invocation
# ---------------------------------------------------------------------------

def invoke_coder(
    *,
    coder_config: Dict[str, Any],
    step: str,
    prompt_text: str,
    schema_text: str,
    cwd: Path,
    prompt_checksum: str,
    now_iso_fn,
    sidecar_path: Optional[Path] = None,
) -> InvocationResult:
    """
    Invoke coder CLI with data-driven command building.

    Args:
        coder_config: Full coder config from resolve_coder()
        step: Step/adapter name
        prompt_text: Full prompt (system + user message)
        schema_text: JSON schema as string
        cwd: Working directory
        prompt_checksum: SHA256 of prompt text
        now_iso_fn: Function to get current ISO timestamp
        sidecar_path: Path to meta.json sidecar for polling

    Returns:
        InvocationResult with parsed output, usage, and manifest

    Raises:
        CoderInvocationError: If coder process fails
    """
    coder_name = coder_config.get("coder", "unknown")
    session_id = str(uuid.uuid4())

    # Build command from cli_params
    command, input_text = build_command(
        coder_config=coder_config,
        prompt_text=prompt_text,
        schema_text=schema_text,
        session_id=session_id,
    )

    started_at = now_iso_fn()
    started_monotonic = time.monotonic()

    logger.info(
        f"[coder_invoker] step={step} coder={coder_name} "
        f"model={coder_config.get('model', 'N/A')} "
        f"command={' '.join(command[:6])}..."
    )

    timeout_seconds = _coder_timeout_seconds()

    try:
        return_code, stdout, stderr = _run_with_sidecar_poll(
            command,
            cwd=cwd,
            env=dict(os.environ),
            input_text=input_text,
            timeout_seconds=timeout_seconds,
            sidecar_path=sidecar_path,
            step=step,
        )
    except subprocess.TimeoutExpired as exc:
        raise CoderInvocationError(
            message=f"Coder subprocess timed out after {timeout_seconds} seconds.",
            command=command,
            return_code=124,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or ""),
            raw_events=[],
        ) from exc

    finished_at = now_iso_fn()
    duration_ms = int((time.monotonic() - started_monotonic) * 1000)

    status = "OK" if return_code == 0 else "FAILED"
    logger.info(
        f"[coder_invoker] step={step} coder={coder_name} "
        f"return_code={return_code} duration_ms={duration_ms} status={status}"
    )

    # Parse output
    raw_events = [line for line in stdout.splitlines() if line.strip()]
    payload = _parse_json_payload(stdout)

    try:
        if payload is not None and isinstance(payload, dict):
            parsed_result = _extract_result_from_payload(payload)
        elif stdout.strip():
            parsed_result = _extract_json_object(stdout)
        else:
            parsed_result = {}  # sidecar-triggered: stdout may be empty
    except ValueError as exc:
        # If sidecar exists and is valid, don't fail on stdout parsing
        if sidecar_path and _is_valid_sidecar_json(sidecar_path):
            parsed_result = {}
        else:
            raise CoderInvocationError(
                message=f"Failed to extract structured result: {exc}",
                command=command,
                return_code=return_code,
                stdout=stdout,
                stderr=stderr,
                raw_events=raw_events,
            ) from exc

    usage = _usage_from_payload(payload, step=step, coder=coder_name)
    usage.duration_ms = duration_ms
    usage.started_at = started_at
    usage.finished_at = finished_at

    manifest = InvocationManifest(
        step_name=step,
        coder_used=coder_name,
        command=command,
        cwd=str(cwd),
        prompt_checksum=prompt_checksum,
        started_at=started_at,
        finished_at=finished_at,
        return_code=return_code,
    )

    return InvocationResult(
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        parsed_result=parsed_result,
        usage=usage,
        manifest=manifest,
        raw_events=raw_events,
    )
