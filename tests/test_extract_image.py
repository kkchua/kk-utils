"""
Standalone test: extract image description using real prompt + output schema.

Runs the AIService generate_vision_raw() against the extract_image prompt
and validates the response against the output schema.

Usage:
    # Uses .env or shell env vars for API key
    .venv/bin/python tests/test_extract_image.py

    # Override provider/model
    TEST_PROVIDER=openai TEST_MODEL=gpt-4o .venv/bin/python tests/test_extract_image.py

    # Override image
    TEST_IMAGE_PATH=/path/to/my/image.png .venv/bin/python tests/test_extract_image.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env from project root (override=False preserves shell env vars)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

from kk_utils.ai.ai_service import AIService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

import os

_DEFAULT_PROVIDER = os.environ.get("TEST_PROVIDER", "anthropic")
_DEFAULT_MODEL = os.environ.get("TEST_MODEL", "claude-haiku-4-5-20251001")

_PROVIDER_DEFAULTS = {
    "openai": "gpt-4o",
    "anthropic": "claude-haiku-4-5-20251001",
    "qwen": "qwen-vl-max-latest",
    "dashscope": "qwen-vl-max-latest",
    "ollama": "llama3.2-vision",
}

_PROMPT_PATH = Path(__file__).parent / "test_extract_image_prompt.txt"
_SCHEMA_PATH = Path(__file__).parent / "test_extract_image_output_schema.json"
_TEST_IMAGES_DIR = Path(__file__).parent / "fixtures"


def _get_image_path() -> Path:
    custom = os.environ.get("TEST_IMAGE_PATH", "")
    if custom:
        p = Path(custom)
        if p.exists():
            return p
    # Use whatever PNG is in fixtures (the user's actual test image)
    for f in sorted(_TEST_IMAGES_DIR.glob("*.png")):
        return f
    print("ERROR: No test image found. Set TEST_IMAGE_PATH.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    import base64

    provider = _DEFAULT_PROVIDER
    model = _DEFAULT_MODEL

    print(f"{'='*60}")
    print(f"Extract Image Description Test")
    print(f"{'='*60}")
    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print(f"Prompt   : {_PROMPT_PATH}")
    print(f"Schema   : {_SCHEMA_PATH}")
    print(f"{'='*60}\n")

    # Validate fixtures exist
    if not _PROMPT_PATH.exists():
        print(f"ERROR: Prompt not found: {_PROMPT_PATH}")
        sys.exit(1)
    if not _SCHEMA_PATH.exists():
        print(f"ERROR: Schema not found: {_SCHEMA_PATH}")
        sys.exit(1)

    # Load image
    image_path = _get_image_path()
    image_data = image_path.read_bytes()
    image_b64 = base64.b64encode(image_data).decode("utf-8")
    suffix = image_path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix, "image/png")
    filename = image_path.name
    stem = image_path.stem

    print(f"Image    : {filename} ({len(image_b64)} b64 chars)")
    print()

    # Load prompt + schema
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    output_schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

    # Substitute filename placeholders
    system_prompt = system_prompt.replace("{image_filename}", filename)
    system_prompt = system_prompt.replace("{image_stem}", stem)

    user_text = f"Analyze this image: {filename}"

    # Create AIService (lets it resolve API key from provider-specific env var)
    service = AIService(
        api_model=f"{provider}/{model}",
        temperature=0.3,
        max_tokens=4000,
    )

    if not service.api_key and not service.anthropic_client and not service.client:
        print(f"ERROR: No API key for provider '{provider}'.")
        print(f"Set the appropriate env var (e.g. ANTHROPIC_API_KEY) or fill it in .env")
        sys.exit(1)

    print(f"Calling API: {provider}/{model} ...\n")

    t0 = time.monotonic()
    result = await service.generate_vision_raw(
        system_prompt=system_prompt,
        user_text=user_text,
        image_b64=image_b64,
        image_mime=mime,
    )
    elapsed = int((time.monotonic() - t0) * 1000)

    print(f"{'='*60}")
    print(f"Response")
    print(f"{'='*60}")
    print(f"Latency : {elapsed}ms")
    print(f"Tokens  : {result['prompt_tokens']}+{result['completion_tokens']}={result['total_tokens']}")
    print(f"{'='*60}\n")

    # Parse raw content
    raw = result["raw_content"]

    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Find the opening brace
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1:
            cleaned = cleaned[first_brace:last_brace + 1]

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"FAILED: Could not parse JSON response")
        print(f"Error: {e}")
        print(f"\nRaw response:\n{raw}")
        sys.exit(1)

    # Validate top-level keys
    errors = []
    for key in ["image_filename", "image_stem", "items"]:
        if key not in parsed:
            errors.append(f"Missing top-level key: {key}")

    # Validate items array
    if "items" in parsed:
        items = parsed["items"]
        if not isinstance(items, list):
            errors.append(f"'items' is not an array")
        elif len(items) != 1:
            errors.append(f"'items' must contain exactly 1 object, got {len(items)}")
        else:
            item = items[0]
            required_fields = [
                "subject", "mood", "composition", "lighting",
                "style", "color_palette", "animatable_elements"
            ]
            for field in required_fields:
                if field not in item:
                    errors.append(f"Missing field in item: {field}")
                elif not item[field]:
                    errors.append(f"Empty value for field: {field}")

    # Print parsed result
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    print()

    if errors:
        print(f"FAILED: {len(errors)} validation error(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Save output
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"extract_{stem}_{provider}_{model}_{ts}.json"
    output_path.write_text(json.dumps({
        "test": "extract_image",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"provider": provider, "model": model},
        "image": filename,
        "metrics": {
            "latency_ms": elapsed,
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        },
        "result": parsed,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")
    print(f"\nPASSED")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
