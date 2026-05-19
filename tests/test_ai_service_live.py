"""
Live integration tests for AIService — configurable provider + model.

Run against real AI APIs (not mocked). Supports all providers:
    openai, anthropic, qwen/dashscope, ollama, deepseek

Usage:
    # Set env vars then run:
    export TEST_PROVIDER=anthropic
    export TEST_MODEL=claude-sonnet-4-20250514
    export TEST_API_KEY=sk-ant-...          # or let it read from ANTHROPIC_API_KEY
    export TEST_API_BASE_URL=               # optional, defaults per provider

    pytest tests/test_ai_service_live.py -v -s

    # Or inline:
    TEST_PROVIDER=openai TEST_MODEL=gpt-4o-mini pytest tests/test_ai_service_live.py -v -s

    # With a custom test image:
    TEST_IMAGE_PATH=/path/to/image.png pytest tests/test_ai_service_live.py -v -s -k vision

Provider API key mapping (fallback env vars):
    openai    → OPENAI_API_KEY
    anthropic → ANTHROPIC_API_KEY
    qwen      → DASHSCOPE_API_KEY
    dashscope → DASHSCOPE_API_KEY
    ollama    → OLLAMA_API_KEY  (often not required)
    deepseek  → DEEPSEEK_API_KEY

Tests:
    test_text_chat          — Plain text conversation
    test_text_chat_json     — Structured JSON output (generate_json_raw)
    test_vision_description — Image upload + description (generate_vision_raw)
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from kk_utils.ai.ai_service import AIService, TextResult

# Load .env from project root before reading any env vars (avoids CWD mismatch)
# override=False so .env won't overwrite your shell env vars with empty values
_project_root = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    _env_path = _project_root / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response saving
# ---------------------------------------------------------------------------

def _save_response(test_name: str, config: dict, response: dict) -> Path:
    """Save API response to tests/output/{provider}_{model}_{test_name}_{timestamp}.json."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_provider = config["provider"].replace("/", "_")
    safe_model = config["model"].replace("/", "_").replace(" ", "_")
    filename = f"{safe_provider}_{safe_model}_{test_name}_{timestamp}.json"
    filepath = output_dir / filename

    output = {
        "test": test_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "provider": config["provider"],
            "model": config["model"],
            "api_model": config["api_model"],
            "base_url": config.get("base_url"),
        },
        "response": response,
    }

    filepath.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return filepath

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

# Default provider/model — override via env vars
DEFAULT_PROVIDER = os.environ.get("TEST_PROVIDER", "anthropic")
DEFAULT_MODEL = os.environ.get("TEST_MODEL", "claude-sonnet-4-20250514")

# Per-provider default models (used when TEST_MODEL is not set)
_PROVIDER_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "qwen": "qwen-vl-max-latest",
    "dashscope": "qwen-vl-max-latest",
    "ollama": "llama3",
    "deepseek": "deepseek-chat",
}

# Per-provider API key env var mapping
_PROVIDER_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# Per-provider default base URLs
_PROVIDER_BASE_URLS = {
    "qwen": "https://coding-intl.dashscope.aliyuncs.com/v1",
    "dashscope": "https://coding-intl.dashscope.aliyuncs.com/v1",
    "anthropic": "https://api.anthropic.com",
    "deepseek": "https://api.deepseek.com",
    "ollama": "http://localhost:11434/v1",
}


def _resolve_config() -> dict:
    """Resolve provider, model, api_key, base_url from env vars."""
    provider = os.environ.get("TEST_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    model = os.environ.get("TEST_MODEL", "")

    if not model:
        model = _PROVIDER_DEFAULTS.get(provider, "unknown")

    # Resolve API key: let AIService read from provider-specific env var
    # (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) — no need for TEST_API_KEY override
    api_key = os.environ.get("TEST_API_KEY", "")

    # Resolve base URL
    base_url = os.environ.get("TEST_API_BASE_URL", "")
    if not base_url:
        base_url = _PROVIDER_BASE_URLS.get(provider, "")

    return {
        "provider": provider,
        "model": model,
        "api_model": f"{provider}/{model}",
        "api_key": api_key or None,
        "base_url": base_url or None,
    }


def _skip_if_no_api_key(config: dict) -> None:
    """Skip test if no API key is available (env var or config override)."""
    if not config["api_key"]:
        key_var = _PROVIDER_KEY_MAP.get(config["provider"], "API_KEY")
        # Check if the env var actually has a value
        if not os.environ.get(key_var, ""):
            pytest.skip(
                f"No API key for provider '{config['provider']}'. "
                f"Uncomment and fill in {key_var} in .env, or set it in your shell."
            )


def _get_test_image_path() -> Path:
    """Return path to test image file."""
    custom = os.environ.get("TEST_IMAGE_PATH", "")
    if custom:
        p = Path(custom)
        if p.exists():
            return p
        logger.warning(f"TEST_IMAGE_PATH not found: {custom}, using default fixture")

    default = Path(__file__).parent / "fixtures" / "test_image.png"
    if default.exists():
        return default

    pytest.skip(f"No test image found at {default}")
    return default  # never reached but satisfies type checker


def _encode_image_file(image_path: Path) -> tuple[str, str]:
    """Read image file and return (base64_string, mime_type)."""
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime = mime_map.get(suffix, "image/png")
    return b64, mime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def service_config():
    """Resolved provider config for all tests in module."""
    return _resolve_config()


@pytest.fixture(scope="module")
def ai_service(service_config):
    """AIService instance configured for live API testing."""
    _skip_if_no_api_key(service_config)
    service = AIService(
        api_model=service_config["api_model"],
        api_key=service_config["api_key"],
        api_base_url=service_config["base_url"],
        temperature=0.3,  # Lower temp for more deterministic test results
        max_tokens=1024,
    )
    return service


# ---------------------------------------------------------------------------
# Test 1: Plain text chat (no image)
# ---------------------------------------------------------------------------

class TestTextChat:
    """Test plain text conversation via generate_text()."""

    @pytest.mark.asyncio
    async def test_text_chat(self, ai_service: AIService, service_config: dict):
        """Send a simple text prompt and verify a non-empty response."""
        prompt = "What is 2 + 2? Reply with just the number."
        result = await ai_service.generate_text(prompt=prompt)

        assert result is not None, "Result should not be None"
        assert "response" in result, "Result should have 'response' key"
        response_text = result["response"]
        assert response_text, "Response should not be empty"
        assert "4" in response_text, f"Response should contain '4', got: {response_text}"

        saved = _save_response("text_chat", service_config, result)

        print(f"\n[Text Chat] provider={service_config['api_model']}")
        print(f"  Prompt: {prompt}")
        print(f"  Response: {response_text}")
        print(f"  Saved: {saved}")

    @pytest.mark.asyncio
    async def test_text_chat_longer(self, ai_service: AIService, service_config: dict):
        """Send a more complex prompt and verify meaningful response."""
        prompt = (
            "Explain the difference between synchronous and asynchronous "
            "programming in 2-3 sentences."
        )
        result = await ai_service.generate_text(prompt=prompt)

        response_text = result["response"]
        assert response_text, "Response should not be empty"
        assert len(response_text) > 20, f"Response too short: {response_text}"

        # Should mention sync/async concepts
        lower = response_text.lower()
        has_concept = any(w in lower for w in [
            "synchronous", "asynchronous", "async", "sync",
            "block", "wait", "concurrent", "parallel"
        ])
        assert has_concept, f"Response doesn't discuss sync/async: {response_text}"

        saved = _save_response("text_chat_longer", service_config, result)

        print(f"\n[Text Chat (long)] provider={service_config['api_model']}")
        print(f"  Response ({len(response_text)} chars): {response_text[:200]}...")
        print(f"  Saved: {saved}")


# ---------------------------------------------------------------------------
# Test 2: Vision / Image description
# ---------------------------------------------------------------------------

class TestVisionDescription:
    """Test image upload + description via generate_vision_raw()."""

    @pytest.mark.asyncio
    async def test_vision_description(self, ai_service: AIService, service_config: dict):
        """Upload a test image and ask the model to describe it."""
        image_path = _get_test_image_path()
        image_b64, image_mime = _encode_image_file(image_path)

        system_prompt = (
            "You are a vision assistant. Describe the image in JSON format "
            "with keys: 'description' (string), 'dominant_color' (string), "
            "'width' (int), 'height' (int)."
        )
        user_text = "Describe this image in detail."

        result = await ai_service.generate_vision_raw(
            system_prompt=system_prompt,
            user_text=user_text,
            image_b64=image_b64,
            image_mime=image_mime,
        )

        assert result is not None, "Result should not be None"
        assert "raw_content" in result, "Result should have 'raw_content'"
        raw = result["raw_content"]
        assert raw, "Raw content should not be empty"

        saved = _save_response("vision_description", service_config, result)

        # Report metrics
        print(f"\n[Vision] provider={service_config['api_model']}")
        print(f"  Image: {image_path.name} ({len(image_b64)} b64 chars)")
        print(f"  Tokens: {result['prompt_tokens']}+{result['completion_tokens']}={result['total_tokens']}")
        print(f"  Latency: {result['elapsed_ms']}ms")
        print(f"  Raw content: {raw[:300]}...")
        print(f"  Saved: {saved}")

        # Try to parse as JSON and validate structure
        try:
            parsed = json.loads(raw)
            assert isinstance(parsed, dict), "Parsed JSON should be a dict"
            # At minimum, should have some descriptive content
            has_content = any(
                key in parsed
                for key in ["description", "desc", "content", "answer", "text"]
            )
            if not has_content:
                # Some models may not follow the exact schema but still respond
                assert len(str(parsed)) > 10, f"Parsed JSON too sparse: {parsed}"
        except json.JSONDecodeError:
            # Some providers may return markdown or plain text
            assert len(raw) > 10, f"Response too short: {raw}"

    @pytest.mark.asyncio
    async def test_vision_simple_color(self, ai_service: AIService, service_config: dict):
        """Upload the test image and verify a color is identified."""
        image_path = _get_test_image_path()
        image_b64, image_mime = _encode_image_file(image_path)

        system_prompt = "What is the dominant color in this image? Reply with just the color name."
        user_text = "What color is this?"

        result = await ai_service.generate_vision_raw(
            system_prompt=system_prompt,
            user_text=user_text,
            image_b64=image_b64,
            image_mime=image_mime,
        )

        raw = result["raw_content"].lower()
        saved = _save_response("vision_color", service_config, result)

        # Verify some color word is present (works with any test image)
        color_words = [
            "red", "blue", "green", "yellow", "orange", "purple", "white",
            "black", "brown", "gray", "grey", "pink", "gold", "silver",
            "crimson", "scarlet", "ruby", "teal", "cyan", "magenta",
        ]
        has_color = any(w in raw for w in color_words)
        assert has_color, f"No color word found in response: {raw}"

        print(f"\n[Vision (color)] provider={service_config['api_model']}")
        print(f"  Response: {raw}")
        print(f"  Saved: {saved}")


# ---------------------------------------------------------------------------
# Test 3: JSON raw generation (text-only, structured)
# ---------------------------------------------------------------------------

class TestJsonGeneration:
    """Test text-to-JSON generation via generate_json_raw()."""

    @pytest.mark.asyncio
    async def test_json_generation(self, ai_service: AIService, service_config: dict):
        """Generate structured JSON from a text prompt."""
        system_prompt = (
            "Extract entities from the text below. Return JSON with keys: "
            "'person' (string), 'company' (string), 'role' (string)."
        )
        user_text = "John Smith works at Google as a senior software engineer."

        result = await ai_service.generate_json_raw(
            system_prompt=system_prompt,
            user_text=user_text,
        )

        assert result is not None, "Result should not be None"
        raw = result["raw_content"]
        assert raw, "Raw content should not be empty"

        saved = _save_response("json_generation", service_config, result)

        print(f"\n[JSON Gen] provider={service_config['api_model']}")
        print(f"  Tokens: {result['prompt_tokens']}+{result['completion_tokens']}={result['total_tokens']}")
        print(f"  Latency: {result['elapsed_ms']}ms")
        print(f"  Raw: {raw[:300]}")
        print(f"  Saved: {saved}")

        # Validate JSON
        parsed = json.loads(raw)
        assert isinstance(parsed, dict), "Should parse as dict"

        # Should extract at least one entity
        has_entity = any(
            key in parsed
            for key in ["person", "company", "role", "entities", "name"]
        )
        assert has_entity, f"No expected entity keys in JSON: {parsed}"


# ---------------------------------------------------------------------------
# Standalone CLI runner (non-pytest)
# ---------------------------------------------------------------------------

async def run_standalone():
    """
    Run tests outside of pytest — for quick manual testing.

    Usage:
        python -c "
        import asyncio
        from tests.test_ai_service_live import run_standalone
        asyncio.run(run_standalone())
        "
    """
    config = _resolve_config()

    print(f"\n{'='*60}")
    print(f"AIService Live Test")
    print(f"{'='*60}")
    print(f"Provider : {config['provider']}")
    print(f"Model    : {config['model']}")
    print(f"API Model: {config['api_model']}")
    print(f"API Key  : {'****' + config['api_key'][-4:] if config['api_key'] else '(none)'}")
    print(f"Base URL : {config['base_url'] or '(default)'}")
    print(f"{'='*60}\n")

    if not config["api_key"]:
        key_var = _PROVIDER_KEY_MAP.get(config["provider"], "API_KEY")
        print(f"ERROR: No API key found. Set TEST_API_KEY or {key_var}")
        return

    service = AIService(
        api_model=config["api_model"],
        api_key=config["api_key"],
        api_base_url=config["base_url"],
        temperature=0.3,
        max_tokens=1024,
    )

    # Test 1: Text chat
    print("\n[Test 1] Text Chat")
    print("-" * 40)
    try:
        t0 = time.monotonic()
        result = await service.generate_text(prompt="What is 2+2? Reply with just the number.")
        elapsed = int((time.monotonic() - t0) * 1000)
        saved = _save_response("text_chat", config, result)
        print(f"  OK ({elapsed}ms): {result['response']}")
        print(f"  Saved: {saved}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test 2: Vision
    print("\n[Test 2] Vision / Image Description")
    print("-" * 40)
    try:
        image_path = _get_test_image_path()
        image_b64, image_mime = _encode_image_file(image_path)
        t0 = time.monotonic()
        result = await service.generate_vision_raw(
            system_prompt="Describe the image in JSON with 'description' and 'dominant_color' keys.",
            user_text="What is in this image?",
            image_b64=image_b64,
            image_mime=image_mime,
        )
        elapsed = int((time.monotonic() - t0) * 1000)
        saved = _save_response("vision", config, result)
        print(f"  OK ({elapsed}ms, {result['total_tokens']} tokens)")
        print(f"  Response: {result['raw_content'][:200]}...")
        print(f"  Saved: {saved}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test 3: JSON generation
    print("\n[Test 3] JSON Generation")
    print("-" * 40)
    try:
        t0 = time.monotonic()
        result = await service.generate_json_raw(
            system_prompt="Extract: person, company, role as JSON.",
            user_text="Jane Doe works at Apple as CEO.",
        )
        elapsed = int((time.monotonic() - t0) * 1000)
        saved = _save_response("json_gen", config, result)
        print(f"  OK ({elapsed}ms, {result['total_tokens']} tokens)")
        print(f"  Response: {result['raw_content'][:200]}...")
        print(f"  Saved: {saved}")
    except Exception as e:
        print(f"  FAILED: {e}")

    print(f"\n{'='*60}")
    print("Done")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_standalone())
