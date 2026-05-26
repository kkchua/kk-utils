from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import kk_utils.ai.ai_service as ai_service


class _FakeModelSettings:
    def __init__(self, extra_body=None):
        self.extra_body = extra_body


class _FakeSDKAgent:
    last_init = None

    def __init__(self, **kwargs):
        type(self).last_init = kwargs


@contextmanager
def _fake_trace(_name):
    yield


@pytest.mark.asyncio
async def test_generate_json_raw_passes_configured_extra_body(monkeypatch):
    async def fake_run(_agent, _user_messages):
        return SimpleNamespace(
            final_output='{"ok":true}',
            raw_responses=[],
        )

    monkeypatch.setattr(ai_service, "AGENTS_SDK_AVAILABLE", True)
    monkeypatch.setattr(ai_service, "ModelSettings", _FakeModelSettings)
    monkeypatch.setattr(ai_service, "SDKAgent", _FakeSDKAgent)
    monkeypatch.setattr(ai_service, "OpenAIChatCompletionsModel", lambda **kwargs: kwargs)
    monkeypatch.setattr(ai_service, "Runner", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(ai_service, "trace", _fake_trace)

    service = ai_service.AIService(
        api_model="deepseek/deepseek-v4-flash",
        api_key="test-key",
        extra_body={"thinking": {"type": "disabled"}},
    )

    result = await service.generate_json_raw(
        system_prompt="Return JSON.",
        user_text="Say hello.",
    )

    assert result["raw_content"] == '{"ok":true}'
    model_settings = _FakeSDKAgent.last_init["model_settings"]
    assert model_settings.extra_body == {
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 4000,
    }
