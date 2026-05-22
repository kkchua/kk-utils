from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No more fake Anthropic responses configured")
        return self._responses.pop(0)


class _FakeAnthropicClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def _make_response(content, *, input_tokens=11, output_tokens=7, stop_reason="end_turn", model="claude-sonnet-4-20250514"):
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
        model=model,
    )


@pytest.fixture
def anthropic_service(monkeypatch):
    fake_module = ModuleType("anthropic")
    created_clients = []

    def fake_async_anthropic(*args, **kwargs):
        client = _FakeAnthropicClient([])
        client._init_args = args
        client._init_kwargs = kwargs
        created_clients.append(client)
        return client

    fake_module.AsyncAnthropic = fake_async_anthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    import kk_utils.ai.ai_service as ai_service

    importlib.reload(ai_service)
    return ai_service, created_clients


@pytest.mark.asyncio
async def test_anthropic_text_call_returns_model(anthropic_service):
    ai_service, created_clients = anthropic_service
    created_clients.clear()

    client_response = _make_response([SimpleNamespace(type="text", text='{"response":"hello"}')])

    def fake_client_factory(*args, **kwargs):
        client = _FakeAnthropicClient([client_response])
        client._init_args = args
        client._init_kwargs = kwargs
        created_clients.append(client)
        return client

    ai_service.AsyncAnthropic = fake_client_factory

    service = ai_service.AIService(api_model="anthropic/claude-sonnet-4-20250514", api_key="test-key")
    result = await service._call_ai(
        system_prompt="You are a helpful assistant.",
        user_text="Say hello",
        output_type=ai_service.TextResult,
    )

    assert result.response == "hello"
    assert service.anthropic_client.messages.calls[0]["model"] == "claude-sonnet-4-20250514"
    assert service.anthropic_client.messages.calls[0]["messages"][0]["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_anthropic_generate_vision_raw_uses_image_blocks(anthropic_service):
    ai_service, created_clients = anthropic_service
    created_clients.clear()

    client_response = _make_response([SimpleNamespace(type="text", text='{"description":"a cat"}')])

    def fake_client_factory(*args, **kwargs):
        client = _FakeAnthropicClient([client_response])
        client._init_args = args
        client._init_kwargs = kwargs
        created_clients.append(client)
        return client

    ai_service.AsyncAnthropic = fake_client_factory

    service = ai_service.AIService(api_model="anthropic/claude-sonnet-4-20250514", api_key="test-key")
    result = await service.generate_vision_raw(
        system_prompt="Describe the image in JSON.",
        user_text="What is in this image?",
        image_b64="ZmFrZQ==",
        image_mime="image/png",
    )

    assert result["raw_content"] == '{"description":"a cat"}'
    call = service.anthropic_client.messages.calls[0]
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "What is in this image?"
    assert content[1]["type"] == "image"
    assert content[1]["source"]["media_type"] == "image/png"
    assert "Return ONLY valid JSON" in call["system"]


@pytest.mark.asyncio
async def test_anthropic_chat_with_tools_executes_tool_calls(anthropic_service, monkeypatch):
    ai_service, created_clients = anthropic_service
    created_clients.clear()

    first_response = _make_response(
        [
            SimpleNamespace(
                type="tool_use",
                id="tooluse-1",
                name="lookup",
                input={"query": "abc"},
            )
        ]
    )
    second_response = _make_response([SimpleNamespace(type="text", text="done")])

    def fake_client_factory(*args, **kwargs):
        client = _FakeAnthropicClient([first_response, second_response])
        client._init_args = args
        client._init_kwargs = kwargs
        created_clients.append(client)
        return client

    ai_service.AsyncAnthropic = fake_client_factory

    class FakeRegistry:
        def __init__(self):
            self.calls = []

        def execute(self, name, **kwargs):
            self.calls.append((name, kwargs))
            return {"result": "ok"}

    fake_registry = FakeRegistry()
    monkeypatch.setattr("kk_utils.agent_tools.get_registry", lambda: fake_registry)

    service = ai_service.AIService(api_model="anthropic/claude-sonnet-4-20250514", api_key="test-key")
    text = await service.chat_with_tools(
        message="look up abc",
        tools=[
            {
                "function": {
                    "name": "lookup",
                    "description": "Look something up",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            }
        ],
        system_prompt="You are a tool-using assistant.",
    )

    assert text == "done"
    assert fake_registry.calls == [("lookup", {"query": "abc"})]
    assert len(service.anthropic_client.messages.calls) == 2
