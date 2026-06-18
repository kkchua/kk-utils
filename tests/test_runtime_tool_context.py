import json

import pytest

from kk_utils.agent_tools import agent_tool
from kk_utils.ai.ai_service import AIService, CallContext


@agent_tool(
    tags=["test"],
    runtime_parameters=["user_id", "persona_collection"],
)
def runtime_tool(
    query: str,
    user_id: str | None = None,
    persona_collection: str | None = None,
) -> dict:
    return {
        "query": query,
        "user_id": user_id,
        "persona_collection": persona_collection,
    }


def _tool_definition():
    schema = dict(runtime_tool.__openai_schema__)
    schema["function_ref"] = runtime_tool
    return schema


def test_runtime_parameters_are_hidden_from_llm_schema():
    parameters = runtime_tool.__openai_schema__["function"]["parameters"]
    assert set(parameters["properties"]) == {"query"}
    assert parameters["required"] == ["query"]


@pytest.mark.asyncio
async def test_sdk_tool_overwrites_untrusted_runtime_arguments(monkeypatch):
    calls = []

    class FakeRegistry:
        def execute(self, name, **kwargs):
            calls.append((name, kwargs))
            return {"ok": True}

    class FakeFunctionTool:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr("kk_utils.agent_tools.get_registry", lambda: FakeRegistry())
    monkeypatch.setattr("kk_utils.ai.ai_service.FunctionTool", FakeFunctionTool)

    service = AIService.__new__(AIService)
    tools = service._build_sdk_tools(
        [_tool_definition()],
        {},
        persona_collection="persona_kengkoon",
        context=CallContext(
            agent_name="agent_me",
            feature_name="chat_with_tools",
            user_id="demo_user",
        ),
    )

    result = await tools[0].on_invoke_tool(
        None,
        json.dumps(
            {
                "query": "experience",
                "user_id": "attacker",
                "persona_collection": "other",
            }
        ),
    )

    assert json.loads(result) == {"ok": True}
    assert calls == [
        (
            "runtime_tool",
            {
                "query": "experience",
                "user_id": "demo_user",
                "persona_collection": "persona_kengkoon",
            },
        )
    ]


def test_runtime_injection_does_not_create_search_filter():
    effective = AIService._inject_runtime_tool_args(
        _tool_definition(),
        {"query": "experience"},
        context=CallContext(
            agent_name="agent_me",
            feature_name="chat_with_tools",
            user_id="demo_user",
        ),
        persona_collection="persona_kengkoon",
    )
    assert effective["user_id"] == "demo_user"
    assert "filter_metadata" not in effective
