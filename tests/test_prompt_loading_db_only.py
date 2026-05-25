from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_KK_UTILS_DIR = Path(__file__).resolve().parent.parent / "kk_utils"
_AGENTS_DIR = _KK_UTILS_DIR / "agents"


def _install_package_shims():
    if "kk_utils" not in sys.modules:
        pkg = ModuleType("kk_utils")
        pkg.__path__ = [str(_KK_UTILS_DIR)]
        sys.modules["kk_utils"] = pkg

    if "kk_utils.agents" not in sys.modules:
        pkg = ModuleType("kk_utils.agents")
        pkg.__path__ = [str(_AGENTS_DIR)]
        sys.modules["kk_utils.agents"] = pkg


def _import_persona_config():
    _install_package_shims()
    return importlib.import_module("kk_utils.persona_config")


def _import_master_agent():
    _install_package_shims()
    return importlib.import_module("kk_utils.agents.master_agent")


def _import_prompt_module():
    _install_package_shims()
    return importlib.import_module("kk_utils.agents.prompts")


def _import_prompt_loader():
    _install_package_shims()
    return importlib.import_module("kk_utils.llm_prompt_loader")


def _install_prompt_service(monkeypatch, prompt_obj):
    app_module = ModuleType("app")
    services_module = ModuleType("app.services")
    prompt_service_module = ModuleType("app.services.prompt_service")

    class _PromptService:
        def get(self, db, namespace: str, adapter: str = "", name: str = ""):
            return prompt_obj

    prompt_service_module.get_prompt_service = lambda: _PromptService()

    monkeypatch.setitem(sys.modules, "app", app_module)
    monkeypatch.setitem(sys.modules, "app.services", services_module)
    monkeypatch.setitem(sys.modules, "app.services.prompt_service", prompt_service_module)


def _install_persona_dependencies(monkeypatch, persona_obj, prompt_obj):
    _install_prompt_service(monkeypatch, prompt_obj)

    models_module = ModuleType("app.models")
    persona_module = ModuleType("app.models.persona")
    persona_module.Persona = type("Persona", (), {"persona_name": "persona_name"})

    skill_discovery_module = ModuleType("app.services.skill_discovery_service")

    class _Discovery:
        def get_skill_details(self, skill_name):
            return SimpleNamespace(tags=[skill_name])

    skill_discovery_module.get_skill_discovery_service = lambda: _Discovery()

    monkeypatch.setitem(sys.modules, "app.models", models_module)
    monkeypatch.setitem(sys.modules, "app.models.persona", persona_module)
    monkeypatch.setitem(sys.modules, "app.services.skill_discovery_service", skill_discovery_module)
    skill_manifest_module = importlib.import_module("kk_utils.skill_manifest")
    monkeypatch.setattr(
        skill_manifest_module,
        "get_skill_manifest",
        lambda skill_name: SimpleNamespace(tags=[skill_name]),
    )

    class _Query:
        def __init__(self, result):
            self._result = result

        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return self._result

    class _Session:
        def query(self, model):
            return _Query(persona_obj)

    return _Session()


def test_master_agent_loads_persona_prompt_from_db(monkeypatch):
    master_agent_module = _import_master_agent()
    persona_config_module = _import_persona_config()
    _install_prompt_service(monkeypatch, SimpleNamespace(id=11, prompt_text="Prompt from DB"))

    persona = persona_config_module.PersonaConfig(
        name="kengkoon",
        display_name="Keng Koon",
        collection="persona_kengkoon",
        skills=[],
        skill_tags=[],
        system_prompt="",
        adapter_prompt_template="kengkoon",
    )
    agent = master_agent_module.MasterAgent(auto_register_adapters=False, auto_register_handlers=False)

    prompt = agent._load_system_prompt(object(), persona, db_session=object())

    assert prompt == "Prompt from DB"


def test_master_agent_raises_when_persona_prompt_missing(monkeypatch):
    master_agent_module = _import_master_agent()
    persona_config_module = _import_persona_config()
    _install_prompt_service(monkeypatch, None)

    persona = persona_config_module.PersonaConfig(
        name="kengkoon",
        display_name="Keng Koon",
        collection="persona_kengkoon",
        skills=[],
        skill_tags=[],
        system_prompt="",
        adapter_prompt_template="kengkoon",
    )
    agent = master_agent_module.MasterAgent(auto_register_adapters=False, auto_register_handlers=False)

    with pytest.raises(ValueError, match="Missing enabled llm_prompts row"):
        agent._load_system_prompt(object(), persona, db_session=object())


def test_load_persona_requires_enabled_db_prompt(monkeypatch):
    persona_config_module = _import_persona_config()
    persona_record = SimpleNamespace(
        persona_name="kengkoon",
        display_name="Keng Koon",
        collection="persona_kengkoon",
        skills=["digital_me"],
        adapter_type="agent_me",
    )
    session = _install_persona_dependencies(monkeypatch, persona_record, None)

    with pytest.raises(ValueError, match="missing an enabled llm_prompts row"):
        persona_config_module.load_persona("kengkoon", db_session=session)


def test_load_llm_prompt_is_db_only():
    prompt_loader_module = _import_prompt_loader()
    with pytest.raises(ValueError, match="Static prompt fallbacks have been removed"):
        prompt_loader_module.load_llm_prompt("agent", "", "kengkoon", fallback_text="legacy")


def test_static_master_prompt_files_removed():
    prompt_module = _import_prompt_module()
    assert prompt_module.list_master_prompts() == []
    with pytest.raises(FileNotFoundError, match="have been removed"):
        prompt_module.load_master_prompt("kengkoon")
