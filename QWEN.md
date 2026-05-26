# QWEN.md — KK-Utils

## Project Overview

**KK-Utils** is a shared Python utility library used across multiple projects in the workspace. It provides core infrastructure for environment loading, logging configuration, YAML config loading with caching, path resolution helpers, and AI/RAG capabilities including a ChromaDB-based RAG engine and multi-provider AI service (OpenAI Agents SDK).

**Version:** 1.0.0  
**Author:** KK  
**License:** MIT

---

## Project Structure

```
kk-utils/
├── kk_utils/
│   ├── __init__.py              # Public API re-exports
│   ├── env_loader.py            # load_environment(), is_environment_loaded()
│   ├── logging_config.py        # setup_logging(), get_logger(), LogContext, log_function_call()
│   ├── config_loader.py         # ConfigLoader (YAML with caching), singleton pattern
│   ├── path_resolver.py         # get_project_root(), get_backend_root(), get_config_path()
│   ├── rag_client.py            # RAGClient (API wrapper)
│   ├── database.py              # Database utilities
│   ├── factory.py               # AgentMeFactory, AgentConfig
│   ├── persona_config.py        # PersonaConfig, load_persona(), list_personas()
│   ├── skill_manifest.py        # SkillManifest, get_skill_manifest(), discover_skills()
│   ├── ai/                      # AI Service (multi-provider, OpenAI Agents SDK)
│   │   ├── ai_service.py        # AIService, CallContext, result types
│   │   ├── ai_runner.py         # AI runner utilities
│   │   ├── base_ai_adapter.py   # Base adapter
│   │   ├── schema_adapter_mixin.py
│   │   └── prompts/             # YAML prompt templates
│   ├── rag/                     # RAG Core Engine (direct ChromaDB access)
│   │   ├── rag_engine.py        # RAGEngine
│   │   ├── rag_service.py       # RAG service layer
│   │   ├── chunking.py          # WordChunker, SentenceChunker, ChunkingStrategy
│   │   ├── collection_manager.py
│   │   ├── config.py            # RAGConfig
│   │   ├── context_builder.py
│   │   └── embedding.py         # EmbeddingProvider, get_embedding_function()
│   ├── agent_tools/             # Agent Tools infrastructure
│   ├── agents/                  # Master Agent architecture
│   │   ├── master_agent.py      # MasterAgent orchestrator
│   │   ├── base_agent_adapter.py # Abstract base for all adapters
│   │   ├── agent_registry.py    # Adapter registry singleton
│   │   ├── agent_response.py    # AgentResponse dataclass
│   │   ├── adapters/            # Concrete adapter implementations
│   │   │   ├── agent_me/        # AgentMeAdapter (digital twin)
│   │   │   └── ai_assistant/    # AIAssistantAdapter (general AI)
│   │   ├── prompts/             # Centralized prompt templates
│   │   │   └── master/          # Master prompt files (*.txt)
│   │   └── skill_handlers/      # Skill execution handlers
│   │       ├── base_handler.py  # BaseSkillHandler, SkillContext, SkillResult
│   │       ├── registry.py      # SkillHandlerRegistry
│   │       ├── standard_handler.py
│   │       └── comfyui_handler.py
│   │   └── coder/               # Coder Agent (direct CLI invocation)
│   │       ├── base_coder_adapter.py    # Abstract base for coder adapters
│   │       ├── coder_invoker.py         # Command builder + subprocess + sidecar
│   │       ├── coder_response.py        # CoderResponse dataclass
│   │       ├── coder_registry.py        # Adapter registry singleton
│   │       ├── model_resolver.py        # model_mapping.json loader
│   │       ├── sidecar.py               # meta.json contract
│   │       ├── schema/                  # LLM response schema
│   │       └── adapters/                # Concrete coder adapters
│   │           ├── desc_image/          # Image description
│   │           └── csv_generator/       # CSV generation
│   ├── article_generation/      # Article generation utilities
│   ├── digital_me/              # Digital me (persona) utilities
│   ├── notes/                   # Notes utilities
│   └── web_search/              # Web search utilities
├── config/                      # Configuration files
│   └── model_mapping.json       # Coder CLI params configuration
├── tests/
│   ├── conftest.py
│   ├── test_config_loader.py
│   ├── test_env_loader.py
│   ├── test_logging_config.py
│   └── test_path_resolver.py
├── setup.py
├── requirements.txt
├── config.example.yaml          # Example YAML config for testing
├── .env.example                 # Example .env file
├── CODING_STANDARDS.md          # Development guide & coding standards
└── CLAUDE.md                    # Additional project context
```

---

## Key Dependencies

- **PyYAML >= 6.0** — YAML config loading
- **python-dotenv >= 1.0.0** — .env file loading

Python 3.9+ required.

---

## Installation

```bash
# Development mode (editable, into workspace .venv)
cd kk-utils
../../.venv/bin/pip install -e .   # Linux/macOS
# or
..\..\.venv\Scripts\pip install -e .  # Windows

# Global Python
pip install -e .

# Production
pip install kk-utils
```

Install in editable mode (`-e`) so changes take effect immediately.

---

## Core Module Usage

### Environment Loading
```python
from kk_utils import load_environment

load_environment()  # Call once at startup, fail-fast if .env missing
```

### Logging
```python
from kk_utils import setup_logging, get_logger

setup_logging(level="INFO", log_file="logs/app.log", json_format=False)
logger = get_logger(__name__)
logger.info("Message")
```

Relative `log_file` paths auto-resolve to `backend/logs/<file>`.

### Config Loading
```python
from kk_utils import ConfigLoader

config = ConfigLoader.load_yaml("config/settings.yaml")
loader = ConfigLoader.instance()
config = loader.load_config("subscriptions")  # Loads config/subscriptions.yaml with caching
```

### Path Resolution
```python
from kk_utils import get_project_root, get_config_path, resolve_path

root = get_project_root()
cfg = get_config_path()  # root / "config"
```

### RAG Engine (Direct ChromaDB)
```python
from kk_utils.rag import RAGEngine, RAGConfig

engine = RAGEngine(RAGConfig(...))
results = engine.search("query")
```

### AI Service (Multi-Provider)
```python
from kk_utils.ai import get_ai_service

service = get_ai_service()
result = service.text("prompt")
```

### Agent Tools
```python
from kk_utils.agent_tools import agent_tool, get_registry, execute_tool

@agent_tool
def my_tool(param: str) -> str:
    return f"Result: {param}"
```

---

## Agents Architecture

The `kk_utils.agents` module implements a **Master Agent orchestrator** with pluggable adapter pattern.

### Core Layers

| Layer | File | Purpose |
|-------|------|---------|
| **Orchestrator** | `master_agent.py` | Routes requests, resolves persona, selects adapter, loads tools, builds prompts |
| **Adapter Interface** | `base_agent_adapter.py` | Abstract base class all adapters implement; shared `execute_chat()` via AIService |
| **Adapter Registry** | `agent_registry.py` | Singleton registry for adapter lookup by name |
| **Skill Handlers** | `skill_handlers/` | Handle execution patterns: standard (direct tool call), ComfyUI (submit & forget) |

### Request Flow

```
User message → MasterAgent.chat()
  ↓
1. Load persona metadata from PostgreSQL personas
2. Select adapter by persona.adapter_type (e.g., "agent_me", "ai_assistant")
3. Derive tool tags from the persona's assigned skills
4. Load tools from AgentRegistry by derived skill_tags
5. Build system prompt (DB llm_prompts → master prompt file → adapter default → generic fallback)
6. [Pipeline check] If execution_type matches a handler, skip LLM → route to handler
7. Execute chat via adapter.execute_chat() using AIService
8. Post-process response → return AgentResponse
```

### Built-in Adapters

| Adapter | adapter_name | Default Skills | Purpose |
|---------|-------------|----------------|---------|
| **AgentMeAdapter** | `"agent_me"` | `digital_me`, `notes`, `web_search` | Digital twin with personal knowledge base |
| **AIAssistantAdapter** | `"ai_assistant"` | None (empty) | General conversational AI |

### Pipeline Bypass (Skill Handlers)

If `execution_type` is provided and a matching handler is registered, MasterAgent **skips LLM entirely** and routes directly to the handler:

```python
response = await agent.chat(
    message="Generate image",
    persona_name="ai_assistant",
    user_id="user123",
    execution_type="comfyui",     # Routes to ComfyUISkillHandler
    skill_adapter="image_variation",
    prompt_name="master_prompt_qwen",
)
```

**Built-in handlers:**
- **StandardSkillHandler** — Direct tool call → return result (fallback for most skills)
- **ComfyUISkillHandler** — Submit to ComfyUI server → return task_id (submit & forget, results to Google Drive)

### Prompt Loading Priority

1. **Database** — `llm_prompts` table (namespace="agent", adapter="", name={template_name})
2. **Master prompts** — `prompts/master/{template_name}.txt`
3. **Adapter default** — `adapters/{adapter_type}/prompts/default.txt`
4. **Fallback** — Minimal generic system prompt

### Prompt Naming Convention

- Persona prompts:
  - `namespace="agent"`
  - `adapter=""`
  - `name="{persona_name}"`
- Skill prompts:
  - `namespace="{skill_name}"`
  - `adapter="{variant}"`
  - `name="{prompt_key}"`

Examples:
- `csv_generator / image_variation / master`
- `csv_generator / picturebook / schema`
- `hot_topics / analysis / master`

### Key Classes & Functions

| Class/Function | Purpose |
|---------------|---------|
| `MasterAgent` | Orchestrator — `chat()`, `_load_system_prompt()`, `_build_messages()` |
| `BaseAgentAdapter` | Abstract base — `execute_chat()`, `get_skills()`, `get_skill_tags()`, `build_system_prompt()` |
| `AgentRegistry` | Singleton — `instance()`, `register()`, `get_adapter()` |
| `AgentResponse` | Dataclass — standardized response with `response_text`, `agent_type`, `metadata`, `error` |
| `SkillContext` | Context data — user_id, persona, attachments, model, extra dict |
| `SkillResult` | Result data — output, success, error, downloads, requires_polling |
| `load_master_prompt(name)` | Load prompt from `prompts/master/{name}.txt` |

### Adapter Directory Structure

Each adapter has its own directory for prompts and schemas:
```
adapters/
├── agent_me/
│   ├── adapter.py          # AgentMeAdapter implementation
│   ├── tools.py            # Agent-specific tool definitions
│   └── prompts/            # Adapter-specific prompts/schemas
└── ai_assistant/
    └── adapter.py          # AIAssistantAdapter implementation
```

### Governor Integration

`BaseAgentAdapter._execute_chat_with_tools()` integrates with `PersonalAssistantGovernor` for:
- Tool call limit validation before AI call
- Global system prompt suffix (tool limits, rules) appended in `MasterAgent._load_system_prompt()`

---

## Coder Agent Architecture

The `kk_utils.agents.coder` module provides **direct coder CLI invocation** as an alternative to API-based AI service. Instead of using `AIService`, coder adapters launch coder CLIs (qwen, claude, codex) via subprocess and parse results from a `meta.json` sidecar contract.

### Key Design: Data-Driven CLI Parameters

All CLI flags are configured in `model_mapping.json` — no hardcoding of `--output-format`, `--approval-mode`, etc. in Python code:

```json
{
  "coder_aliases": {
    "desc_image": {
      "coder": "qwen",
      "model": "qwen-coder-plus-latest",
      "cli_params": {
        "cmd": ["qwen"],
        "flags": ["--output-format", "json", "--approval-mode", "yolo"],
        "prompt_flag": "-p",
        "model_flag": "-m",
        "api_key_flag": "--openai-api-key",
        "base_url_flag": "--openai-base-url"
      }
    }
  }
}
```

**`cli_params` fields:**
| Field | Purpose |
|-------|---------|
| `cmd` | Base command + binary name |
| `flags` | Static flags always included (supports `{{session_id}}` template) |
| `prompt_flag` | How to pass prompt (`-p` for qwen, null=stdin) |
| `input_flag` | `"stdin"` = pipe via stdin, null = not used |
| `model_flag` | Flag for model override (`-m`) |
| `schema_flag` | Flag for schema file (`--json-schema` for claude) |
| `schema_inline` | Pass schema as inline string vs temp file path |
| `api_key_flag` | Flag for API key injection |
| `base_url_flag` | Flag for base URL override |

### Architecture

```
MasterAgent.chat()
  ↓
adapter_type starts with "coder_"? → route to coder path
  ↓
CoderRegistry.get_adapter("desc_image") → DescImageCoderAdapter
  ↓
resolve_coder("desc_image") → model_mapping.json → full config with cli_params
  ↓
build_command() → assembles CLI from cli_params
  ↓
invoke_coder() → subprocess.Popen + sidecar polling
  ↓
read & validate meta.json sidecar
  ↓
return CoderResponse
```

### Request Flow (Coder)

```
1. MasterAgent detects persona.adapter_type = "coder_desc_image"
2. Resolves coder adapter from CoderRegistry
3. Resolves coder config from model_mapping.json
4. Builds system prompt (DB → file → fallback)
5. Builds CLI command from cli_params
6. Launches coder via subprocess, polls for meta.json sidecar
7. Reads & validates sidecar (v2 schema)
8. Validates artifact files exist
9. Enriches sidecar with runner_data
10. Returns CoderResponse wrapped as AgentResponse
```

### Built-in Coder Adapters

| Adapter | adapter_name | coder_alias | Purpose |
|---------|-------------|-------------|---------|
| **DescImageCoderAdapter** | `"desc_image"` | `"desc_image"` | Image description generation |
| **CsvGeneratorCoderAdapter** | `"csv_generator"` | `"csv_generator"` | CSV file generation |

### Adding a New Coder Adapter

1. **Create adapter directory:** `coder/adapters/my_adapter/`
2. **Implement adapter:**
   ```python
   class MyCoderAdapter(BaseCoderAdapter):
       adapter_name = "my_adapter"
       coder_alias = "my_adapter"  # key in model_mapping.json

       def build_system_prompt(self, context):
           return self.load_prompt_from_db_or_file("default")

       async def execute_coder(self, prompt_text, context):
           return await self._invoke(prompt_text, context=context)
   ```
3. **Add to model_mapping.json** with full `cli_params` config
4. **Register in master_agent.py** `_register_builtin_adapters()`
5. **Register in coder/__init__.py** exports

### Sidecar Contract (meta.json)

Coder agents communicate results via a `meta.json` sidecar file:

```json
{
  "schema_version": "v2",
  "coder_result": {
    "status": "APPROVED",
    "remark": "Image description generated",
    "artifacts": {"OUTPUT_FILE": "images/desc.json"},
    "recorded_at": "2026-04-26T..."
  },
  "runner_data": { ... }  // appended by enrich_sidecar
}
```

### Prompt Loading (Coder)

1. **Database** — `llm_prompts` table (namespace="coder", adapter="{adapter_name}", name="{template}")
2. **Adapter file** — `adapters/{adapter_name}/prompts/{template}.txt`
3. **Shared file** — `coder/prompts/{adapter_name}_{template}.txt`
4. **Fallback** — Hardcoded `FALLBACK_SYSTEM_PROMPT` constant

### Key Files

| File | Purpose |
|------|---------|
| `coder/base_coder_adapter.py` | Abstract base for all coder adapters |
| `coder/coder_invoker.py` | Command builder + subprocess + sidecar polling |
| `coder/coder_response.py` | CoderResponse dataclass |
| `coder/coder_registry.py` | Adapter registry singleton |
| `coder/model_resolver.py` | model_mapping.json loader + alias resolver |
| `coder/sidecar.py` | meta.json read/validate/enrich + exceptions |
| `coder/schema/llm_response_schema.json` | Coder output schema |
| `config/model_mapping.json` | CLI params configuration |

---

## Development Conventions

### Coding Standards (from CODING_STANDARDS.md)

- **Type hints** required on ALL functions
- **Docstrings** required on ALL public functions
- **Fail-fast** principle: missing config → ERROR and exit, no silent fallbacks
- **Singletons** use `ClassName.instance()` class method
- **Public exports** live in `kk_utils/__init__.py`
- **Error handling** must be comprehensive with clear messages

### Startup Order
1. `load_environment()` — first, before any env access
2. `setup_logging()` — after env loading

### Testing
```bash
cd kk-utils/tests
python -m pytest -v
```

### Build & Publish
```bash
python setup.py sdist bdist_wheel
```

### Version Management
Semantic versioning (MAJOR.MINOR.PATCH). Update in:
- `setup.py` → `version="1.0.0"`
- `kk_utils/__init__.py` → `__version__ = "1.0.0"`
- `README.md` → Version badge

---

## Module Guidelines

### env_loader
- Singleton pattern (`_env_loaded` flag)
- Fail-fast on missing .env
- Tests: .env exists/missing/required=False/already loaded

### logging_config
- Multiple formatters (JSON, structured)
- Console + file handlers
- Log rotation support
- Quiet third-party loggers

### config_loader
- Singleton instance
- Config caching
- FileNotFoundError for missing files
- Clear cache support

### path_resolver
- Auto-detect project root
- Backend/project/config paths
- Add to sys.path helper

---

## Adding New Modules

1. Create module: `kk_utils/new_module.py`
2. Add tests: `tests/test_new_module.py`
3. Update `__init__.py` exports and `__all__`
4. Update README.md (usage + API reference)
5. Run tests: `python -m pytest -v`
6. Increment MINOR version for new features

---

## Notes

- This library is a **local dependency** — not published to PyPI
- Used by `personal-assistant/backend/app/core/env_loader.py` and `logging_config.py` (thin wrappers)
- Modules should remain **generic** — no project-specific logic
- Keep modules focused on single responsibilities
