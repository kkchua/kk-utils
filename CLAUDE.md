# kk-utils — CLAUDE.md

## Project Overview

Shared Python utility library used by the Personal Assistant backend and other Python projects in the workspace. Provides env loading, logging, YAML config, path resolution, and a RAG client.

---

## Installation

```bash
# From workspace root — installs into shared .venv
cd kk-utils
..\..\.venv\Scripts\pip install -e .

# Or with global Python
pip install -e .
```

Install in editable mode (`-e`) so changes take effect immediately without reinstalling.

---

## Project Structure

```
kk-utils/
  kk_utils/
    __init__.py         # Public API re-exports
    env_loader.py       # load_environment(), is_environment_loaded()
    logging_config.py   # setup_logging(), get_logger(), LogContext, log_function_call()
    config_loader.py    # ConfigLoader (YAML with caching), singleton pattern
    path_resolver.py    # get_project_root(), get_backend_root(), get_config_path(), resolve_path()
    rag/
      rag_client.py     # RAG client (ChromaDB wrapper)
  tests/
    test_*.py
  setup.py
  requirements.txt
  README.md
```

---

## Module Reference

### env_loader
```python
from kk_utils import load_environment
load_environment()                          # Load .env, fail-fast if missing
load_environment(env_file=".env", required=False)  # Non-strict
```

### logging_config
```python
from kk_utils import setup_logging, get_logger
setup_logging(level="INFO", log_file="logs/app.log", json_format=False, rotation="midnight")
logger = get_logger(__name__)
```
If `log_file` is relative, auto-resolves to `backend/logs/<file>`.

### config_loader
```python
from kk_utils import ConfigLoader
config = ConfigLoader.load_yaml("config/settings.yaml")
loader = ConfigLoader.instance()           # Singleton
config = loader.load_config("governor")   # Loads config/governor.yaml with caching
ConfigLoader.clear_cache()
```

### path_resolver
```python
from kk_utils import get_project_root, get_config_path, resolve_path
root = get_project_root()
cfg  = get_config_path()                  # root / "config"
```

---

## Conventions

- All public exports live in `kk_utils/__init__.py`
- Singletons use `ClassName.instance()` class method
- `load_environment()` should be called once at app startup before any env access
- `setup_logging()` should be called once after `load_environment()`

---

## Testing

```bash
cd kk-utils/tests
python -m pytest -v
```

---

## Notes

- This library is a local dependency — not published to PyPI
- Used by `personal-assistant/backend/app/core/env_loader.py` and `logging_config.py` (thin wrappers)
- The RAG module in `kk_utils/rag/` provides a shared ChromaDB client interface
