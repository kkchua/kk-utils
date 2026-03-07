# KK-Utils

Core utility library for Python projects.

## Features

- **Environment Loading** - Centralized .env loading with fail-fast
- **Logging Configuration** - Structured logging with multiple formatters
- **Config Loading** - YAML configuration with caching
- **Path Resolution** - Standardized path helpers

## Installation

### For Personal Assistant Project (Workspace venv)

```bash
# Install in workspace virtual environment
cd kk-utils
..\..\.venv\Scripts\pip install -e .
```

### Development Mode (Global Python)

```bash
cd kk-utils
pip install -e .
```

### Production Mode

```bash
pip install kk-utils
```

## Usage

### Environment Loading

```python
from kk_utils import load_environment

# Load .env once at startup
load_environment()

# Use environment variables
import os
api_key = os.getenv("OPENAI_API_KEY")
```

### Logging

```python
from kk_utils import setup_logging, get_logger

# Setup once at startup
setup_logging(
    level="INFO",
    log_file="logs/app.log",
    json_format=False,
)

# Get logger anywhere
logger = get_logger(__name__)
logger.info("Message")
```

### Config Loading

```python
from kk_utils import ConfigLoader

# Load config
config = ConfigLoader.load_yaml("config/settings.yaml")

# Or with caching
loader = ConfigLoader.instance()
config = loader.load_config("subscriptions")
```

### Path Resolution

```python
from kk_utils import get_project_root, get_config_path

# Get paths
project_root = get_project_root()
config_path = get_config_path()
settings = config_path / "settings.yaml"
```

## API Reference

### Environment

- `load_environment(env_file=".env", required=True)` - Load .env file
- `is_environment_loaded()` - Check if loaded
- `get_env_path(env_file=".env")` - Get .env path

### Logging

- `setup_logging(level, log_file, json_format, rotation)` - Setup logging
- `get_logger(name)` - Get logger instance
- `LogContext(logger, **context)` - Context manager for structured logging
- `log_function_call()` - Decorator to log function calls

### Config

- `ConfigLoader.load_yaml(path, cache_key)` - Load YAML file
- `ConfigLoader.load_config(name, config_dir)` - Load config by name
- `ConfigLoader.instance()` - Get singleton instance
- `ConfigLoader.clear_cache()` - Clear cache

### Paths

- `get_project_root()` - Get project root
- `get_backend_root()` - Get backend root
- `get_config_path(config_dir)` - Get config directory
- `resolve_path(base, *parts)` - Resolve path from base
- `add_to_path(path)` - Add path to sys.path

## Development

### Run Tests

```bash
cd kk-utils/tests
python -m pytest -v
```

### Build Package

```bash
python setup.py sdist bdist_wheel
```

### Install Locally

```bash
pip install -e .
```

## License

MIT License

## Author

KK
