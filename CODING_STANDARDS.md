# KK-Utils - Coding Standards & Development Guide

**Version:** 1.0.0  
**Author:** KK  
**Status:** ✅ Active

---

## Overview

KK-Utils is a shared utility library for all Python projects. It provides:
- Environment loading with fail-fast
- Centralized logging configuration
- YAML config loading with caching
- Path resolution helpers

---

## Development Setup

### 1. Install in Development Mode

```bash
cd kk-utils
pip install -e .
```

### 2. Run Tests

```bash
cd kk-utils/tests
python -m pytest -v
```

### 3. Build Package

```bash
python setup.py sdist bdist_wheel
```

---

## Code Standards

### 1. Module Structure

Each module should have:
- Module docstring at top
- Type hints for all functions
- Comprehensive error handling
- Clear error messages

**Example:**
```python
"""
KK-Utils - Environment Loader

Centralized environment variable loading with fail-fast behavior.

Usage:
    from kk_utils import load_environment
    load_environment()  # Call once at application startup
"""
from pathlib import Path
import logging
import sys

logger = logging.getLogger(__name__)


def load_environment(env_file: str = ".env", required: bool = True) -> bool:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Name of the .env file (default: ".env")
        required: If True, exit if .env not found (default: True)
    
    Returns:
        bool: True if loaded successfully
    
    Raises:
        SystemExit: If .env is missing and required=True
    """
    ...
```

### 2. Error Handling

**Fail-Fast Principle:**
- Missing config → ERROR and exit
- No silent fallbacks
- Clear error messages

**Example:**
```python
if not env_path.exists():
    if required:
        logger.error(f"❌ .env not found at: {env_path}")
        print(f"\n❌ ERROR: .env file missing at: {env_path}")
        print(f"❌ ERROR: Create {env_file} with required variables")
        sys.exit(1)
    else:
        logger.warning(f".env not found at: {env_path}")
        return False
```

### 3. Logging

Use Python's standard logging:
```python
logger = logging.getLogger(__name__)

logger.debug("Detailed info")
logger.info("Normal operation")
logger.warning("Unexpected but handled")
logger.error("Operation failed", exc_info=True)
logger.critical("System-wide failure")
```

### 4. Type Hints

**ALL functions must have type hints:**
```python
from typing import Dict, Any, Optional, List

def load_yaml(
    config_path: str,
    cache_key: Optional[str] = None
) -> Dict[str, Any]:
    ...
```

### 5. Documentation

**Docstrings for all public functions:**
```python
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """
    Setup centralized logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    
    Example:
        setup_logging(level="INFO", log_file="logs/app.log")
    """
    ...
```

---

## Module Guidelines

### env_loader.py

**Purpose:** Centralized .env loading

**Key Features:**
- Singleton pattern (`_env_loaded` flag)
- Fail-fast on missing .env
- Clear error messages

**Test Coverage:**
- ✅ .env exists → loads successfully
- ✅ .env missing, required=True → exits with error
- ✅ .env missing, required=False → returns False
- ✅ Already loaded → returns True immediately

### logging_config.py

**Purpose:** Centralized logging setup

**Key Features:**
- Multiple formatters (JSON, Structured)
- Console + file handlers
- Log rotation support
- Quiet third-party loggers

**Test Coverage:**
- ✅ Console logging works
- ✅ File logging works
- ✅ JSON format works
- ✅ Rotation works

### config_loader.py

**Purpose:** YAML config loading with caching

**Key Features:**
- Singleton instance
- Config caching
- Error handling for missing files

**Test Coverage:**
- ✅ Load YAML file
- ✅ Cache works (second load is instant)
- ✅ Missing file raises FileNotFoundError
- ✅ Clear cache works

### path_resolver.py

**Purpose:** Standardized path resolution

**Key Features:**
- Auto-detect project root
- Backend/project/config paths
- Add to sys.path helper

**Test Coverage:**
- ✅ get_project_root() finds project
- ✅ get_backend_root() finds backend
- ✅ resolve_path() builds correct paths

---

## Testing Standards

### Test Structure

```python
"""
Test KK-Utils Environment Loader
"""
import pytest
from pathlib import Path
from kk_utils.env_loader import load_environment, is_environment_loaded


class TestEnvLoader:
    """Test environment loading."""
    
    def test_load_existing_env(self, tmp_path, monkeypatch):
        """Test loading existing .env file."""
        # Create test .env
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=test_value")
        
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Load environment
        result = load_environment()
        
        # Verify
        assert result is True
        assert is_environment_loaded()
```

### Test Categories

1. **Unit Tests** - Test individual functions
2. **Integration Tests** - Test modules work together
3. **Error Tests** - Test error handling

---

## Version Control

### Version Numbering

**Semantic Versioning:** `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking changes
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes (backward compatible)

### Update Version In:
- `setup.py` → `version="1.0.0"`
- `kk_utils/__init__.py` → `__version__ = "1.0.0"`
- `README.md` → Version badge

### Changelog

Update `CHANGELOG.md` for each release:
```markdown
## [1.0.0] - 2026-03-04

### Added
- Initial release
- env_loader module
- logging_config module
- config_loader module
- path_resolver module
```

---

## Publishing

### To PyPI (Future)

1. **Build package:**
   ```bash
   python setup.py sdist bdist_wheel
   ```

2. **Test on TestPyPI:**
   ```bash
   twine upload --repository testpypi dist/*
   ```

3. **Publish to PyPI:**
   ```bash
   twine upload dist/*
   ```

### Installation from PyPI

```bash
pip install kk-utils
```

---

## Best Practices

### ✅ DO:
- Write comprehensive tests
- Use type hints everywhere
- Document all public functions
- Handle errors gracefully
- Provide clear error messages
- Keep modules focused (single responsibility)
- Use caching where appropriate

### ❌ DON'T:
- Add project-specific logic
- Import from specific projects
- Use global state (except singletons)
- Silent failures
- Skip tests

---

## Adding New Modules

### Checklist

1. **Create module file:**
   ```bash
   touch kk_utils/new_module.py
   ```

2. **Add tests:**
   ```bash
   touch tests/test_new_module.py
   ```

3. **Update `__init__.py`:**
   ```python
   from kk_utils.new_module import new_function
   __all__.append("new_function")
   ```

4. **Update README.md:**
   - Add to usage section
   - Add API reference

5. **Update documentation:**
   - Module docstring
   - Function docstrings
   - Examples

6. **Run tests:**
   ```bash
   python -m pytest -v
   ```

7. **Update version:**
   - Increment MINOR version for new features

---

## Code Review Checklist

Before merging:
- [ ] All tests passing
- [ ] Type hints added
- [ ] Docstrings complete
- [ ] Error handling comprehensive
- [ ] No project-specific code
- [ ] README updated
- [ ] Version incremented
- [ ] CHANGELOG updated

---

## Support

**Issues:** Create issue in GitHub repository  
**Questions:** Check README.md or source code  
**Contributions:** Submit pull request

---

**License:** MIT  
**Author:** KK  
**Last Updated:** 2026-03-04
