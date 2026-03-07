"""
KK-Utils Test Suite

Run tests:
    cd kk-utils/tests
    python -m pytest -v

Run with coverage:
    python -m pytest -v --cov=kk_utils --cov-report=html
"""
import pytest
import sys
from pathlib import Path

# Add kk_utils to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
