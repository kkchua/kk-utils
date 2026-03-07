"""
Test KK-Utils Environment Loader
"""
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Test module
from kk_utils.env_loader import load_environment, is_environment_loaded, get_env_path


class TestEnvLoader:
    """Test environment loading."""
    
    def setup_method(self):
        """Reset state before each test."""
        import kk_utils.env_loader as env_module
        env_module._env_loaded = False
    
    def test_get_env_path_default(self):
        """Test get_env_path with default .env."""
        env_path = get_env_path()
        assert env_path.name == ".env"
        assert env_path.parent == Path.cwd()
    
    def test_get_env_path_custom(self):
        """Test get_env_path with custom filename."""
        env_path = get_env_path(".env.test")
        assert env_path.name == ".env.test"
    
    def test_is_environment_loaded_initial(self):
        """Test initial state - not loaded."""
        assert is_environment_loaded() is False
    
    def test_load_environment_missing_required(self, tmp_path, monkeypatch):
        """Test missing .env with required=True exits."""
        # Change to temp directory without .env
        monkeypatch.chdir(tmp_path)
        
        # Should exit when .env missing and required=True
        with pytest.raises(SystemExit):
            load_environment(required=True)
    
    def test_load_environment_missing_optional(self, tmp_path, monkeypatch):
        """Test missing .env with required=False returns False."""
        # Change to temp directory without .env
        monkeypatch.chdir(tmp_path)
        
        # Should return False when .env missing and required=False
        result = load_environment(required=False)
        
        assert result is False
    
    def test_load_environment_creates_file(self, tmp_path, monkeypatch):
        """Test loading creates .env file if it doesn't exist."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Create .env file
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=test_value")
        
        # Load environment (should work now)
        result = load_environment(required=False)
        
        # Should load successfully
        assert result is True
