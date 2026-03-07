"""
Test KK-Utils Path Resolver
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from kk_utils.path_resolver import (
    get_project_root,
    get_backend_root,
    get_config_path,
    resolve_path,
    add_to_path,
)


class TestGetProjectRoot:
    """Test get_project_root function."""
    
    def test_finds_git_marker(self, tmp_path):
        """Test finding project root via .git marker."""
        # Create fake project structure
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        
        # Create subdirectory
        subdir = project_root / "sub" / "dir"
        subdir.mkdir(parents=True)
        
        # Change to subdirectory
        import os
        old_cwd = os.getcwd()
        os.chdir(str(subdir))
        
        try:
            # Should find project root
            result = get_project_root()
            assert result == project_root
        finally:
            os.chdir(old_cwd)
    
    def test_finds_requirements_marker(self, tmp_path):
        """Test finding project root via requirements.txt marker."""
        # Create fake project structure
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "requirements.txt").write_text("test")
        
        # Create subdirectory
        subdir = project_root / "sub"
        subdir.mkdir()
        
        # Change to subdirectory
        import os
        old_cwd = os.getcwd()
        os.chdir(str(subdir))
        
        try:
            result = get_project_root()
            assert result == project_root
        finally:
            os.chdir(old_cwd)
    
    def test_fallback_to_cwd(self):
        """Test fallback to current directory."""
        # No markers in current directory
        result = get_project_root()
        # Should return a Path object (might be parent dir if markers found)
        assert isinstance(result, Path)
        assert result.exists()


class TestGetBackendRoot:
    """Test get_backend_root function."""
    
    @patch('kk_utils.path_resolver.get_project_root')
    def test_finds_backend_folder(self, mock_get_project_root, tmp_path):
        """Test finding backend folder."""
        # Create project structure
        project_root = tmp_path / "project"
        project_root.mkdir()
        backend_root = project_root / "backend"
        backend_root.mkdir()
        
        mock_get_project_root.return_value = project_root
        
        result = get_backend_root()
        assert result == backend_root
    
    @patch('kk_utils.path_resolver.get_project_root')
    def test_fallback_current_is_backend(self, mock_get_project_root, tmp_path):
        """Test fallback when current dir is backend."""
        # Create backend structure
        backend_root = tmp_path / "backend"
        backend_root.mkdir()
        (backend_root / "app").mkdir()
        
        mock_get_project_root.return_value = tmp_path / "nonexistent"
        
        import os
        old_cwd = os.getcwd()
        os.chdir(str(backend_root))
        
        try:
            result = get_backend_root()
            assert result == backend_root
        finally:
            os.chdir(old_cwd)


class TestGetConfigPath:
    """Test get_config_path function."""
    
    @patch('kk_utils.path_resolver.get_project_root')
    def test_default_config_dir(self, mock_get_project_root, tmp_path):
        """Test default config directory."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        mock_get_project_root.return_value = project_root
        
        result = get_config_path()
        assert result == project_root / "config"
    
    @patch('kk_utils.path_resolver.get_project_root')
    def test_custom_config_dir(self, mock_get_project_root, tmp_path):
        """Test custom config directory name."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        mock_get_project_root.return_value = project_root
        
        result = get_config_path("settings")
        assert result == project_root / "settings"


class TestResolvePath:
    """Test resolve_path function."""
    
    @patch('kk_utils.path_resolver.get_project_root')
    def test_resolve_from_project(self, mock_get_project_root, tmp_path):
        """Test resolving path from project root."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        mock_get_project_root.return_value = project_root
        
        result = resolve_path("project", "subdir", "file.txt")
        assert result == project_root / "subdir" / "file.txt"
    
    @patch('kk_utils.path_resolver.get_backend_root')
    def test_resolve_from_backend(self, mock_get_backend_root, tmp_path):
        """Test resolving path from backend root."""
        backend_root = tmp_path / "backend"
        backend_root.mkdir()
        
        mock_get_backend_root.return_value = backend_root
        
        result = resolve_path("backend", "app", "main.py")
        assert result == backend_root / "app" / "main.py"
    
    @patch('kk_utils.path_resolver.get_config_path')
    def test_resolve_from_config(self, mock_get_config_path, tmp_path):
        """Test resolving path from config directory."""
        config_path = tmp_path / "config"
        config_path.mkdir()
        
        mock_get_config_path.return_value = config_path
        
        result = resolve_path("config", "settings.yaml")
        assert result == config_path / "settings.yaml"
    
    def test_resolve_unknown_base(self):
        """Test resolving with unknown base uses cwd."""
        result = resolve_path("unknown", "file.txt")
        assert result == Path.cwd() / "file.txt"


class TestAddToPath:
    """Test add_to_path function."""
    
    def test_adds_new_path(self):
        """Test adding new path to sys.path."""
        initial_len = len(sys.path)
        
        test_path = Path("/test/path")
        add_to_path(test_path)
        
        assert len(sys.path) == initial_len + 1
        assert str(test_path.resolve()) in sys.path
    
    def test_doesnt_add_duplicate(self):
        """Test not adding duplicate path."""
        test_path = Path("/test/path")
        test_path_str = str(test_path.resolve())
        
        # Add once
        add_to_path(test_path)
        initial_len = len(sys.path)
        
        # Try to add again
        add_to_path(test_path)
        
        # Length should not change
        assert len(sys.path) == initial_len
