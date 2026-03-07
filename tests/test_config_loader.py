"""
Test KK-Utils Configuration Loader
"""
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from kk_utils.config_loader import ConfigLoader


class TestConfigLoader:
    """Test configuration loading."""
    
    def setup_method(self):
        """Reset cache before each test."""
        ConfigLoader._config_cache.clear()
    
    def teardown_method(self):
        """Clear cache after each test."""
        ConfigLoader._config_cache.clear()
    
    def test_load_yaml_existing_file(self, tmp_path):
        """Test loading existing YAML file."""
        # Create test YAML file
        config_file = tmp_path / "test.yaml"
        config_data = {"key": "value", "number": 42}
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config
        result = ConfigLoader.load_yaml(str(config_file))
        
        assert result == config_data
    
    def test_load_yaml_missing_file(self):
        """Test loading missing YAML file raises error."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_yaml("nonexistent.yaml")
    
    def test_load_yaml_with_cache(self, tmp_path):
        """Test loading with caching."""
        # Create test YAML file
        config_file = tmp_path / "test.yaml"
        config_data = {"key": "value"}
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load with cache
        result1 = ConfigLoader.load_yaml(str(config_file), cache_key="test")
        
        # Modify file
        with open(config_file, 'w') as f:
            yaml.dump({"key": "changed"}, f)
        
        # Load again - should return cached value
        result2 = ConfigLoader.load_yaml(str(config_file), cache_key="test")
        
        assert result1 == result2  # Cached
        assert result1["key"] == "value"  # Original value
    
    def test_load_config_by_name(self, tmp_path):
        """Test loading config by name."""
        # Create config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create config file
        config_file = config_dir / "test_config.yaml"
        config_data = {"setting": "value"}
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load by name
        loader = ConfigLoader.instance()
        result = loader.load_config("test_config", config_dir=config_dir)
        
        assert result == config_data
    
    def test_clear_cache_specific(self, tmp_path):
        """Test clearing specific cache entry."""
        # Create test YAML file
        config_file = tmp_path / "test.yaml"
        config_data = {"key": "value"}
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load with cache
        ConfigLoader.load_yaml(str(config_file), cache_key="test")
        
        # Verify cached
        assert "test" in ConfigLoader._config_cache
        
        # Clear specific
        ConfigLoader._config_cache.pop("test", None)
        
        # Verify cleared
        assert "test" not in ConfigLoader._config_cache
    
    def test_clear_cache_all(self, tmp_path):
        """Test clearing all cache entries."""
        # Create test YAML files
        for i in range(3):
            config_file = tmp_path / f"test{i}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump({"key": i}, f)
            
            ConfigLoader.load_yaml(str(config_file), cache_key=f"test{i}")
        
        # Verify cached
        assert len(ConfigLoader._config_cache) == 3
        
        # Clear all
        ConfigLoader._config_cache.clear()
        
        # Verify cleared
        assert len(ConfigLoader._config_cache) == 0
    
    def test_instance_singleton(self):
        """Test instance method returns singleton."""
        loader1 = ConfigLoader.instance()
        loader2 = ConfigLoader.instance()
        
        assert loader1 is loader2
