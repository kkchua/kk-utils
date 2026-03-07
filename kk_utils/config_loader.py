"""
KK-Utils - Configuration Loader

Centralized YAML configuration loading with caching.

Usage:
    from kk_utils import ConfigLoader
    
    # Load config
    config = ConfigLoader.load_yaml("config/settings.yaml")
    
    # Or use instance method with caching
    loader = ConfigLoader.instance()
    config = loader.load_config("subscriptions")
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Centralized YAML configuration loader with caching."""
    
    _instance: Optional["ConfigLoader"] = None
    _config_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        pass

    @classmethod
    def instance(cls) -> "ConfigLoader":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def load_yaml(config_path: str, cache_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Load YAML configuration file.

        Args:
            config_path: Path to YAML file
            cache_key: Optional cache key (if None, no caching)

        Returns:
            Dictionary with configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        path = Path(config_path)

        if cache_key and cache_key in ConfigLoader._config_cache:
            return ConfigLoader._config_cache[cache_key]

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if cache_key:
            ConfigLoader._config_cache[cache_key] = config

        return config

    def load_config(self, config_name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load config by name with automatic path resolution.

        Args:
            config_name: Config name without extension (e.g., "subscriptions")
            config_dir: Optional config directory (default: current dir / "config")

        Returns:
            Dictionary with configuration
        """
        if config_dir is None:
            config_dir = Path.cwd() / "config"

        config_path = config_dir / f"{config_name}.yaml"
        return self.load_yaml(str(config_path), cache_key=config_name)

    def clear_cache(self, config_name: Optional[str] = None) -> None:
        """
        Clear config cache.

        Args:
            config_name: Specific config to clear, or None for all
        """
        if config_name:
            ConfigLoader._config_cache.pop(config_name, None)
        else:
            ConfigLoader._config_cache.clear()
