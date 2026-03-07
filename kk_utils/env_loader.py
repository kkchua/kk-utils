"""
KK-Utils - Environment Loader

Centralized environment variable loading with fail-fast behavior.

Usage:
    from kk_utils import load_environment
    load_environment()  # Call once at application startup

Fail-Fast Principle:
    - If .env is missing → ERROR and exit immediately
    - No silent fallbacks - makes debugging easier
    - Clear error messages tell user exactly what to do
"""
from pathlib import Path
import logging
import sys

logger = logging.getLogger(__name__)

_env_loaded = False


def load_environment(env_file: str = ".env", required: bool = True) -> bool:
    """
    Load environment variables from .env file.
    
    This is the SINGLE centralized function for loading .env.
    Call this ONCE at application startup.
    
    Args:
        env_file: Name of the .env file (default: ".env")
        required: If True, exit if .env not found (default: True)
    
    Returns:
        bool: True if .env loaded successfully
    
    Raises:
        SystemExit: If .env is missing and required=True
    
    Example:
        # In main.py
        from kk_utils import load_environment
        load_environment()
    """
    global _env_loaded
    
    if _env_loaded:
        logger.debug("Environment already loaded")
        return True
    
    try:
        from dotenv import load_dotenv
        
        # Get .env path from current directory
        env_path = Path.cwd() / env_file
        
        if not env_path.exists():
            if required:
                logger.error(f"❌ .env not found at: {env_path}")
                logger.error(f"❌ This is REQUIRED - application cannot start without it")
                logger.error(f"❌ Create the file with required environment variables")
                print(f"\n❌ ERROR: .env file missing at: {env_path}")
                print(f"❌ ERROR: Application cannot start without .env file")
                print(f"❌ ERROR: Create {env_file} with required variables")
                sys.exit(1)
            else:
                logger.warning(f".env not found at: {env_path}")
                return False
        
        load_dotenv(dotenv_path=env_path)
        logger.info(f"✅ Loaded environment from: {env_path}")
        _env_loaded = True
        return True
            
    except ImportError:
        logger.error("❌ python-dotenv not installed")
        logger.error("❌ Install with: pip install python-dotenv")
        print(f"\n❌ ERROR: python-dotenv not installed")
        print(f"❌ ERROR: Install with: pip install python-dotenv\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to load .env: {e}")
        print(f"\n❌ ERROR: Failed to load .env: {e}\n")
        sys.exit(1)


def is_environment_loaded() -> bool:
    """
    Check if environment has been loaded.
    
    Returns:
        bool: True if load_environment() has been called successfully
    """
    return _env_loaded


def get_env_path(env_file: str = ".env") -> Path:
    """
    Get the path to the .env file.
    
    Args:
        env_file: Name of the .env file (default: ".env")
    
    Returns:
        Path: Absolute path to .env file
    """
    return Path.cwd() / env_file
