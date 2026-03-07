"""
KK-Utils - Path Resolution Helpers

Standardized path resolution for consistent project structure.

Usage:
    from kk_utils import get_project_root, get_backend_root, get_config_path
    
    # Get paths
    project_root = get_project_root()
    backend_root = get_backend_root()
    config_path = get_config_path()
"""
from pathlib import Path
import sys


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Searches upward from current file to find project root.
    Looks for common markers: .git/, setup.py, requirements.txt
    
    Returns:
        Path: Absolute path to project root
    
    Raises:
        RuntimeError: If project root cannot be determined
    """
    current = Path.cwd()
    
    # Search upward for project markers
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or \
           (parent / "setup.py").exists() or \
           (parent / "requirements.txt").exists():
            return parent
    
    # Fallback: use current directory
    return current


def get_backend_root() -> Path:
    """
    Get backend root directory.
    
    Assumes structure: project_root/backend/
    
    Returns:
        Path: Absolute path to backend root
    """
    project_root = get_project_root()
    backend_path = project_root / "backend"
    
    if backend_path.exists():
        return backend_path
    
    # Fallback: check if current dir is backend
    if (Path.cwd() / "app").exists():
        return Path.cwd()
    
    return project_root


def get_config_path(config_dir: str = "config") -> Path:
    """
    Get configuration directory path.

    Args:
        config_dir: Name of config directory (default: "config")

    Returns:
        Path: Absolute path to config directory
    """
    project_root = get_project_root()
    return project_root / config_dir


def get_logs_path(log_dir: str = "logs") -> Path:
    """
    Get logs directory path.

    Automatically detects if running in backend or gradio-apps context
    and returns the appropriate logs directory.

    Args:
        log_dir: Name of logs directory (default: "logs")

    Returns:
        Path: Absolute path to logs directory

    Examples:
        - If called from backend/app/*/*.py → returns backend/logs/
        - If called from gradio-apps/*/*.py → returns gradio-apps/logs/
    """
    # First, check if current working directory gives us a hint
    cwd = Path.cwd()
    if cwd.name == "backend" and (cwd / "app").exists():
        return cwd / log_dir
    if cwd.name == "gradio-apps" and (cwd.parent / "backend").exists():
        return cwd / log_dir
    
    # Try to detect context from the calling module's location
    import inspect

    # Get the frame of the caller
    frame = inspect.currentframe()
    try:
        # Walk up the frame stack to find the caller
        caller_frame = frame
        for _ in range(5):  # Check up to 5 levels
            if caller_frame is None:
                break
            filename = caller_frame.f_code.co_filename
            if filename and not filename.startswith('<'):
                caller_path = Path(filename).resolve()
                parent = caller_path.parent

                # Detect backend context (looking for 'app' folder inside 'backend')
                if parent.name == "app":
                    backend_parent = parent.parent
                    if backend_parent.name == "backend":
                        return backend_parent / log_dir

                # Detect gradio-apps context
                if parent.name == "gradio-apps":
                    return parent / log_dir

                # Check parent directories for backend or gradio-apps
                for p in caller_path.parents:
                    if p.name == "backend" and (p.parent / "gradio-apps").exists():
                        # We're in personal-assistant/backend/
                        return p / log_dir
                    if p.name == "gradio-apps" and (p.parent / "backend").exists():
                        # We're in personal-assistant/gradio-apps/
                        return p / log_dir

            caller_frame = caller_frame.f_back
    finally:
        del frame

    # Fallback: use project root detection
    project_root = get_project_root()

    # Check if we're in personal-assistant project
    personal_assistant = project_root / "personal-assistant"
    if personal_assistant.exists():
        # Default to backend logs for project root calls
        backend_logs = personal_assistant / "backend" / log_dir
        return backend_logs

    # Last fallback: create logs in project root
    return project_root / log_dir


def resolve_path(base: str, *parts: str) -> Path:
    """
    Resolve path from base directory.
    
    Args:
        base: Base directory ("project", "backend", "config")
        *parts: Path parts to append
    
    Returns:
        Path: Resolved absolute path
    """
    if base == "project":
        root = get_project_root()
    elif base == "backend":
        root = get_backend_root()
    elif base == "config":
        root = get_config_path()
    else:
        root = Path.cwd()
    
    return root / Path(*parts)


def add_to_path(path: Path) -> None:
    """
    Add path to sys.path if not already present.
    
    Args:
        path: Path to add
    """
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
