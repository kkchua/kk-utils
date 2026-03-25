"""
KK-Utils - Logging Configuration

Centralized logging setup with multiple formatters and handlers.

Usage:
    from kk_utils import setup_logging, get_logger

    # Setup once at startup
    setup_logging(level="INFO", log_file="logs/app.log")

    # Get logger anywhere
    logger = get_logger(__name__)
    logger.info("Message")
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def _resolve_log_path(log_file: str) -> Path:
    """
    Resolve log file path using intelligent context detection.

    If log_file is relative (not absolute path), automatically detects
    the appropriate logs directory based on the caller's location.

    Args:
        log_file: Log file path (can be relative or absolute)

    Returns:
        Path: Resolved absolute path to log file
    """
    log_path = Path(log_file)

    # If absolute path, use as-is
    if log_path.is_absolute():
        return log_path

    # If path contains directory separator, resolve relative to cwd
    if '/' in log_file or '\\' in log_file:
        return log_path.resolve()

    # Just a filename like "app.log" - detect context
    from kk_utils.path_resolver import get_logs_path

    # Get logs directory based on caller context
    logs_dir = get_logs_path()

    return logs_dir / log_file


class JsonFormatter(logging.Formatter):
    """JSON formatter for production logging."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName']:
                try:
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data)


class StructuredFormatter(logging.Formatter):
    """Structured formatter for development logging with colors."""

    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        message = f"{color}[{record.levelname:8}]{reset} {timestamp} | {record.name}: {record.getMessage()}"
        
        extra_context = self._get_extra_context(record)
        if extra_context:
            message += f" | {extra_context}"

        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message

    def _get_extra_context(self, record: logging.LogRecord) -> str:
        context = []
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                          'message', 'asctime']:
                try:
                    context.append(f"{key}={value}")
                except:
                    pass
        return " | ".join(context)


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: bool = False,
    rotation: str = "size",  # Always use size-based rotation (10MB files, 5 backups)
    quiet_loggers: Optional[list] = None,
    verbose_packages: bool = False,
) -> None:
    """
    Setup centralized logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, reads from LOG_LEVEL environment variable, defaults to "INFO".
        log_file: Optional log file path
        json_format: Use JSON format (True for production)
        rotation: Log rotation strategy (always uses "size" for 10MB files, 5 backups)
        quiet_loggers: List of logger names to set to WARNING level.
                       If None (default): uses default quiet loggers list
                       If []: shows all logs including HTTP clients
                       If list: quiets only specified loggers
        verbose_packages: If True, shows all package logs (overrides quiet_loggers default).
                          Set to True or LOG_VERBOSE_PACKAGES=true in .env to debug HTTP calls.
    """
    import os
    
    # Get level from parameter or environment variable
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    # Verbose packages from parameter or environment
    if os.getenv("LOG_VERBOSE_PACKAGES", "false").lower() == "true":
        verbose_packages = True

    # Resolve log file path (auto-detect context if relative)
    log_path = None
    if log_file:
        log_path = _resolve_log_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if json_format:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(StructuredFormatter())

    root_logger.addHandler(console_handler)

    if log_file and log_path:
        # Always use size-based rotation for predictable log file sizes
        # Files rotate when size > 10MB, not just at midnight
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            str(log_path),  # ✅ Use resolved log_path
            maxBytes=10*1024*1024,  # 10MB per file
            backupCount=5,  # Keep 5 backup files (~50MB total)
        )

        file_handler.setLevel(level)
        if json_format:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(StructuredFormatter())

        root_logger.addHandler(file_handler)

    # Quiet loggers - only apply if not in verbose mode
    # Default: quiet third-party packages, only show application logs
    if verbose_packages:
        # Show all logs - don't quiet anything
        quiet = quiet_loggers if quiet_loggers is not None else []
    elif quiet_loggers is None:
        # Default: quiet common verbose packages
        quiet = [
            "urllib3", "httpx", "httpcore", "openai",
            "chromadb", "tiktoken", "sqlalchemy", "aiosqlite",
        ]
    else:
        # Use provided list
        quiet = quiet_loggers
    
    for logger_name in quiet:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"Logging initialized: level={level}, file={log_file}, json={json_format}, verbose={verbose_packages}"
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for structured logging."""

    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)
        if exc_type is not None:
            # Log with extra context via extra parameter
            extra_context = ", ".join(f"{k}={v}" for k, v in self.context.items())
            self.logger.error(f"Operation failed: {exc_type.__name__} ({extra_context})", exc_info=True)


def log_function_call(logger_name: str = None):
    """Decorator to log function calls."""
    def decorator(func):
        import functools
        import inspect

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or func.__module__)
            sig = inspect.signature(func)
            params = sig.bind(*args, **kwargs)
            params.apply_defaults()
            logger.debug(f"Entering {func.__name__} with args: {params.arguments}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__} with result: {result}")
                return result
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator
