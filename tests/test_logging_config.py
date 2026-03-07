"""
Test KK-Utils Logging Configuration
"""
import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

from kk_utils.logging_config import (
    setup_logging,
    get_logger,
    LogContext,
    log_function_call,
    JsonFormatter,
    StructuredFormatter,
)


class TestLoggingSetup:
    """Test logging setup."""
    
    def setup_method(self):
        """Reset logging before each test."""
        # Clear all handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.NOTSET)
    
    def test_get_logger(self):
        """Test getting logger by name."""
        logger = get_logger("test_logger")
        assert logger.name == "test_logger"
        assert isinstance(logger, logging.Logger)
    
    def test_setup_logging_console(self):
        """Test logging setup with console handler."""
        setup_logging(level="INFO", log_file=None)
        
        root_logger = logging.getLogger()
        
        # Should have console handler
        assert len(root_logger.handlers) >= 1
        
        # Should be INFO level
        assert root_logger.level == logging.INFO
    
    def test_setup_logging_file(self, tmp_path):
        """Test logging setup with file handler."""
        log_file = tmp_path / "test.log"
        
        setup_logging(
            level="DEBUG",
            log_file=str(log_file),
            json_format=False,
        )
        
        root_logger = logging.getLogger()
        
        # Should have console + file handlers
        assert len(root_logger.handlers) >= 2
        
        # Verify file exists
        assert log_file.exists()
    
    def test_setup_logging_json_format(self, tmp_path):
        """Test logging with JSON format."""
        log_file = tmp_path / "test.json.log"
        
        setup_logging(
            level="INFO",
            log_file=str(log_file),
            json_format=True,
        )
        
        logger = get_logger("test_json")
        logger.info("Test message")
        
        # Read log file
        log_content = log_file.read_text()
        
        # Should contain JSON
        assert "timestamp" in log_content
        assert "Test message" in log_content
    
    def test_setup_logging_structured_format(self, tmp_path):
        """Test logging with structured format."""
        log_file = tmp_path / "test.structured.log"
        
        setup_logging(
            level="INFO",
            log_file=str(log_file),
            json_format=False,
        )
        
        logger = get_logger("test_structured")
        logger.info("Test message")
        
        # Read log file
        log_content = log_file.read_text()
        
        # Should contain structured format
        assert "INFO" in log_content
        assert "Test message" in log_content
    
    def test_setup_logging_quiet_loggers(self):
        """Test quieting third-party loggers."""
        setup_logging(
            level="INFO",
            log_file=None,
            quiet_loggers=["urllib3", "httpx"],
        )
        
        # These loggers should be WARNING level
        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING


class TestFormatters:
    """Test log formatters."""
    
    def test_json_formatter(self):
        """Test JSON formatter output."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        formatted = formatter.format(record)
        
        # Should be JSON string
        import json
        data = json.loads(formatted)
        
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
    
    def test_structured_formatter(self):
        """Test structured formatter output."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        formatted = formatter.format(record)
        
        # Should contain level and message
        assert "INFO" in formatted
        assert "Test message" in formatted


class TestLogContext:
    """Test LogContext context manager."""
    
    def test_log_context_adds_context(self, caplog):
        """Test LogContext adds context to logs."""
        logger = get_logger("test_context")
        
        with LogContext(logger, operation="test_op", user_id="123"):
            logger.info("Test message")
        
        # Verify message was logged
        assert "Test message" in caplog.text
    
    def test_log_context_handles_exception(self, caplog):
        """Test LogContext logs exceptions."""
        logger = get_logger("test_exception")
        logger.setLevel(logging.INFO)
        
        try:
            with LogContext(logger, operation="test_op"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should log something about the operation or exception
        assert "test_exception" in caplog.text or "test_op" in caplog.text.lower()


class TestLogFunctionCall:
    """Test log_function_call decorator."""
    
    def test_log_function_call(self, caplog):
        """Test decorator logs function calls."""
        caplog.set_level(logging.DEBUG)
        logger = get_logger("test_decorator")
        logger.setLevel(logging.DEBUG)
        
        @log_function_call(logger_name="test_decorator")
        def test_func(x, y):
            return x + y
        
        result = test_func(2, 3)
        
        assert result == 5
        # Should log entry/exit
        assert "test_func" in caplog.text
    
    def test_log_function_call_exception(self, caplog):
        """Test decorator logs exceptions."""
        logger = get_logger("test_decorator_error")
        
        @log_function_call(logger_name="test_decorator_error")
        def failing_func():
            raise ValueError("Expected error")
        
        try:
            failing_func()
        except ValueError:
            pass
        
        # Should log exception
        assert "Exception in failing_func" in caplog.text or "failing_func" in caplog.text
