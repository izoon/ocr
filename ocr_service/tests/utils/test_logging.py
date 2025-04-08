import pytest
import logging
from api.core.logging import logger, setup_logging

def test_logger_initialization():
    """Test logger initialization."""
    assert isinstance(logger, logging.Logger)
    assert logger.name == "ocr_service"
    assert logger.level == logging.INFO

def test_logger_handlers():
    """Test logger handlers."""
    assert len(logger.handlers) > 0
    for handler in logger.handlers:
        assert isinstance(handler, (logging.StreamHandler, logging.FileHandler))

def test_logger_levels():
    """Test different logging levels."""
    # Test info level
    with pytest.LogCapture() as logs:
        logger.info("Test info message")
        assert "Test info message" in logs.text
    
    # Test warning level
    with pytest.LogCapture() as logs:
        logger.warning("Test warning message")
        assert "Test warning message" in logs.text
    
    # Test error level
    with pytest.LogCapture() as logs:
        logger.error("Test error message")
        assert "Test error message" in logs.text

def test_setup_logging():
    """Test logging setup function."""
    # Reset logger
    logger.handlers = []
    
    # Setup logging
    setup_logging()
    
    # Verify setup
    assert len(logger.handlers) > 0
    assert logger.level == logging.INFO 