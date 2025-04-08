import logging
import sys
from .config import settings

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ocr_service.log")
        ]
    )
    
    # Create logger for OCR service
    logger = logging.getLogger("ocr_service")
    logger.setLevel(settings.LOG_LEVEL)
    
    return logger

logger = setup_logging() 