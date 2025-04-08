from typing import Dict, Any
from .base import BaseOCR
from .easyocr import EasyOCRService
from .trocr import TrOCRService
from .paddleocr import PaddleOCRService
from .tesseract import TesseractService
from .hiocr import HiOCRService
from api.core.config import settings
from api.core.logging import logger

class OCRFactory:
    """Factory for creating OCR service instances."""
    
    @staticmethod
    def create_engine(engine_type: str, use_gpu: bool = False, force_cpu: bool = False) -> BaseOCR:
        """Create an OCR engine instance."""
        logger.info(f"Creating OCR engine: {engine_type} (GPU: {use_gpu}, Force CPU: {force_cpu})")
        
        engines = {
            'easyocr': EasyOCRService,
            'trocr': TrOCRService,
            'paddleocr': PaddleOCRService,
            'tesseract': TesseractService,
            'hiocr': HiOCRService
        }
        
        if engine_type not in engines:
            raise ValueError(f"Unsupported OCR engine: {engine_type}")
        
        try:
            engine = engines[engine_type](use_gpu=use_gpu, force_cpu=force_cpu)
            logger.info(f"Successfully created {engine_type} engine")
            return engine
        except Exception as e:
            logger.error(f"Failed to create {engine_type} engine: {str(e)}")
            raise 