from .paddleocr import PaddleOCRService
from api.core.logging import logger

class HiOCRService(PaddleOCRService):
    """HiOCR service - Currently using PaddleOCR as backend."""
    
    def __init__(self, use_gpu: bool = False, force_cpu: bool = False):
        """Initialize HiOCR service."""
        logger.info("Initializing HiOCR service (using PaddleOCR backend)")
        super().__init__(use_gpu=use_gpu, force_cpu=force_cpu)
    
    def get_engine_info(self) -> dict:
        """Get information about the OCR engine."""
        base_info = super().get_engine_info()
        return {
            'name': 'HiOCR',
            'version': '2.7.0.3',
            'device': self.device,
            'backend': 'PaddleOCR',
            'capabilities': base_info['capabilities']
        } 