import time
import psutil
import easyocr
import torch
import platform
from typing import Dict, Any
from PIL import Image
import numpy as np
from .base import BaseOCR
from api.core.logging import logger

class EasyOCRService(BaseOCR):
    """EasyOCR implementation."""
    
    def __init__(self, use_gpu: bool = False, force_cpu: bool = False):
        """Initialize EasyOCR service."""
        super().__init__()
        
        # Determine device
        if force_cpu:
            self.device = "cpu"
            use_gpu = False
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            self.device = "mps"
            use_gpu = True
        elif torch.cuda.is_available() and use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"
            use_gpu = False
        
        logger.info(f"Using {self.device} for EasyOCR")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        
    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process an image using EasyOCR."""
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Process image
        start_time = self.get_time()
        start_memory = self.get_memory()
        
        results = self.reader.readtext(image_np)
        
        # Extract text and confidence
        texts = []
        confidences = []
        for bbox, text, conf in results:
            texts.append(text)
            confidences.append(conf)
        
        # Calculate metrics
        end_time = self.get_time()
        end_memory = self.get_memory()
        
        return {
            'text': texts,
            'confidence': confidences,
            'time': end_time - start_time,
            'memory_used': end_memory - start_memory
        }
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the OCR engine."""
        return {
            'name': 'EasyOCR',
            'version': '1.7.1',
            'device': self.device,
            'capabilities': [
                'text_detection',
                'text_recognition',
                'multiple_languages'
            ]
        } 