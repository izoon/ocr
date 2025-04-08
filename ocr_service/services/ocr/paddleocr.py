import time
import psutil
import paddle
import platform
from paddleocr import PaddleOCR
from typing import Dict, Any
from PIL import Image
import numpy as np
import os
from .base import BaseOCR
from api.core.logging import logger

class PaddleOCRService(BaseOCR):
    """PaddleOCR implementation."""
    
    def __init__(self, use_gpu: bool = False, force_cpu: bool = False):
        """Initialize PaddleOCR with optional GPU support."""
        super().__init__()
        self.use_gpu = use_gpu
        self.force_cpu = force_cpu
        
        # Check platform compatibility
        is_macos = platform.system() == "Darwin"
        if is_macos or force_cpu:
            if force_cpu:
                logger.info("Forcing CPU usage as requested")
            else:
                logger.warning("PaddleOCR is not fully supported on macOS. Some features may not work.")
            self.use_gpu = False
            self.device = "cpu"
            paddle.device.set_device('cpu')
            return
        
        try:
            # Check CUDA availability
            if paddle.device.is_compiled_with_cuda() and use_gpu and not force_cpu:
                logger.info("CUDA is available, using GPU")
                self.device = "cuda"
                paddle.device.set_device('gpu')
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.8'
            else:
                logger.info("Using CPU for PaddleOCR")
                self.use_gpu = False
                self.device = "cpu"
                paddle.device.set_device('cpu')
            
            # Initialize PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=self.use_gpu,
                show_log=True,
                det_db_thresh=0.1,
                det_db_box_thresh=0.1,
                det_db_unclip_ratio=1.6,
                use_mp=False,
                total_process_num=1,
                use_fp16=False
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise
    
    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process image using PaddleOCR."""
        # Check platform compatibility
        if platform.system() == "Darwin":
            logger.error("PaddleOCR image processing is not supported on macOS")
            return {
                'text': [],
                'confidence': [],
                'time': 0.0,
                'memory_used': 0.0,
                'error': 'PaddleOCR is not supported on macOS'
            }
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Convert PIL Image to numpy array
            image_array = np.array(image)
            
            # Run OCR
            results = self.ocr.ocr(image_array, cls=True)
            
            if results is None:
                logger.warning("PaddleOCR returned None results")
                return {
                    'text': [],
                    'confidence': [],
                    'time': time.time() - start_time,
                    'memory_used': psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
                }
            
            # Extract text and confidence scores
            text_results = []
            confidences = []
            for line in results[0]:
                if line is not None and len(line) >= 2:
                    text_results.append(line[1][0])
                    confidences.append(line[1][1])
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            return {
                'text': text_results,
                'confidence': confidences,
                'time': end_time - start_time,
                'memory_used': end_memory - start_memory
            }
        except Exception as e:
            logger.error(f"Error processing image with PaddleOCR: {str(e)}")
            raise
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get PaddleOCR engine information."""
        is_macos = platform.system() == "Darwin"
        return {
            'name': 'PaddleOCR',
            'version': paddle.__version__,
            'device': self.device,
            'capabilities': ['text_detection', 'text_recognition', 'angle_classification'],
            'platform_compatible': not is_macos
        } 