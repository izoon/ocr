import time
import easyocr
import cv2
import os
import torch
import psutil
import pytesseract
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Tuple
from .base import BaseOCR
from api.core.logging import logger
import platform

class TesseractService(BaseOCR):
    """Tesseract implementation using CRAFT for detection."""
    
    def __init__(self, use_gpu: bool = False, force_cpu: bool = False):
        """Initialize Tesseract with CRAFT detection."""
        super().__init__()
        self.use_gpu = use_gpu
        
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
        
        try:
            # Initialize EasyOCR for CRAFT detection
            logger.info("Initializing CRAFT detection...")
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
            
            # Set Tesseract config
            self.tesseract_config = '--oem 3 --psm 6'  # Use LSTM OCR Engine Mode with automatic page segmentation
            
            logger.info(f"Tesseract service initialized successfully using {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract service: {str(e)}")
            raise
    
    def detect_text_regions(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect text regions using CRAFT."""
        try:
            # Save temporarily for EasyOCR
            temp_path = os.path.join("temp.png")
            image.save(temp_path)
            
            try:
                # Use CRAFT for detection
                results = self.reader.readtext(temp_path)
                boxes = []
                
                for bbox, _, _ in results:
                    x_min = int(min([p[0] for p in bbox]))
                    y_min = int(min([p[1] for p in bbox]))
                    x_max = int(max([p[0] for p in bbox]))
                    y_max = int(max([p[1] for p in bbox]))
                    
                    # Handle any negative coordinates
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
                
                return boxes
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logger.error(f"Error in CRAFT detection: {str(e)}")
            # Use fallback detection
            return self.fallback_detection(image)
    
    def recognize_text(self, image: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> List[str]:
        """Recognize text using Tesseract."""
        try:
            texts = []
            img = np.array(image)
            
            for x, y, w, h in boxes:
                # Crop the region
                crop = img[y:y+h, x:x+w]
                
                # Convert to grayscale
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                
                # Apply thresholding
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Use Tesseract for recognition
                text = pytesseract.image_to_string(binary, config=self.tesseract_config)
                texts.append(text.strip())
            
            return texts
        except Exception as e:
            logger.error(f"Error in Tesseract recognition: {str(e)}")
            return []
    
    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process image using CRAFT for detection and Tesseract for recognition."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Detect text regions
            logger.info("Detecting text regions with CRAFT...")
            boxes = self.detect_text_regions(image)
            
            if not boxes:
                logger.warning("No text regions detected, using fallback detection")
                boxes = self.fallback_detection(image)
            
            # Recognize text
            logger.info("Recognizing text with Tesseract...")
            text_results = self.recognize_text(image, boxes)
            
            # Use default confidence since Tesseract doesn't provide it
            confidences = [0.95] * len(text_results)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            return {
                'text': text_results,
                'confidence': confidences,
                'time': end_time - start_time,
                'memory_used': end_memory - start_memory
            }
        except Exception as e:
            logger.error(f"Error processing image with Tesseract: {str(e)}")
            raise
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get Tesseract engine information."""
        return {
            'name': 'Tesseract',
            'version': pytesseract.get_tesseract_version(),
            'device': self.device,
            'capabilities': ['text_detection', 'text_recognition']
        } 