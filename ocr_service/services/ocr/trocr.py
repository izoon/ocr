import time
import easyocr
import cv2
import os
import torch
import psutil
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Tuple
from .base import BaseOCR
from api.core.logging import logger

class TrOCRService(BaseOCR):
    """TrOCR implementation using EasyOCR for detection and TrOCR for recognition."""
    
    def __init__(self, use_gpu: bool = False):
        """Initialize TrOCR with optional GPU support."""
        self.use_gpu = use_gpu
        
        # Set device
        if torch.cuda.is_available() and use_gpu:
            self.device = "cuda"
            logger.info("🚀 Using CUDA (NVIDIA GPU)")
        elif torch.backends.mps.is_available() and use_gpu:
            self.device = "mps"
            logger.info("🚀 Using MPS (Apple Silicon GPU)")
        else:
            self.device = "cpu"
            logger.info("🖥️ Using CPU (Fallback)")
        
        # Create crops directory
        self.crop_dir = "crops"
        os.makedirs(self.crop_dir, exist_ok=True)
        
        try:
            # Initialize EasyOCR for detection
            logger.info("Initializing EasyOCR for text detection...")
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
            
            # Initialize TrOCR for recognition
            logger.info("Loading TrOCR model...")
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(self.device)
            
            logger.info("TrOCR service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TrOCR service: {str(e)}")
            raise
    
    def crop_detected_lines(self, image: Image.Image) -> List[Tuple[int, Image.Image, List[Tuple[float, float]]]]:
        """Detect and crop text lines using EasyOCR with fallback."""
        try:
            # Try EasyOCR detection first
            temp_path = os.path.join(self.crop_dir, "temp.png")
            image.save(temp_path)
            
            try:
                results = self.reader.readtext(temp_path)
                if not results:
                    logger.warning("EasyOCR detection failed, using fallback detection")
                    # Use fallback detection
                    boxes = self.fallback_detection(image)
                    results = []
                    for x, y, w, h in boxes:
                        # Convert box to EasyOCR format
                        bbox = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                        results.append((bbox, "", 0.0))  # Empty text and confidence for now
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Convert PIL Image to OpenCV format
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            cropped_lines = []
            for i, (bbox, _, _) in enumerate(results):
                x_min = int(min([p[0] for p in bbox]))
                y_min = int(min([p[1] for p in bbox]))
                x_max = int(max([p[0] for p in bbox]))
                y_max = int(max([p[1] for p in bbox]))
                
                # Handle any negative coordinates
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                crop = img[y_min:y_max, x_min:x_max]
                
                # Convert back to PIL Image
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                cropped_lines.append((i, crop_pil, bbox))
            
            return cropped_lines
        except Exception as e:
            logger.error(f"Error in text detection: {str(e)}")
            # Use fallback detection as last resort
            boxes = self.fallback_detection(image)
            cropped_lines = []
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            for i, (x, y, w, h) in enumerate(boxes):
                crop = img[y:y+h, x:x+w]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                bbox = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                cropped_lines.append((i, crop_pil, bbox))
            
            return cropped_lines
    
    def read_with_trocr(self, image: Image.Image) -> str:
        """Recognize text in a cropped line using TrOCR."""
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    
    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process image using EasyOCR for detection and TrOCR for recognition."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Detect and crop text lines
            logger.info("Detecting text regions with EasyOCR...")
            cropped_lines = self.crop_detected_lines(image)
            
            # Recognize text in each line
            logger.info("Recognizing text with TrOCR...")
            text_results = []
            confidences = []
            
            for i, crop, bbox in cropped_lines:
                text = self.read_with_trocr(crop)
                text_results.append(text)
                # Use a default confidence since TrOCR doesn't provide it
                confidences.append(0.95)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            return {
                'text': text_results,
                'confidence': confidences,
                'time': end_time - start_time,
                'memory_used': end_memory - start_memory
            }
        except Exception as e:
            logger.error(f"Error processing image with TrOCR: {str(e)}")
            raise
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get TrOCR engine information."""
        return {
            'name': 'TrOCR',
            'version': 'microsoft/trocr-base-printed',
            'gpu_enabled': self.use_gpu,
            'device': self.device,
            'capabilities': ['text_detection', 'text_recognition']
        } 