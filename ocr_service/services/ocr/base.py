from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
import cv2
from api.core.logging import logger
import pytesseract

class BaseOCR(ABC):
    """Base class for OCR implementations."""
    
    @abstractmethod
    def __init__(self):
        """Initialize the OCR engine."""
        pass
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess the image for OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image object
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if image is too large
        max_size = 2000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def fallback_detection(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Fallback text detection using traditional computer vision.
        Returns list of (x, y, width, height) bounding boxes.
        """
        try:
            # Convert PIL Image to OpenCV format
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort contours by area
            boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                if area > 100:  # Minimum area threshold
                    boxes.append((x, y, w, h))
            
            # Sort boxes from top to bottom
            boxes.sort(key=lambda b: b[1])
            
            logger.info(f"Fallback detection found {len(boxes)} text regions")
            return boxes
            
        except Exception as e:
            logger.error(f"Error in fallback detection: {str(e)}")
            return []
    
    def fallback_recognition(self, image: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> List[str]:
        """
        Fallback text recognition using basic image processing.
        Returns list of recognized text.
        """
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
                
                # Use Tesseract as fallback recognition
                text = pytesseract.image_to_string(binary)
                texts.append(text.strip())
            
            return texts
            
        except Exception as e:
            logger.error(f"Error in fallback recognition: {str(e)}")
            return []
    
    @abstractmethod
    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process an image and return OCR results.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict containing:
            - text: List of detected text strings
            - confidence: List of confidence scores
            - time: Processing time in seconds
            - memory_used: Memory usage in MB
        """
        pass
    
    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the OCR engine.
        
        Returns:
            Dict containing engine name, version, and capabilities
        """
        pass 