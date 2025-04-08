import pytest
import os
import platform
import torch
from PIL import Image
from ocr_service.services.ocr.easyocr import EasyOCRService
from api.core.logging import logger

@pytest.fixture
def test_image():
    """Load test image."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "..", "data", "food.png")
    logger.info(f"Loading test image from: {image_path}")
    return Image.open(image_path)

@pytest.fixture
def easyocr_service():
    """Create EasyOCR service instance."""
    return EasyOCRService(use_gpu=False)  # Use CPU for testing

def test_easyocr_init():
    """Test EasyOCR initialization."""
    service = EasyOCRService(use_gpu=False)
    assert isinstance(service, EasyOCRService)
    assert service.device == "cpu"

def test_process_image(easyocr_service, test_image):
    """Test image processing with EasyOCR."""
    # Process image
    result = easyocr_service.process_image(test_image)
    
    # Check result structure
    assert 'text' in result
    assert 'confidence' in result
    assert 'time' in result
    assert 'memory_used' in result
    
    # Check result types
    assert isinstance(result['text'], list)
    assert isinstance(result['confidence'], list)
    assert isinstance(result['time'], float)
    assert isinstance(result['memory_used'], float)
    
    # Log results
    logger.info(f"Processed text: {result['text']}")
    logger.info(f"Confidence scores: {result['confidence']}")
    logger.info(f"Processing time: {result['time']}s")
    logger.info(f"Memory used: {result['memory_used']}MB")

def test_engine_info(easyocr_service):
    """Test engine information."""
    info = easyocr_service.get_engine_info()
    assert info['name'] == 'EasyOCR'
    assert info['version'] == '1.7.1'
    assert info['device'] == 'cpu'
    assert 'capabilities' in info

def test_platform_compatibility():
    """Test platform-specific behavior."""
    is_macos = platform.system() == "Darwin"
    cuda_available = torch.cuda.is_available()
    
    service = EasyOCRService(use_gpu=True)
    
    if is_macos:
        assert service.device == "mps" if torch.backends.mps.is_available() else "cpu"
    elif cuda_available:
        assert service.device == "cuda"
    else:
        assert service.device == "cpu" 