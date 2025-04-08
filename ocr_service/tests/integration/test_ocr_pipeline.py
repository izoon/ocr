import pytest
import os
from PIL import Image
from ocr_service.services.ocr.factory import OCRFactory
from api.core.logging import logger

@pytest.fixture
def test_image():
    """Load test image."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "..", "data", "food.png")
    logger.info(f"Loading test image from: {image_path}")
    return Image.open(image_path)

@pytest.fixture
def ocr_factory():
    """Create OCR factory instance."""
    return OCRFactory()

def test_all_engines_initialization(ocr_factory):
    """Test initialization of all OCR engines."""
    engines = ["easyocr", "paddleocr", "trocr", "hiocr"]
    for engine in engines:
        ocr_engine = ocr_factory.create_engine(engine, use_gpu=False)
        assert ocr_engine is not None
        assert hasattr(ocr_engine, 'process_image')
        assert hasattr(ocr_engine, 'get_engine_info')

def test_all_engines_processing(ocr_factory, test_image):
    """Test image processing with all OCR engines."""
    engines = ["easyocr", "paddleocr", "trocr", "hiocr"]
    for engine in engines:
        # Create engine
        ocr_engine = ocr_factory.create_engine(engine, use_gpu=False)
        
        # Process image
        result = ocr_engine.process_image(test_image)
        
        # Check result structure
        assert 'text' in result
        assert 'confidence' in result
        assert 'time' in result
        assert 'memory_used' in result
        
        # Log results
        logger.info(f"{engine} results:")
        logger.info(f"Text: {result['text']}")
        logger.info(f"Confidence: {result['confidence']}")
        logger.info(f"Time: {result['time']}s")
        logger.info(f"Memory: {result['memory_used']}MB")

def test_engine_switching(ocr_factory, test_image):
    """Test switching between different OCR engines."""
    # Process with EasyOCR
    easyocr = ocr_factory.create_engine("easyocr", use_gpu=False)
    easyocr_result = easyocr.process_image(test_image)
    
    # Process with PaddleOCR
    paddleocr = ocr_factory.create_engine("paddleocr", use_gpu=False)
    paddleocr_result = paddleocr.process_image(test_image)
    
    # Process with TrOCR
    trocr = ocr_factory.create_engine("trocr", use_gpu=False)
    trocr_result = trocr.process_image(test_image)
    
    # Process with HiOCR
    hiocr = ocr_factory.create_engine("hiocr", use_gpu=False)
    hiocr_result = hiocr.process_image(test_image)
    
    # Compare results
    assert easyocr_result['text'] != paddleocr_result['text']
    assert easyocr_result['text'] != trocr_result['text']
    assert easyocr_result['text'] != hiocr_result['text']

def test_gpu_switching(ocr_factory, test_image):
    """Test switching between GPU and CPU modes."""
    engines = ["easyocr", "paddleocr", "trocr", "hiocr"]
    for engine in engines:
        # CPU mode
        cpu_engine = ocr_factory.create_engine(engine, use_gpu=False)
        cpu_result = cpu_engine.process_image(test_image)
        
        # GPU mode (if available)
        gpu_engine = ocr_factory.create_engine(engine, use_gpu=True)
        gpu_result = gpu_engine.process_image(test_image)
        
        # Results should have same structure
        assert set(cpu_result.keys()) == set(gpu_result.keys())
        
        # Log performance comparison
        logger.info(f"{engine} performance comparison:")
        logger.info(f"CPU time: {cpu_result['time']}s")
        logger.info(f"GPU time: {gpu_result['time']}s")
        logger.info(f"CPU memory: {cpu_result['memory_used']}MB")
        logger.info(f"GPU memory: {gpu_result['memory_used']}MB") 