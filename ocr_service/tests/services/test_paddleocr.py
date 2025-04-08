import pytest
import os
import platform
import paddle
from PIL import Image
from ocr_service.services.ocr.paddleocr import PaddleOCRService
from api.core.logging import logger

# Skip all tests on macOS
pytestmark = pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="PaddleOCR tests are skipped on macOS due to compatibility issues"
)

@pytest.fixture
def test_image():
    """Load test image."""
    # Get the absolute path to the test data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "..", "data", "food.png")
    logger.info(f"Loading test image from: {image_path}")
    return Image.open(image_path)

@pytest.fixture
def paddle_service():
    """Create PaddleOCR service instance."""
    return PaddleOCRService(use_gpu=True)

def test_gpu_availability():
    """Test GPU availability detection."""
    # Check if CUDA is compiled
    cuda_available = paddle.device.is_compiled_with_cuda()
    logger.info(f"CUDA available: {cuda_available}")
    
    # Check platform
    is_macos = platform.system() == "Darwin"
    logger.info(f"Platform: {platform.system()}")
    
    # Create service with GPU
    service = PaddleOCRService(use_gpu=True)
    
    # Verify GPU usage
    if cuda_available and not is_macos:
        assert service.use_gpu is True
        assert paddle.device.get_device() == 'gpu'
    else:
        assert service.use_gpu is False
        assert paddle.device.get_device() == 'cpu'

def test_paddleocr_init():
    """Test PaddleOCR initialization."""
    # Test GPU initialization
    service_gpu = PaddleOCRService(use_gpu=True)
    assert isinstance(service_gpu, PaddleOCRService)
    
    # Test CPU initialization
    service_cpu = PaddleOCRService(use_gpu=False)
    assert isinstance(service_cpu, PaddleOCRService)

def test_process_image(paddle_service, test_image):
    """Test image processing with PaddleOCR."""
    # Skip on macOS
    if platform.system() == "Darwin":
        pytest.skip("PaddleOCR image processing is not supported on macOS")
    
    # Process image
    result = paddle_service.process_image(test_image)
    
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

def test_platform_compatibility():
    """Test platform-specific behavior."""
    is_macos = platform.system() == "Darwin"
    cuda_available = paddle.device.is_compiled_with_cuda()
    
    service = PaddleOCRService(use_gpu=True)
    
    if is_macos:
        assert service.use_gpu is False
        assert paddle.device.get_device() == 'cpu'
    elif cuda_available:
        assert service.use_gpu is True
        assert paddle.device.get_device() == 'gpu'
    else:
        assert service.use_gpu is False
        assert paddle.device.get_device() == 'cpu' 