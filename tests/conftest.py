import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np
from ocr_service.api.main import app

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def test_image():
    """Create a test image."""
    # Create a white image with black text
    img = Image.new('RGB', (100, 30), color='white')
    return img

@pytest.fixture
def test_image_bytes(test_image):
    """Convert test image to bytes."""
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

@pytest.fixture
def mock_easyocr_response():
    """Create a mock EasyOCR response."""
    return {
        "text": ["Test Text"],
        "confidence": [0.95],
        "time": 0.1,
        "memory_used": 100.0
    } 