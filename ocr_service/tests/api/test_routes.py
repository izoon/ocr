import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np
from ocr_service.api.main import app
from ocr_service.api.core.config import settings

client = TestClient(app)

def create_test_image():
    """Create a test image with text."""
    # Create a white image with black text
    img = Image.new('RGB', (100, 30), color='white')
    return img

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_get_engine_info():
    """Test engine info endpoints for all OCR engines."""
    engines = ["easyocr", "paddleocr", "trocr", "hiocr"]
    for engine in engines:
        response = client.get(f"{settings.API_V1_STR}/ocr/engines/{engine}")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "device" in data
        assert "capabilities" in data

def test_process_image_all_engines():
    """Test image processing endpoints for all OCR engines."""
    # Create a test image
    img = create_test_image()
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Test all OCR engines
    engines = ["easyocr", "paddleocr", "trocr", "hiocr"]
    for engine in engines:
        # Create test file
        files = {"file": ("test.png", img_byte_arr, "image/png")}
        
        # Send request
        response = client.post(f"{settings.API_V1_STR}/ocr/{engine}", files=files)
        assert response.status_code == 200
        
        # Check response structure
        data = response.json()
        assert "text" in data
        assert "confidence" in data
        assert "processing_time" in data
        assert "memory_used" in data

def test_invalid_engine():
    """Test invalid engine type."""
    response = client.get(f"{settings.API_V1_STR}/ocr/engines/invalid_engine")
    assert response.status_code == 404

def test_invalid_image():
    """Test invalid image upload."""
    # Create invalid file
    files = {"file": ("test.txt", b"invalid data", "text/plain")}
    
    # Test all engines
    engines = ["easyocr", "paddleocr", "trocr", "hiocr"]
    for engine in engines:
        response = client.post(f"{settings.API_V1_STR}/ocr/{engine}", files=files)
        assert response.status_code == 400

def test_missing_file():
    """Test missing file in request."""
    # Test all engines
    engines = ["easyocr", "paddleocr", "trocr", "hiocr"]
    for engine in engines:
        response = client.post(f"{settings.API_V1_STR}/ocr/{engine}", files={})
        assert response.status_code == 400

def test_force_cpu_option():
    """Test force_cpu option for all engines."""
    # Create a test image
    img = create_test_image()
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Test all OCR engines with force_cpu=true
    engines = ["easyocr", "paddleocr", "trocr", "hiocr"]
    for engine in engines:
        # Create test file
        files = {"file": ("test.png", img_byte_arr, "image/png")}
        
        # Send request with force_cpu=true
        response = client.post(
            f"{settings.API_V1_STR}/ocr/{engine}?force_cpu=true",
            files=files
        )
        assert response.status_code == 200
        
        # Check response structure
        data = response.json()
        assert "text" in data
        assert "confidence" in data
        assert "processing_time" in data
        assert "memory_used" in data 