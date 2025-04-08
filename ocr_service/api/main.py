from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import List, Dict
import platform
from ..services.ocr.factory import OCRFactory
from .core.logging import logger
from .models.schemas import OCRResponse, EngineInfo
from .core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A service for optical character recognition using multiple engines",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def check_paddle_availability():
    """Check if PaddleOCR is available on the current platform."""
    if platform.system() == "Darwin":  # macOS
        logger.warning("PaddleOCR requested but not available on macOS/MPS platform")
        return False, "PaddleOCR is not available on macOS/MPS platform"
    logger.info("PaddleOCR is available on this platform")
    return True, None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    import psutil
    import time
    
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Get process metrics
    process = psutil.Process()
    process_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used": memory.used / 1024 / 1024,  # MB
            "disk_percent": disk.percent,
            "disk_free": disk.free / 1024 / 1024 / 1024  # GB
        },
        "process": {
            "memory_used": process_memory,
            "cpu_percent": process.cpu_percent(interval=0.1),
            "threads": process.num_threads(),
            "uptime": time.time() - process.create_time()
        }
    }

@app.get("/engines", response_model=Dict[str, EngineInfo])
async def list_engines():
    """List available OCR engines and their capabilities."""
    engines = {}
    for engine_type in ['easyocr', 'trocr', 'paddleocr', 'tesseract', 'hiocr']:
        try:
            engine = OCRFactory.create_engine(engine_type)
            engines[engine_type] = engine.get_engine_info()
        except Exception as e:
            logger.error(f"Failed to get info for {engine_type}: {str(e)}")
            engines[engine_type] = {
                'name': engine_type,
                'version': 'unknown',
                'device': 'unknown',
                'capabilities': [],
                'error': str(e)
            }
    return engines

@app.post(f"{settings.API_V1_STR}/ocr/{{engine_type}}", response_model=OCRResponse)
async def process_image(
    engine_type: str,
    file: UploadFile = File(...),
    use_gpu: bool = settings.USE_GPU,
    force_cpu: bool = False
):
    """Process an image using the specified OCR engine."""
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Check image size
        if max(image.size) > settings.MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Maximum dimension is {settings.MAX_IMAGE_SIZE}px"
            )
        
        # Create OCR engine
        engine = OCRFactory.create_engine(engine_type, use_gpu=use_gpu, force_cpu=force_cpu)
        
        # Process image
        logger.info(f"Processing image with {engine_type} engine")
        result = engine.process_image(image)
        
        return OCRResponse(
            text=result['text'],
            confidence=result['confidence'],
            processing_time=result['time'],
            memory_used=result['memory_used']
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/ocr/engines/{{engine}}", response_model=EngineInfo)
async def get_engine_info(engine: str):
    """
    Get information about a specific OCR engine.
    
    Args:
        engine: OCR engine name
        
    Returns:
        Engine information including version and capabilities
    """
    try:
        engine = engine.lower()  # Normalize engine name
        logger.info(f"Getting engine info for: {engine}")
        
        # Check PaddleOCR availability if requested
        if engine == "paddleocr":
            is_available, error_msg = check_paddle_availability()
            if not is_available:
                raise HTTPException(status_code=400, detail=error_msg)
        
        ocr_engine = OCRFactory.create_engine(engine)
        info = ocr_engine.get_engine_info()
        logger.info(f"Engine info retrieved: {info}")
        return info
    except Exception as e:
        logger.error(f"Error getting engine info for {engine}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 