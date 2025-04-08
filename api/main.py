from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import List
from api.models.schemas import OCRResponse, EngineInfo
from services.ocr.easyocr import EasyOCRService

app = FastAPI(
    title="OCR Service",
    description="A REST API service for OCR processing using multiple engines",
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

# Initialize OCR services
easyocr_service = EasyOCRService(use_gpu=False)  # Force CPU for now

@app.post("/ocr/easyocr", response_model=OCRResponse)
async def process_image_easyocr(file: UploadFile = File(...)):
    """
    Process an image using EasyOCR.
    
    Args:
        file: Image file to process
        
    Returns:
        OCR results including text, confidence scores, and performance metrics
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process image
        results = easyocr_service.process_image(image)
        
        return OCRResponse(
            text=results['text'],
            confidence=results['confidence'],
            processing_time=results['time'],
            memory_used=results['memory_used']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/engines/easyocr", response_model=EngineInfo)
async def get_easyocr_info():
    """Get information about the EasyOCR engine."""
    return easyocr_service.get_engine_info()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 