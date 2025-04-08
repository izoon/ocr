# OCR Service

A REST API service for OCR processing using multiple engines (EasyOCR, TrOCR, PaddleOCR, Tesseract).

## Features

- Multiple OCR engines:
  - EasyOCR (default)
  - TrOCR (Transformer-based)
  - PaddleOCR (NVIDIA GPU only, not supported on macOS)
  - Tesseract (fallback)
  - HiOCR (development)
- Fallback mechanisms for text detection and recognition
- Performance metrics and logging
- Docker support
- Health monitoring
- Platform-specific optimizations (CUDA, MPS, CPU)

## Models

### Text Detection
- CRAFT (Character Region Awareness for Text Detection)
- Fallback: Tesseract's text detection

### Text Recognition
- EasyOCR: english_g2.pth
- TrOCR: microsoft/trocr-base-printed
- PaddleOCR: PP-OCRv4
- Fallback: Tesseract

## API Endpoints

### OCR Processing
- `POST /api/v1/ocr/{engine}` - Process image with specified engine
  - Supported engines: `easyocr`, `trocr`, `paddleocr`, `tesseract`, `hiocr`
  - Content-Type: multipart/form-data
  - Parameters:
    - `file` (image file)
    - `use_gpu` (boolean, optional): Enable GPU processing (default: false)
    - `force_cpu` (boolean, optional): Force CPU usage even if GPU is available (default: false)

### Engine Information
- `GET /api/v1/ocr/engines/{engine}` - Get information about a specific OCR engine
  - Returns: Engine name, version, device, and capabilities

### Health Check
- `GET /health` - Service health status
- `GET /metrics` - Performance metrics

## Installation

### Prerequisites
- Python 3.11
- CUDA 12.4 (for GPU support)
- Tesseract OCR

### System Dependencies (Linux)
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr swig libmupdf-dev
```

### Python Dependencies
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements_4090.txt  # For NVIDIA 4090
# or
pip install -r requirements.txt       # For other systems
```

## Running the Service

### Development Mode
```bash
# Run from project root directory
cd /path/to/ocr_rd
uvicorn ocr_service.api.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --reload \
    --log-level info
```

### Production Mode
```bash
# Run from project root directory
cd /path/to/ocr_rd
uvicorn ocr_service.api.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 4 \
    --timeout-keep-alive 30 \
    --log-level info \
    --access-log \
    --proxy-headers
```

## Testing the API

### Health Check
```bash
curl http://localhost:8001/health
```

### OCR Processing

#### EasyOCR (default)
```bash
curl -X POST http://localhost:8001/api/v1/ocr/easyocr \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/your/image.png" | jq
```

#### TrOCR
```bash
curl -X POST http://localhost:8001/api/v1/ocr/trocr \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/your/image.png" | jq
```

#### PaddleOCR (NVIDIA GPU only, not supported on macOS)
```bash
curl -X POST http://localhost:8001/api/v1/ocr/paddleocr \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/your/image.png" | jq
```

#### Tesseract
```bash
curl -X POST http://localhost:8001/api/v1/ocr/tesseract \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/your/image.png" | jq
```

#### HiOCR (Development)
```bash
curl -X POST http://localhost:8001/api/v1/ocr/hiocr \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/your/image.png" | jq
```

### Force CPU Usage
```bash
curl -X POST http://localhost:8001/api/v1/ocr/easyocr?force_cpu=true \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/your/image.png" | jq
```

### Get Engine Information
```bash
curl http://localhost:8001/api/v1/ocr/engines/easyocr | jq
```

## Docker Support

### Building the Image
```bash
docker build -t ocr_service .
```

### Running the Container
```bash
docker run -d \
    -p 8001:8001 \
    -v /path/to/models:/app/models \
    -v /path/to/logs:/app/logs \
    --gpus all \
    ocr_service
```

## Response Format

### OCR Processing Response
```json
{
    "text": ["extracted text line 1", "extracted text line 2"],
    "confidence": [0.95, 0.87],
    "processing_time": 0.5,
    "memory_used": 1024
}
```

### Engine Information Response
```json
{
    "name": "EasyOCR",
    "version": "1.7.1",
    "device": "cuda",
    "capabilities": ["text_detection", "text_recognition", "multiple_languages"]
}
```

## Platform Compatibility

- **EasyOCR**: Supports CPU, CUDA (NVIDIA), and MPS (Apple Silicon)
- **TrOCR**: Supports CPU, CUDA (NVIDIA), and MPS (Apple Silicon)
- **PaddleOCR**: Supports CPU and CUDA (NVIDIA), but not supported on macOS
- **Tesseract**: CPU only
- **HiOCR**: Inherits compatibility from PaddleOCR

## Troubleshooting

### Common Issues
1. GPU not detected
   - Check CUDA installation
   - Verify GPU drivers
   - Check PyTorch CUDA support
   - Try using `force_cpu=true` parameter

2. Model download failures
   - Check internet connection
   - Verify disk space
   - Check model directory permissions

3. Memory issues
   - Reduce image size
   - Use CPU mode with `force_cpu=true`
   - Adjust batch size

4. PaddleOCR on macOS
   - PaddleOCR is not supported on macOS
   - Use EasyOCR or TrOCR instead

### Logs
- Application logs: `logs/app.log`
- Access logs: `logs/access.log`
- Error logs: `logs/error.log`

## Minimum System Requirements

### CPU Mode
- Python 3.11
- 8GB RAM
- 2GB free disk space

### GPU Mode (NVIDIA)
- CUDA 12.4
- 16GB RAM
- 4GB free disk space
- NVIDIA GPU with 8GB+ VRAM

## License

MIT License 