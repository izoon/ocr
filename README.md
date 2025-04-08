# OCR Research and Development Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GPU Support](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-orange)](https://developer.nvidia.com/cuda)

This repository contains research and development work on Optical Character Recognition (OCR) using various engines and approaches.

## 📁 Project Structure

```
.
├── ocr_service/           # REST API service for OCR
│   ├── api/              # FastAPI application
│   ├── services/         # OCR service implementations
│   ├── tests/           # Test suite
│   └── README.md        # Service-specific documentation
├── models/              # Trained models and weights
├── crops/              # Cropped text regions
├── utils/              # Utility scripts
├── tests/              # Test scripts
└── ocr_comparison*.py  # Comparison scripts
```

## 🚀 Components

### OCR Service
A REST API service that provides OCR capabilities using multiple engines:

| Engine | Description | Best For |
|--------|-------------|----------|
| EasyOCR | General purpose OCR with good balance of speed and accuracy | General text recognition |
| TrOCR | Specialized for printed text with high accuracy | Printed documents |
| PaddleOCR | High-performance OCR engine with GPU acceleration support | High-speed processing |

See [ocr_service/README.md](ocr_service/README.md) for detailed documentation.

### Comparison Scripts
- `ocr_comparison.py`: Basic comparison of OCR engines
- `ocr_comparison_4090_M4.py`: Optimized for NVIDIA RTX 4090
- `ocr_comparison_all.py`: Comprehensive comparison

### Training Scripts
- `train_trocr.py`: Script for training TrOCR models
- `prepare_training_data.py`: Data preparation utilities

## 🛠️ Getting Started

1. Set up the environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the OCR service:

**Basic development mode:**
```bash
cd ocr_service
uvicorn api.main:app --reload --port 8001
```

**Production mode with multiple workers:**
```bash
cd ocr_service
uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 4 \
    --timeout-keep-alive 30 \
    --log-level info \
    --access-log \
    --proxy-headers
```

### Common uvicorn options:
| Option | Description |
|--------|-------------|
| `--reload` | Enable auto-reload (development only) |
| `--host 0.0.0.0` | Listen on all interfaces |
| `--port 8001` | Port to listen on |
| `--workers 4` | Number of worker processes |
| `--timeout-keep-alive 30` | Keep-alive timeout |
| `--log-level info` | Logging level |
| `--access-log` | Enable access logging |
| `--proxy-headers` | Trust proxy headers |
| `--limit-concurrency 1000` | Max concurrent connections |
| `--backlog 2048` | Connection backlog size |

For production deployment, consider using:
- Process manager (e.g., Supervisor, systemd)
- Reverse proxy (e.g., Nginx)
- SSL/TLS termination
- Load balancing

See [uvicorn documentation](https://www.uvicorn.org/deployment/) for more options.

## 💻 Development

### Running Tests
```bash
pytest tests/
```

### Adding New OCR Engines
1. Create new service class in `ocr_service/services/ocr/`
2. Implement the `BaseOCR` interface
3. Add to the factory
4. Update API endpoints

## 📊 Research Notes

### Model Performance
| Engine | Performance Characteristics |
|--------|---------------------------|
| EasyOCR | Good balance of speed and accuracy |
| TrOCR | High accuracy for printed text |
| PaddleOCR | Best performance on NVIDIA GPUs, good accuracy on CPU |

### Platform Support
| Platform | Supported Engines |
|----------|------------------|
| macOS | EasyOCR, TrOCR (CPU/MPS), PaddleOCR (CPU) |
| NVIDIA GPU | All engines (PaddleOCR optimized for GPU) |
| CPU-only | EasyOCR, TrOCR, PaddleOCR |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License allows for:
- Commercial use
- Modification
- Distribution
- Private use
- Sublicensing

The only requirement is that the license and copyright notice must be included in all copies or substantial portions of the software.

## 🙏 Acknowledgments

- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [TrOCR](https://github.com/microsoft/unilm/tree/master/trocr)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 