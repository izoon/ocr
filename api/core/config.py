from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "OCR Service"
    
    # OCR Settings
    OCR_MODELS_DIR: Path = Path("./models")
    MAX_IMAGE_SIZE: int = 2000
    SUPPORTED_IMAGE_TYPES: set = {"image/jpeg", "image/png", "image/tiff"}
    
    # GPU Settings
    USE_GPU: bool = False
    CUDA_DEVICE: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 