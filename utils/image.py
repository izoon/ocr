from PIL import Image
import io
from typing import Tuple, Optional
from api.core.config import settings
from api.core.logging import logger

def validate_image(image: Image.Image) -> bool:
    """Validate image format and size."""
    if image.format.lower() not in ['jpeg', 'png', 'tiff']:
        logger.error(f"Unsupported image format: {image.format}")
        return False
    
    if max(image.size) > settings.MAX_IMAGE_SIZE:
        logger.warning(f"Image size {image.size} exceeds maximum allowed size")
        return False
    
    return True

def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image for OCR."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if image is too large
    if max(image.size) > settings.MAX_IMAGE_SIZE:
        ratio = settings.MAX_IMAGE_SIZE / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        logger.info(f"Resized image to {new_size}")
    
    return image

def get_image_info(image: Image.Image) -> dict:
    """Get image information."""
    return {
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "width": image.width,
        "height": image.height
    }

def bytes_to_image(image_bytes: bytes) -> Optional[Image.Image]:
    """Convert bytes to PIL Image."""
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Error converting bytes to image: {str(e)}")
        return None 