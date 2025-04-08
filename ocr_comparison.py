import time
import platform
import pytesseract
import easyocr
import cv2
import numpy as np
import os
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def get_device_info():
    device_info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'cuda_available': torch.cuda.is_available(),
        'device_name': None,
        'compute_capability': None
    }
    
    if device_info['cuda_available']:
        device_info['device_name'] = torch.cuda.get_device_name(0)
        device_info['compute_capability'] = torch.cuda.get_device_capability()
    
    return device_info

def run_trocr(image_path, device_info):
    print("\nRunning Microsoft TrOCR...")
    start_time = time.time()
    
    # Load model and processor
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    
    # Set device based on platform
    if device_info['cuda_available']:
        model = model.to('cuda')
        print(f"Using CUDA: {device_info['device_name']}")
    else:
        print("Using CPU")
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    # Use Resampling.LANCZOS instead of ANTIALIAS
    if hasattr(Image, 'Resampling'):
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
    else:
        image = image.resize((224, 224), Image.LANCZOS)
    
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    # Move input to appropriate device
    if device_info['cuda_available']:
        pixel_values = pixel_values.to('cuda')
    
    # Generate text
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    end_time = time.time()
    trocr_time = end_time - start_time
    
    return [generated_text], trocr_time

def run_easyocr(image_path, device_info):
    print("\nRunning EasyOCR...")
    start_time = time.time()
    
    # Initialize EasyOCR with platform-specific settings
    if device_info['platform'] == 'Darwin':  # Mac
        reader = easyocr.Reader(['en'], gpu=True)  # Use Metal acceleration on Mac
        print(f"Using Metal: {device_info['device_name']}")
    else:  # Linux/Windows
        reader = easyocr.Reader(['en'], gpu=device_info['cuda_available'])
        if device_info['cuda_available']:
            print(f"Using CUDA: {device_info['device_name']}")
        else:
            print("Using CPU")
    
    # Run OCR
    results = reader.readtext(image_path)
    
    end_time = time.time()
    easyocr_time = end_time - start_time
    
    # Extract text
    easyocr_text = [text for _, text, _ in results]
    
    return easyocr_text, easyocr_time, results

def run_tesseract(image_path):
    print("\nRunning Tesseract...")
    start_time = time.time()
    
    # Read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Run OCR
    text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
    
    end_time = time.time()
    tesseract_time = end_time - start_time
    
    # Split into lines
    tesseract_text = [line.strip() for line in text.split('\n') if line.strip()]
    
    return tesseract_text, tesseract_time

def compare_results(trocr_text, tesseract_text, easyocr_text, 
                   trocr_time, tesseract_time, easyocr_time,
                   device_info, results=None):
    print("\n=== OCR Comparison Results ===")
    print(f"Platform: {device_info['platform']} {device_info['machine']}")
    if device_info['cuda_available']:
        print(f"GPU: {device_info['device_name']}")
        print(f"CUDA Compute Capability: {device_info['compute_capability']}")
    else:
        print("GPU: Not available (using CPU)")
    
    print(f"\nProcessing Times:")
    print(f"TrOCR: {trocr_time:.2f} seconds")
    print(f"Tesseract: {tesseract_time:.2f} seconds")
    print(f"EasyOCR: {easyocr_time:.2f} seconds")
    
    print("\nTrOCR Results:")
    for i, text in enumerate(trocr_text, 1):
        print(f"{i}. {text}")
    
    print("\nTesseract Results:")
    for i, text in enumerate(tesseract_text, 1):
        print(f"{i}. {text}")
    
    print("\nEasyOCR Results:")
    for i, text in enumerate(easyocr_text, 1):
        print(f"{i}. {text}")
    
    # Compare number of lines detected
    print(f"\nNumber of lines detected:")
    print(f"TrOCR: {len(trocr_text)}")
    print(f"Tesseract: {len(tesseract_text)}")
    print(f"EasyOCR: {len(easyocr_text)}")
    
    # Calculate average confidence for EasyOCR
    if results:
        confidences = [conf for _, _, conf in results]
        avg_confidence = sum(confidences) / len(confidences)
        print(f"\nEasyOCR Average Confidence: {avg_confidence:.2%}")

def main():
    # Path to your receipt image
    image_path = 'food.png'
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    try:
        # Get device information
        device_info = get_device_info()
        
        # Run OCR engines
        trocr_text, trocr_time = run_trocr(image_path, device_info)
        tesseract_text, tesseract_time = run_tesseract(image_path)
        easyocr_text, easyocr_time, results = run_easyocr(image_path, device_info)
        
        # Compare results
        compare_results(trocr_text, tesseract_text, easyocr_text,
                       trocr_time, tesseract_time, easyocr_time,
                       device_info, results)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check if all OCR engines are installed")
        print("2. Verify the image file exists and is readable")
        print("3. Check if CUDA is properly configured")
        print("4. Make sure all dependencies are installed")

if __name__ == "__main__":
    main() 