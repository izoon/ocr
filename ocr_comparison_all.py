import time
import platform
import pytesseract
import easyocr
import cv2
import numpy as np
import os
from PIL import Image

# Only import PaddleOCR if not on Mac
if platform.system() != 'Darwin':
    from paddleocr import PaddleOCR

def run_paddleocr(image_path):
    print("\nRunning PaddleOCR...")
    start_time = time.time()
    
    # Initialize PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=True,
        show_log=False
    )
    
    # Run OCR
    results = ocr.ocr(image_path, cls=True)
    
    end_time = time.time()
    paddle_time = end_time - start_time
    
    # Extract text
    paddle_text = []
    if results and len(results) > 0:
        for line in results[0]:
            paddle_text.append(line[1][0])
    
    return paddle_text, paddle_time

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

def run_easyocr(image_path):
    print("\nRunning EasyOCR...")
    start_time = time.time()
    
    # Initialize EasyOCR with platform-specific settings
    if platform.system() == 'Darwin':  # Mac
        reader = easyocr.Reader(['en'], gpu=False)  # Use CPU on Mac
    else:  # Linux/Windows
        reader = easyocr.Reader(['en'], gpu=True)   # Use GPU on other platforms
    
    # Run OCR
    results = reader.readtext(image_path)
    
    end_time = time.time()
    easyocr_time = end_time - start_time
    
    # Extract text
    easyocr_text = [text for _, text, _ in results]
    
    return easyocr_text, easyocr_time, results

def compare_results(paddle_text, tesseract_text, easyocr_text, 
                   paddle_time, tesseract_time, easyocr_time,
                   results=None):
    print("\n=== OCR Comparison Results ===")
    print(f"Platform: {platform.system()} {platform.machine()}")
    
    print(f"\nProcessing Times:")
    if platform.system() != 'Darwin':  # Only show PaddleOCR time if not on Mac
        print(f"PaddleOCR: {paddle_time:.2f} seconds")
    print(f"Tesseract: {tesseract_time:.2f} seconds")
    print(f"EasyOCR: {easyocr_time:.2f} seconds")
    
    if platform.system() != 'Darwin':  # Only show PaddleOCR results if not on Mac
        print("\nPaddleOCR Results:")
        for i, text in enumerate(paddle_text, 1):
            print(f"{i}. {text}")
    
    print("\nTesseract Results:")
    for i, text in enumerate(tesseract_text, 1):
        print(f"{i}. {text}")
    
    print("\nEasyOCR Results:")
    for i, text in enumerate(easyocr_text, 1):
        print(f"{i}. {text}")
    
    # Compare number of lines detected
    print(f"\nNumber of lines detected:")
    if platform.system() != 'Darwin':  # Only show PaddleOCR count if not on Mac
        print(f"PaddleOCR: {len(paddle_text)}")
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
        # Initialize variables
        paddle_text, paddle_time = [], 0
        
        # Run OCR engines based on platform
        if platform.system() != 'Darwin':  # Only run PaddleOCR if not on Mac
            paddle_text, paddle_time = run_paddleocr(image_path)
        
        tesseract_text, tesseract_time = run_tesseract(image_path)
        easyocr_text, easyocr_time, results = run_easyocr(image_path)
        
        # Compare results
        compare_results(paddle_text, tesseract_text, easyocr_text,
                       paddle_time, tesseract_time, easyocr_time,
                       results)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check if all OCR engines are installed")
        print("2. Verify the image file exists and is readable")
        print("3. Check if CUDA is properly configured")
        print("4. Make sure all dependencies are installed")

if __name__ == "__main__":
    main() 