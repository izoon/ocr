from paddleocr import PaddleOCR
import os
import paddle
import sys
import cv2
import numpy as np

try:
    # Check CUDA availability
    if paddle.device.is_compiled_with_cuda():
        print("CUDA is available")
        paddle.device.set_device('gpu')
        # Set CUDA environment variables
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.8'
    else:
        print("CUDA is not available, falling back to CPU")
        paddle.device.set_device('cpu')

    # Initialize the OCR model with CPU configuration
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=False,  # Force CPU mode
        show_log=True,  # Show more detailed logs
        det_db_thresh=0.1,  # Lower threshold
        det_db_box_thresh=0.1,  # Lower threshold
        det_db_unclip_ratio=1.6,
        # Remove memory optimization parameters for CPU
        use_mp=False,
        total_process_num=1,
        use_fp16=False
    )

    # Path to your receipt image
    image_path = 'food.png'

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        exit(1)

    # Read and check image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'!")
        exit(1)
    
    print(f"Image shape: {img.shape}")
    print("Processing image...")
    
    # Run OCR with error handling
    results = ocr.ocr(image_path, cls=True)

    if results is None:
        print("OCR returned None results")
        exit(1)

    if len(results) == 0:
        print("No text was detected in the image")
        exit(0)

    # Print OCR results
    print("\n🧾 OCR Receipt Text:")
    for idx, line in enumerate(results[0]):
        if line is not None and len(line) >= 2:
            text = line[1][0]
            confidence = line[1][1]
            print(f"{idx+1}. Text: {text} (Confidence: {confidence:.2f})")
        else:
            print(f"{idx+1}. Invalid line format: {line}")

    # Save results to a text file
    with open("receipt_text.txt", "w") as f:
        for line in results[0]:
            if line is not None and len(line) >= 2:
                f.write(f"{line[1][0]}\t{line[1][1]:.2f}\n")
    print("\nResults have been saved to 'receipt_text.txt'")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("\nTroubleshooting steps:")
    print("1. Check if CUDA and cuDNN are properly installed")
    print("2. Try running with CPU mode by setting use_gpu=False")
    print("3. Check if the image file exists and is readable")
    print("4. Verify all dependencies are installed correctly")
    
    # Print system information for debugging
    print("\nSystem Information:")
    print(f"Python version: {sys.version}")
    print(f"PaddlePaddle version: {paddle.__version__}")
    print(f"CUDA available: {paddle.device.is_compiled_with_cuda()}")
    if paddle.device.is_compiled_with_cuda():
        print(f"Current device: {paddle.device.get_device()}")
        print(f"CUDA version: {paddle.version.cuda()}")
        print(f"cuDNN version: {paddle.version.cudnn()}")

