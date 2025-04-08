import os
import json
import shutil
from PIL import Image
import random

def prepare_training_data(
    image_path: str,
    ocr_results_file: str,
    output_dir: str = "receipts",
    train_ratio: float = 0.8
):
    # Create directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Read OCR results
    with open(ocr_results_file, 'r') as f:
        results = json.load(f)
    
    # Extract text from EasyOCR (most accurate in our tests)
    text = " ".join(results['metrics']['easyocr']['text'])
    
    # Create annotation
    annotation = {
        "image": os.path.basename(image_path),
        "text": text
    }
    
    # Split into train/val
    if random.random() < train_ratio:
        # Copy image to train directory
        shutil.copy2(image_path, os.path.join(train_dir, os.path.basename(image_path)))
        # Save annotation to train.json
        train_json = os.path.join(output_dir, "train.json")
        if os.path.exists(train_json):
            with open(train_json, 'r') as f:
                train_data = json.load(f)
        else:
            train_data = []
        train_data.append(annotation)
        with open(train_json, 'w') as f:
            json.dump(train_data, f, indent=2)
    else:
        # Copy image to val directory
        shutil.copy2(image_path, os.path.join(val_dir, os.path.basename(image_path)))
        # Save annotation to val.json
        val_json = os.path.join(output_dir, "val.json")
        if os.path.exists(val_json):
            with open(val_json, 'r') as f:
                val_data = json.load(f)
        else:
            val_data = []
        val_data.append(annotation)
        with open(val_json, 'w') as f:
            json.dump(val_data, f, indent=2)
    
    print(f"Data prepared in {output_dir}")

if __name__ == "__main__":
    # Example usage
    prepare_training_data(
        image_path="food.png",
        ocr_results_file="ocr_results_Darwin_20250327_121618.json",
        output_dir="receipts"
    ) 