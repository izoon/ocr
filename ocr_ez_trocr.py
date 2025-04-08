import easyocr
import cv2
import os
import sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# --------------- SETTINGS --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_DIR = "crops"  # folder for cropped lines
os.makedirs(CROP_DIR, exist_ok=True)

# --------------- UTILITY FUNCTIONS --------------------
def crop_detected_lines(image_path, reader):
    img = cv2.imread(image_path)
    results = reader.readtext(image_path)
    cropped_paths = []

    for i, (bbox, _, _) in enumerate(results):
        x_min = int(min([p[0] for p in bbox]))
        y_min = int(min([p[1] for p in bbox]))
        x_max = int(max([p[0] for p in bbox]))
        y_max = int(max([p[1] for p in bbox]))

        # Handle any negative coordinates
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        crop = img[y_min:y_max, x_min:x_max]

        crop_path = os.path.join(CROP_DIR, f"line_{i}.png")
        cv2.imwrite(crop_path, crop)
        cropped_paths.append((i, crop_path, bbox))
    
    return cropped_paths

def read_with_trocr(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

# --------------- MAIN SCRIPT --------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python trocr_pipeline.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        sys.exit(1)

    print(f"📥 Running OCR pipeline on: {image_path}")

    print("🔍 Detecting text regions with EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    cropped_lines = crop_detected_lines(image_path, reader)
    
    print("🔠 Loading TrOCR model...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(DEVICE)

    print("🧠 Recognizing text with TrOCR:\n")
    for i, crop_path, bbox in cropped_lines:
        text = read_with_trocr(crop_path, processor, model)
        print(f"Line {i+1:02d}: {text}")

