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
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import json
from datetime import datetime

# Import PaddleOCR if not on Mac
if platform.system() != 'Darwin':
    from paddleocr import PaddleOCR

class OCRBenchmark:
    def __init__(self):
        self.device_info = self.get_device_info()
        self.results = {
            'device': self.device_info,
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
    
    def get_device_info(self):
        device_info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'cuda_available': torch.cuda.is_available(),
            'device_name': None,
            'compute_capability': None,
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
        
        if device_info['cuda_available']:
            device_info['device_name'] = torch.cuda.get_device_name(0)
            device_info['compute_capability'] = torch.cuda.get_device_capability()
            device_info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        elif device_info['platform'] == 'Darwin':
            device_info['device_name'] = 'Apple Silicon'
            device_info['compute_capability'] = 'Metal'
            device_info['mps_available'] = torch.backends.mps.is_available()
        
        return device_info
    
    def measure_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**2)  # MB
    
    def run_paddleocr(self, image_path):
        print("\nRunning PaddleOCR...")
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        # Initialize PaddleOCR with platform-specific settings
        if self.device_info['cuda_available']:
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=True,
                show_log=False
            )
            print(f"Using CUDA: {self.device_info['device_name']}")
        else:
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                show_log=False
            )
            print("Using CPU")
        
        # Run OCR
        results = ocr.ocr(image_path, cls=True)
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        # Extract text and confidence scores
        text_results = []
        confidences = []
        if results and len(results) > 0:
            for line in results[0]:
                text_results.append(line[1][0])
                confidences.append(line[1][1])
        
        return {
            'text': text_results,
            'time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'confidences': confidences
        }
    
    def run_easyocr(self, image_path):
        print("\nRunning EasyOCR...")
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        # Print device information
        print("Using CPU for EasyOCR")
        
        # Initialize EasyOCR with CPU support
        reader = easyocr.Reader(
            ['en'],
            gpu=False,  # Force CPU usage
            model_storage_directory='./models',
            download_enabled=True
        )
        
        # Run OCR
        results = reader.readtext(image_path)
        
        # Extract text and confidence scores
        text_results = []
        confidences = []
        for _, text, conf in results:
            text_results.append(text)
            confidences.append(conf)
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        return {
            'text': text_results,
            'time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'confidences': confidences
        }
    
    def run_trocr(self, image_path):
        print("\nRunning Microsoft TrOCR...")
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        # Load model and processor
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        
        # Set device based on platform
        if self.device_info['cuda_available']:
            model = model.to('cuda')
            print(f"Using CUDA: {self.device_info['device_name']}")
        elif self.device_info['platform'] == 'Darwin' and self.device_info['mps_available']:
            model = model.to('mps')
            print(f"Using MPS: {self.device_info['device_name']}")
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
        if self.device_info['cuda_available']:
            pixel_values = pixel_values.to('cuda')
        elif self.device_info['platform'] == 'Darwin' and self.device_info['mps_available']:
            pixel_values = pixel_values.to('mps')
        
        # Generate text
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        return {
            'text': [generated_text],
            'time': end_time - start_time,
            'memory_used': end_memory - start_memory
        }
    
    def run_tesseract(self, image_path):
        print("\nRunning Tesseract...")
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        # Set device-specific configuration
        if self.device_info['cuda_available']:
            print(f"Using CUDA: {self.device_info['device_name']}")
            # Use OpenCL for GPU acceleration on NVIDIA
            config = '--oem 3 --psm 6 -l eng'
        else:
            print("Using CPU")
            config = '--oem 3 --psm 6 -l eng'
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            
            # Run OCR
            text = pytesseract.image_to_string(image, config=config)
            
            # Split into lines and remove empty lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            end_time = time.time()
            end_memory = self.measure_memory_usage()
            
            return {
                'text': lines,
                'time': end_time - start_time,
                'memory_used': end_memory - start_memory
            }
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
    
    def plot_results(self):
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Get engines based on platform
        engines = ['tesseract', 'easyocr']
        if self.device_info['platform'] != 'Darwin':
            engines.insert(0, 'paddleocr')
            engines.insert(1, 'trocr')
        
        # Plot processing times
        times = [self.results['metrics'][engine]['time'] for engine in engines]
        engine_names = [engine.capitalize() for engine in engines]
        
        sns.barplot(x=engine_names, y=times, ax=ax1)
        ax1.set_title('Processing Time (seconds)')
        ax1.set_ylabel('Time (s)')
        
        # Plot memory usage
        memory = [self.results['metrics'][engine]['memory_used'] for engine in engines]
        
        sns.barplot(x=engine_names, y=memory, ax=ax2)
        ax2.set_title('Memory Usage (MB)')
        ax2.set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig(f'ocr_comparison_{self.device_info["platform"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()
    
    def save_results(self):
        # Save results to JSON file
        filename = f'ocr_results_{self.device_info["platform"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\nResults saved to {filename}")
    
    def run_benchmark(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found!")
            return
        
        try:
            print("\n=== OCR Benchmark Results ===")
            print(f"Platform: {self.device_info['platform']} {self.device_info['machine']}")
            if self.device_info['cuda_available']:
                print(f"GPU: {self.device_info['device_name']}")
                print(f"CUDA Compute Capability: {self.device_info['compute_capability']}")
                print(f"GPU Memory: {self.device_info['gpu_memory_total']:.2f} GB")
            elif self.device_info['platform'] == 'Darwin':
                print(f"GPU: {self.device_info['device_name']}")
                print(f"Compute Framework: {self.device_info['compute_capability']}")
            else:
                print("GPU: Not available (using CPU)")
            print(f"CPU Cores: {self.device_info['cpu_count']}")
            print(f"CPU Frequency: {self.device_info['cpu_freq']:.2f} MHz")
            print(f"System Memory: {self.device_info['memory_total']:.2f} GB")
            
            # Run OCR engines based on platform
            if self.device_info['platform'] != 'Darwin':
                self.results['metrics']['paddleocr'] = self.run_paddleocr(image_path)
            
            # Run TrOCR on both platforms
            self.results['metrics']['trocr'] = self.run_trocr(image_path)
            self.results['metrics']['tesseract'] = self.run_tesseract(image_path)
            self.results['metrics']['easyocr'] = self.run_easyocr(image_path)
            
            # Print results
            print("\nProcessing Times:")
            for engine, metrics in self.results['metrics'].items():
                print(f"{engine}: {metrics['time']:.2f} seconds")
                print(f"Memory Used: {metrics['memory_used']:.2f} MB")
                if 'confidences' in metrics:
                    avg_conf = sum(metrics['confidences']) / len(metrics['confidences'])
                    print(f"Average Confidence: {avg_conf:.2%}")
            
            # Plot and save results
            self.plot_results()
            self.save_results()
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Check if all OCR engines are installed")
            print("2. Verify the image file exists and is readable")
            print("3. Check if CUDA is properly configured")
            print("4. Make sure all dependencies are installed")

def main():
    # Path to your receipt image
    image_path = 'food.png'
    
    # Run benchmark
    benchmark = OCRBenchmark()
    benchmark.run_benchmark(image_path)

if __name__ == "__main__":
    main() 