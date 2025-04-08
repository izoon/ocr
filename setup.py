from setuptools import setup, find_packages

setup(
    name="ocr_service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.109.2",
        "uvicorn>=0.27.1",
        "python-multipart>=0.0.9",
        "pydantic>=2.6.1",
        "python-dotenv>=1.0.1",
        "easyocr>=1.7.1",
        "pytesseract>=0.3.10",
        "transformers>=4.37.2",
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "opencv-python-headless>=4.8.0.74",
        "numpy>=1.24.3",
        "pillow>=10.2.0",
    ],
    python_requires=">=3.11",
) 