from setuptools import setup, find_packages

setup(
    name="ocr_service",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pillow",
        "easyocr",
        "pytesseract",
        "opencv-python",
        "torch",
        "transformers",
        "psutil",
    ],
    python_requires=">=3.8",
) 