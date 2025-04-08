FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    swig \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create models and logs directories
RUN mkdir -p models logs

# Expose port
EXPOSE 8001

# Run the application
CMD ["uvicorn", "ocr_service.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"] 