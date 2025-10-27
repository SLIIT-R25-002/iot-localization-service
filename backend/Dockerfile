# Multi-stage build for IoT Localization Service with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS base

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libc6 \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements_backend.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=superglue_backend.py
ENV PYTHONUNBUFFERED=1

# Add health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run the application
CMD ["python", "superglue_backend.py", "--host", "0.0.0.0", "--port", "5000", "--superglue", "outdoor", "--max_keypoints", "1024"]
