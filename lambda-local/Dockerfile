# Local Docker Test for Bib Number Extraction
# Uses PaddleOCR with CPU support

FROM python:3.11-slim

# Install system dependencies for PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Set environment variable
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Copy function code
COPY lambda_handler.py .

# Default command
ENTRYPOINT ["python", "lambda_handler.py"]
