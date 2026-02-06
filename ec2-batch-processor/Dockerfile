FROM python:3.11-slim

# System dependencies (including build tools for insightface)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY batch_processor.py .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "batch_processor.py"]
