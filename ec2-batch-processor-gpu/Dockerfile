FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY batch_processor.py .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "batch_processor.py"]
