# AWS Lambda Container Image for Bib Number Extraction
# Uses PaddleOCR with CPU support

FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies for PaddleOCR and build tools
RUN dnf install -y \
    mesa-libGL \
    mesa-libGLU \
    libgomp \
    gcc \
    gcc-c++ \
    make \
    && dnf clean all

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Pre-download PaddleOCR models to avoid cold start downloads
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ch')"

# Copy function code
COPY lambda_handler.py .

# Set the CMD to your handler
CMD [ "lambda_handler.lambda_handler" ]
