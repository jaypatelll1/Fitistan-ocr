FROM amazon/aws-lambda-python:3.12

# Install system dependencies and build tools
RUN dnf install -y \
    mesa-libGL \
    mesa-libGLU \
    libgomp \
    gcc \
    gcc-c++ \
    make \
    && dnf clean all

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Pre-download models to avoid cold start downloads
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ch', show_log=False)"

COPY lambda_handler.py .

CMD ["lambda_handler.lambda_handler"]
