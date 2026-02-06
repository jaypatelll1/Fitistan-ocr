FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies
RUN dnf install -y \
    mesa-libGL \
    mesa-libGLU \
    libgomp \
    && dnf clean all

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy function code
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Set the handler
CMD ["lambda_handler.lambda_handler"]
