FROM python:3.10-slim

# Install system dependencies for PyMuPDF, PIL, YOLO, and general Python builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install all Python dependencies in one layer
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download HuggingFace model and keep only necessary files
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='tomaarsen/static-similarity-mrl-multilingual-v1', local_dir='/app/models/static-similarity-mrl-multilingual-v1')" && \
    # List files before cleanup for debugging
    ls -la /app/models/static-similarity-mrl-multilingual-v1/ && \
    # Keep only essential files: model.onnx and config files
    cd /app/models/static-similarity-mrl-multilingual-v1 && \
    find . -name "*.onnx" ! -name "model.onnx" -delete && \
    # Remove other large files we don't need
    rm -f model_bnb4.onnx model_fp16.onnx model_int8.onnx model_q4.onnx model_q4f16.onnx model_quantized.onnx model_uint8.onnx && \
    # List files after cleanup
    ls -la /app/models/static-similarity-mrl-multilingual-v1/ && \
    # Verify model can still be loaded after cleanup
    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('/app/models/static-similarity-mrl-multilingual-v1'); print('Model loaded successfully')"

# Copy all source code
COPY . .

# Ensure Python output is not buffered (better logging in Docker)
ENV PYTHONUNBUFFERED=1

# Use exec form for CMD for proper signal handling
CMD ["python", "process_pdfs_1b.py"]