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

# Download HuggingFace model after requirements
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='tomaarsen/static-similarity-mrl-multilingual-v1', local_dir='/app/models/static-similarity-mrl-multilingual-v1')"

# Copy all source code
COPY . .

# Ensure Python output is not buffered (better logging in Docker)
ENV PYTHONUNBUFFERED=1

# Use exec form for CMD for proper signal handling
CMD ["python", "process_pdfs_1b.py"]