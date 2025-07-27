
FROM python:3.10-slim

# Install system dependencies for PyMuPDF, PIL, YOLO, and general Python builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install all Python dependencies in one layer (cache efficient)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models directory (if present)
COPY . .


# If the model is not present, download it using SentenceTransformer in Python
RUN pip install --no-cache-dir sentence-transformers && \
    if [ ! -d "/app/models/static-similarity-mrl-multilingual-v1" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/static-similarity-mrl-multilingual-v1'); model.save('/app/models/static-similarity-mrl-multilingual-v1')"; \
    fi

# Ensure Python output is not buffered (better logging in Docker)
ENV PYTHONUNBUFFERED=1

# Use exec form for CMD for proper signal handling
CMD ["python", "process_pdfs_1b.py"]