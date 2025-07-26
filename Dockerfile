# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the sentence transformer model
# This ensures the model is available offline
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('tomaarsen/static-similarity-mrl-multilingual-v1'); print('Model cached successfully')"

# Set environment variables for offline operation and headless mode
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV QT_QPA_PLATFORM=offscreen
ENV MPLBACKEND=Agg

# Copy the models directory
COPY models/ /app/models/

# Copy the application script
COPY process_pdfs_1b.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Command to run the application (example, adjust as needed)
# ENTRYPOINT ["python", "process_pdfs_1b.py"]
