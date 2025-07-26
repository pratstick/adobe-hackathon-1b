FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Sentence Transformer model during build
RUN pip install huggingface_hub && python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"tomaarsen/static-similarity-mrl-multilingual-v1\", local_dir=\"/app/models/static-similarity-mrl-multilingual-v1\")"

COPY . .

CMD ["python", "process_pdfs_1b.py"]